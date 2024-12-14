import re

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import jaxutils
from . import ninjax as nj

cast = jaxutils.cast_to_compute

# ==========================
# Mamba Model Implementation
# ==========================
from dataclasses import dataclass
from functools import partial
import math
import jax.nn.initializers as init
from flax import linen as nn


def masked_mean(x, mask):
    # x: [B,T], mask: [B,T], True or False (True: keep, False: mask out)
    mask_f = mask.astype(f32)
    denom = jnp.sum(mask_f)
    return jnp.sum(x * mask_f) / jnp.maximum(denom, 1e-8)

@dataclass
class MambaArgs:
    d_model: int
    n_layer: int
    d_state: int = 16
    expand: int = 2
    dt_rank: str = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5
    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', init.ones, (self.d_model,))
        normed = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return normed * weight

class MambaBlock(nn.Module):
    args: MambaArgs
    def setup(self):
        self.in_proj = nn.Dense(features=self.args.d_model*self.args.expand*2,
                                kernel_init=init.normal(),
                                use_bias=self.args.bias)
        self.conv1d = nn.Conv(features=self.args.d_model*self.args.expand,
                              kernel_size=[self.args.d_conv],
                              feature_group_count=self.args.d_model*self.args.expand,
                              padding='VALID',
                              use_bias=self.args.conv_bias,
                              kernel_init=init.normal())
        self.x_proj = nn.Dense(self.args.dt_rank + self.args.d_state * 2, use_bias=False)
        self.dt_proj = nn.Dense(self.args.d_model*self.args.expand, use_bias=True)
        d_in = self.args.d_model*self.args.expand
        self.A_log = self.param('A_log', lambda rng, shape: jnp.log(jnp.tile(jnp.arange(1, self.args.d_state+1), (d_in,1))), (d_in,self.args.d_state))
        self.D = self.param('D', init.ones, (d_in,))
        self.out_proj = nn.Dense(self.args.d_model, kernel_init=init.normal(), use_bias=self.args.bias)

    def ssm(self, x):
        b, l, d_in = x.shape
        A = -jnp.exp(self.A_log)
        D = self.D
        x_dbl = self.x_proj(x)
        delta, B, C = jnp.split(x_dbl, [self.args.dt_rank, self.args.dt_rank+self.args.d_state], axis=-1)
        delta = jax.nn.softplus(self.dt_proj(delta))

        # 아래 구현은 시퀀스 길이 l에 대해 O(l)로 순차적이나 jax jit로 벡터화 가능
        n = self.args.d_state
        # 타일링
        B_tiled = jnp.expand_dims(B,2) # [B,T,1,d_state]
        C_tiled = jnp.expand_dims(C,2) # [B,T,1,d_state]
        B_tiled = jnp.repeat(B_tiled, d_in, axis=-2)
        C_tiled = jnp.repeat(C_tiled, d_in, axis=-2)

        u = x
        deltaA = jnp.exp(jnp.einsum('btd,dn->btdn', delta, A))
        # 상태 추적
        x_state = jnp.zeros((b, d_in, n))
        ys = []
        for i in range(l):
            x_state = deltaA[:, i] * x_state + (delta[:, i, :, None] * (B_tiled[:, i] * u[:, i, :, None]))
            y_t = jnp.einsum('bdn,bdn->bd', x_state, C_tiled[:,i])
            y_t = y_t + u[:, i, :]*D
            ys.append(y_t)
        y = jnp.stack(ys, axis=1)
        return y

    @nn.compact
    def __call__(self, x):
        B,L,D = x.shape
        pad_len = self.args.d_conv - 1
        x_padded = jnp.pad(x, ((0,0),(pad_len,0),(0,0)))
        x_and_res = self.in_proj(x)
        d_in = self.args.d_model*self.args.expand
        x_, res = jnp.split(x_and_res, [d_in], axis=-1)
        x_conv = self.conv1d(x_padded)
        x_conv = x_conv[:,:L,:]
        x_conv = jax.nn.silu(x_conv)
        y = self.ssm(x_conv)
        y = y * jax.nn.silu(res)
        return self.out_proj(y)

class ResidualBlock(nn.Module):
    args: MambaArgs
    def setup(self):
        self.mixer = MambaBlock(self.args)
        self.norm = RMSNorm(self.args.d_model)
    def __call__(self, x):
        return self.mixer(self.norm(x)) + x

class MambaModel(nn.Module):
    args: MambaArgs
    def setup(self):
        self.layers = [ResidualBlock(self.args) for _ in range(self.args.n_layer)]
        self.norm_f = RMSNorm(self.args.d_model)
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x

# ==========================
# MambaSSM for World Model (Replacing RSSM)
# ==========================
class MambaSSM(nj.Module):
    def __init__(
            self,
            d_model=1024,
            n_layer=4,
            d_state=16,
            expand=2,
            dt_rank='auto',
            d_conv=4,
            conv_bias=True,
            bias=False,
            stoch=32,
            classes=32,
            unimix=0.01,
            **kw,
    ):
        self._stoch = stoch
        self._classes = classes
        self._unimix = unimix
        self.args = MambaArgs(
            d_model=d_model,
            n_layer=n_layer,
            d_state=d_state,
            expand=expand,
            dt_rank=dt_rank,
            d_conv=d_conv,
            conv_bias=conv_bias,
            bias=bias
        )

    def initial(self, batch_size):
        # 초기 h는 단순히 zeros
        return cast(dict(deter=jnp.zeros([batch_size, self.args.d_model], f32)))

    def observe(self, prev_state, actions, embed, is_first, mask=None):
        """
        observe 단계:
        q(z_t|x_t)는 외부 encoder로부터 얻어서 embed=z_t로 전달받는다고 가정.
        z_t: [B,T,stoch,classes]
        actions: [B,T, ...]
        is_first: [B,T]
        """
        B,T = embed.shape[:2]
        z_shp = embed.shape
        assert z_shp[2:] == (self._stoch,self._classes), z_shp
        z_input = embed.reshape(B,T,self._stoch*self._classes)
        a_flat = jaxutils.concat_dict(actions) # [B,T,action_dim]

        # Mamba: h_t = f_Mamba(z_{1:t-1}, a_{1:t-1})
        z_in = z_input[:,:-1] if T>1 else jnp.zeros((B,0,self._stoch*self._classes), f32)
        a_in = a_flat[:,:-1] if T>1 else jnp.zeros((B,0,a_flat.shape[-1]), f32)
        mamba_in = jnp.concatenate([z_in, a_in], axis=-1) # [B,T-1,Zdim+Adim]

        # projection to d_model
        proj = self.get("mamba_in_proj", Linear, self.args.d_model)(mamba_in)
        h_seq = self.get("mamba_core", MambaModel, self.args)(proj) # [B,T-1,d_model]

        h0 = jnp.zeros((B,1,self.args.d_model), f32)
        h_full = jnp.concatenate([h0,h_seq], axis=1) # [B,T,d_model]
        h_full = jaxutils.reset(h_full, is_first)

        states = dict(deter=h_full)
        return states

    def imagine(self, state, actions):
        """
        imagination 단계:
        여기서도 동일하게 z를 prior로 sample하여 주입할 수 있음.
        만약 z가 아직 없으면 dummy z를 사용 후 상위 레벨에서 iterative하게 z sample.
        """
        B = actions[list(actions.keys())[0]].shape[0]
        T = actions[list(actions.keys())[0]].shape[1]+1
        # 여기서는 dummy z 사용 (실제론 prior에서 샘플한 z를 step by step반영)
        z_dummy = jnp.zeros((B,T,self._stoch*self._classes), f32)
        a_flat = jaxutils.concat_dict(actions) # [B,T-1,adim]

        z_in = z_dummy[:,:-1]
        a_in = a_flat
        mamba_in = jnp.concatenate([z_in, a_in], axis=-1)
        proj = self.get("mamba_imagine_in_proj", Linear, self.args.d_model)(mamba_in)
        h_seq = self.get("mamba_imagine_core", MambaModel, self.args)(proj)
        h0 = jnp.zeros((B,1,self.args.d_model), f32)
        h_full = jnp.concatenate([h0,h_seq], axis=1)
        states = dict(deter=h_full)
        return states

    def _prior(self, h):
        # p(z_t|h_t) logit
        x = self.get("prior_logit", Linear, self._stoch*self._classes)(h)
        logit = x.reshape(x.shape[:-1]+(self._stoch,self._classes))
        if self._unimix:
            probs = jax.nn.softmax(logit, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - self._unimix)*probs + self._unimix*uniform
            logit = jnp.log(probs)
        return logit

    def _dist(self, logit):
        return tfd.Independent(jaxutils.OneHotDist(logit.astype(f32)), 1)

    def loss(self, states, post_logit, free=1.0, mask=None):
        """
        states: {"deter": [B,T,d_model]}
        post_logit: q(z_t|x_t)의 logit [B,T,stoch,classes]
        mask: [B,T] 학습에 포함할 timestep에 True, 제외할 timestep에 False
        """
        metrics = {}
        prior = self._prior(states["deter"])  # [B,T,stoch,classes]
        post = post_logit  # q(z_t|x_t) logit

        # KL 계산
        dyn = self._dist(sg(post)).kl_divergence(self._dist(prior))  # [B,T]
        rep = self._dist(post).kl_divergence(self._dist(sg(prior)))  # [B,T]

        if free:
            dyn = jnp.maximum(dyn, free)
            rep = jnp.maximum(rep, free)

        # mask 적용
        dyn_loss = masked_mean(dyn, mask)
        rep_loss = masked_mean(rep, mask)

        # prior/post entropy
        prior_ent = self._dist(prior).entropy()  # [B,T]
        post_ent = self._dist(post).entropy()    # [B,T]

        metrics["prior_ent"] = masked_mean(prior_ent, mask)
        metrics["post_ent"] = masked_mean(post_ent, mask)

        losses = {"dyn": dyn, "rep": rep}
        return losses, metrics

class MultiEncoder(nj.Module):
    def __init__(
        self,
        spaces,
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        cnn_depth=48,
        cnn_blocks=2,
        resize="stride",
        symlog=False,
        minres=4,
        **kw,
    ):
        self.spaces = spaces
        self.mlp_keys = []
        self.cnn_keys = []
        for key, space in spaces.items():
            if key in ("is_first", "is_last", "is_terminal"):
                continue
            dims = len(space.shape) + bool(space.dtype in (np.uint32, np.uint64))
            if dims in (1, 2) and re.match(mlp_keys, key):
                self.mlp_keys.append(key)
            if dims == 3 and re.match(cnn_keys, key):
                self.cnn_keys.append(key)
        print("Encoder CNN shapes:", {k: spaces[k].shape for k in self.cnn_keys})
        print("Encoder MLP shapes:", {k: spaces[k].shape for k in self.mlp_keys})
        if self.cnn_keys:
            if cnn == "resnet":
                self._cnn = ImageEncoderResnet(
                    cnn_depth, cnn_blocks, resize, minres, **kw, name="cnn"
                )
            else:
                raise NotImplementedError(cnn)
        if self.mlp_keys:
            self._mlp = MLP(None, mlp_layers, mlp_units, **kw, dist="none", name="mlp")
        self._cnn_input = Input(self.cnn_keys, featdims=3)
        self._mlp_input = Input(self.mlp_keys, featdims=1)
        self._symlog = symlog

    def __call__(self, data, batchdims=2):
        assert len(data["is_first"].shape) == batchdims
        data = data.copy()
        for key, space in self.spaces.items():
            if space.dtype in (np.uint32, np.uint64):
                assert space.shape == () and (space.low == 0).all(), space
                data[key] = jax.nn.one_hot(data[key], space.high)
        outputs = []
        if self.cnn_keys:
            x = self._cnn_input(data, batchdims, jaxutils.COMPUTE_DTYPE)
            x = x.reshape((-1, *x.shape[batchdims:]))
            x = self._cnn(x)
            outputs.append(x)
        if self.mlp_keys:
            x = self._mlp_input(data, batchdims, f32)
            x = x.reshape((-1, *x.shape[batchdims:]))
            x = jaxutils.symlog(x) if self._symlog else x
            x = jaxutils.cast_to_compute(x)
            x = self._mlp(x, batchdims=1)
            outputs.append(x)
        x = jnp.concatenate(outputs, -1)
        x = x.reshape((*data["is_first"].shape, -1))
        return x


class MultiDecoder(nj.Module):
    def __init__(
        self,
        spaces,
        inputs=["tensor"],
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        cnn_depth=48,
        cnn_blocks=2,
        cnn_dist="mse",
        mlp_dist="mse",
        resize="stride",
        bins=255,
        outscale=1.0,
        minres=4,
        cnn_sigmoid=False,
        **kw,
    ):
        self.spaces = spaces
        self.mlp_keys = []
        self.cnn_keys = []
        for key, space in spaces.items():
            if key in ("is_first", "is_last", "is_terminal", "reward"):
                continue
            dims = len(space.shape) + bool(space.dtype in (np.uint32, np.uint64))
            if dims in (1, 2) and re.match(mlp_keys, key):
                self.mlp_keys.append(key)
            if dims == 3 and re.match(cnn_keys, key):
                self.cnn_keys.append(key)
        print("Decoder CNN shapes:", {k: spaces[k].shape for k in self.cnn_keys})
        print("Decoder MLP shapes:", {k: spaces[k].shape for k in self.mlp_keys})
        cnn_kw = {**kw, "minres": minres, "sigmoid": cnn_sigmoid}
        mlp_kw = {**kw, "outscale": outscale, "bins": bins}
        if self.cnn_keys:
            shapes = [spaces[k].shape for k in self.cnn_keys]
            assert all(x[:-1] == shapes[0][:-1] for x in shapes)
            shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
            if cnn == "resnet":
                self._cnn = ImageDecoderResnet(
                    shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name="cnn"
                )
            else:
                raise NotImplementedError(cnn)
        if self.mlp_keys:
            shapes, dists = {}, {}
            for key in self.mlp_keys:
                space = self.spaces[key]
                shapes[key] = space.shape
                if space.discrete and space.dtype == np.float32:
                    dists[key] = "onehot"
                elif space.discrete:
                    dists[key] = f"softmax{space.high}"
                else:
                    dists[key] = mlp_dist
            self._mlp = MLP(shapes, mlp_layers, mlp_units, dists, **mlp_kw, name="mlp")
        self._inputs = Input(inputs)
        self._image_dist = cnn_dist

    def __call__(self, inputs, batchdims=2, cnn=True, mlp=True):
        feat = self._inputs(inputs, batchdims, jaxutils.COMPUTE_DTYPE)
        dists = {}
        if self.cnn_keys and cnn:
            flat = feat.reshape([-1, feat.shape[-1]])
            output = self._cnn(flat)
            output = output.reshape(feat.shape[:-1] + output.shape[1:])
            split_indices = np.cumsum(
                [self.spaces[k].shape[-1] for k in self.cnn_keys][:-1]
            )
            means = jnp.split(output, split_indices, -1)
            dists.update(
                {
                    key: self._make_image_dist(key, mean)
                    for key, mean in zip(self.cnn_keys, means)
                }
            )
        if self.mlp_keys and mlp:
            dists.update(self._mlp(feat))
        return dists

    def _make_image_dist(self, name, mean):
        mean = mean.astype(f32)
        if self._image_dist == "normal":
            return tfd.Independent(tfd.Normal(mean, 1), 3)
        if self._image_dist == "mse":
            return jaxutils.MSEDist(mean, 3, "sum")
        raise NotImplementedError(self._image_dist)


class ImageEncoderResnet(nj.Module):
    def __init__(self, depth, blocks, resize, minres, **kw):
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._kw = kw

    def __call__(self, x):
        stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
        depth = self._depth
        x = jaxutils.cast_to_compute(x) - 0.5
        # print(x.shape)
        for i in range(stages):
            kw = {**self._kw, "preact": False}
            if self._resize == "stride":
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, **kw)(x)
            elif self._resize == "stride3":
                s = 2 if i else 3
                k = 5 if i else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, **kw)(x)
            elif self._resize == "mean":
                N, H, W, D = x.shape
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
                x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
            elif self._resize == "max":
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
                x = jax.lax.reduce_window(
                    x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "same"
                )
            else:
                raise NotImplementedError(self._resize)
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, "preact": True}
                x = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw)(x)
                x = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw)(x)
                x += skip
                # print(x.shape)
            depth *= 2
        if self._blocks:
            x = get_act(self._kw["act"])(x)
        x = x.reshape((x.shape[0], -1))
        # print(x.shape)
        return x


class ImageDecoderResnet(nj.Module):
    def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
        self._shape = shape
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._sigmoid = sigmoid
        self._kw = kw

    def __call__(self, x):
        stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
        depth = self._depth * 2 ** (stages - 1)
        x = jaxutils.cast_to_compute(x)
        x = self.get("in", Linear, (self._minres, self._minres, depth))(x)

        for i in range(stages):
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, "preact": True}
                x = self.get(f"s{i}b{j}conv1", Conv2D, depth, 3, **kw)(x)
                x = self.get(f"s{i}b{j}conv2", Conv2D, depth, 3, **kw)(x)
                x += skip
                # print(x.shape)
            depth //= 2
            kw = {**self._kw, "preact": False}
            if i == stages - 1:
                kw = {}
                depth = self._shape[-1]
            if self._resize == "stride":
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, transp=True, **kw)(x)
            elif self._resize == "stride3":
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, transp=True, **kw)(x)
            elif self._resize == "resize":
                x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
            else:
                raise NotImplementedError(self._resize)
        if max(x.shape[1:-1]) > max(self._shape[:-1]):
            padh = (x.shape[1] - self._shape[0]) / 2
            padw = (x.shape[2] - self._shape[1]) / 2
            x = x[:, int(np.ceil(padh)) : -int(padh), :]
            x = x[:, :, int(np.ceil(padw)) : -int(padw)]
        # print(x.shape)
        assert x.shape[-3:] == self._shape, (x.shape, self._shape)
        if self._sigmoid:
            x = jax.nn.sigmoid(x)
        else:
            x = x + 0.5
        return x


class MLP(nj.Module):
    def __init__(self, shape, layers, units, dist="mse", inputs=["tensor"], **kw):
        shape = (shape,) if isinstance(shape, (int, np.integer)) else shape
        assert isinstance(shape, (tuple, dict, type(None))), shape
        assert isinstance(dist, (str, dict)), dist
        assert isinstance(dist, dict) == isinstance(shape, dict), (dist, shape)
        self._shape = shape
        self._dist = dist
        self._layers = layers
        self._units = units
        self._inputs = Input(inputs)
        distkeys = ("outscale", "minstd", "maxstd", "unimix", "bins")
        self._kwdense = {k: v for k, v in kw.items() if k not in distkeys}
        self._kwdist = {k: v for k, v in kw.items() if k in distkeys}

    def __call__(self, inputs, batchdims=2):
        feat = self._inputs(inputs, batchdims, jaxutils.COMPUTE_DTYPE)
        x = feat.reshape([-1, feat.shape[-1]])
        for i in range(self._layers):
            x = self.get(f"h{i}", Linear, self._units, **self._kwdense)(x)
        x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
        if self._shape is None:
            return x
        elif isinstance(self._shape, dict):
            return {
                k: self._out(k, v, self._dist[k], x) for k, v in self._shape.items()
            }
        else:
            return self._out("out", self._shape, self._dist, x)

    def _out(self, name, shape, dist, x):
        return self.get(f"dist_{name}", Dist, shape, dist, **self._kwdist)(x)


class Dist(nj.Module):
    def __init__(
        self,
        shape,
        dist="mse",
        outscale=0.1,
        minstd=1.0,
        maxstd=1.0,
        unimix=0.0,
        bins=255,
    ):
        assert all(isinstance(dim, (int, np.integer)) for dim in shape), shape
        self._shape = shape
        self._dist = dist
        self._minstd = minstd
        self._maxstd = maxstd
        self._unimix = unimix
        self._outscale = outscale
        self._bins = bins

    def __call__(self, inputs):
        dist = self.inner(inputs)
        assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
            dist.batch_shape,
            dist.event_shape,
            inputs.shape,
        )
        return dist

    def inner(self, inputs):
        kw = {}
        kw["outscale"] = self._outscale
        shape = self._shape

        if self._dist.endswith("_twohot"):
            shape = (*self._shape, self._bins)
        if self._dist.startswith("softmax"):
            classes = int(self._dist[len("softmax") :])
            shape = (*self._shape, classes)

        out = self.get("out", Linear, int(np.prod(shape)), **kw)(inputs)
        out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
        if self._dist in ("normal", "trunc_normal"):
            std = self.get("std", Linear, int(np.prod(self._shape)), **kw)(inputs)
            std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)

        if self._dist == "symlog_mse":
            return jaxutils.SymlogDist(out, len(self._shape), "mse", "sum")
        if self._dist == "symlog_and_twohot":
            bins = np.linspace(-20, 20, out.shape[-1])
            return jaxutils.TwoHotDist(
                out, bins, len(self._shape), jaxutils.symlog, jaxutils.symexp
            )
        if self._dist == "symexp_twohot":
            bins = jaxutils.symexp(jnp.linspace(-20, 20, out.shape[-1], dtype=f32))
            return jaxutils.TwoHotDist(out, bins, len(self._shape))
        if self._dist == "symexp_twohot_stable":
            if out.shape[-1] % 2 == 1:
                half = jnp.linspace(-20, 0, (out.shape[-1] - 1) // 2 + 1, dtype=f32)
                half = jaxutils.symexp(half)
                bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
            else:
                half = jnp.linspace(-20, 0, out.shape[-1] // 2, dtype=f32)
                half = jaxutils.symexp(half)
                bins = jnp.concatenate([half, -half[::-1]], 0)
            return jaxutils.TwoHotDist(out, bins, len(self._shape))
        if self._dist == "parab_twohot":
            eps = 0.001
            f = lambda x: np.sign(x) * (
                np.square(
                    np.sqrt(1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps
                )
                - 1
            )
            bins = f(np.linspace(-300, 300, out.shape[-1]))
            return jaxutils.TwoHotDist(out, bins, len(self._shape))
        if self._dist == "mse":
            return jaxutils.MSEDist(out, len(self._shape), "sum")
        if self._dist == "normal":
            lo, hi = self._minstd, self._maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            dist = tfd.Normal(jnp.tanh(out), std)
            dist = tfd.Independent(dist, len(self._shape))
            dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            if self._shape:
                dist = tfd.Independent(dist, len(self._shape))
            return dist
        if self._dist.startswith("softmax"):
            dist = tfd.Categorical(out)
            if len(self._shape) > 1:
                dist = tfd.Independent(dist, len(self._shape) - 1)
            return dist
        if self._dist == "onehot":
            if self._unimix:
                probs = jax.nn.softmax(out, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                out = jnp.log(probs)
            dist = jaxutils.OneHotDist(out)
            if len(self._shape) > 1:
                dist = tfd.Independent(dist, len(self._shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self._shape[:-1]) * np.log(self._shape[-1])
            return dist
        raise NotImplementedError(self._dist)


class Conv2D(nj.Module):
    def __init__(
        self,
        depth,
        kernel,
        stride=1,
        transp=False,
        act="none",
        norm="none",
        pad="same",
        bias=True,
        preact=False,
        winit="uniform",
        fan="avg",
        dtype="default",
    ):
        self._depth = depth
        self._kernel = kernel
        self._stride = stride
        self._transp = transp
        self._act = get_act(act)
        self._norm = Norm(norm, name="norm")
        self._pad = pad.upper()
        self._bias = bias and (preact or norm == "none")
        self._preact = preact
        self._winit = Initializer(winit, 1.0, fan, dtype)
        self._binit = Initializer("zeros", 1.0, fan, dtype)

    def __call__(self, hidden):
        assert hidden.dtype in (jnp.float16, jnp.bfloat16, jnp.float32), (
            hidden.dtype,
            hidden.shape,
            self.path,
        )
        if self._preact:
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
            hidden = self._layer(hidden)
        else:
            hidden = self._layer(hidden)
            hidden = self._norm(hidden)
            hidden = self._act(hidden)
        return hidden

    def _layer(self, x):
        if self._transp:
            shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
            kernel = self.get("kernel", self._winit, shape)
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_transpose(
                x,
                kernel,
                (self._stride, self._stride),
                self._pad,
                dimension_numbers=("NHWC", "HWOI", "NHWC"),
            )
        else:
            shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
            kernel = self.get("kernel", self._winit, shape)
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_general_dilated(
                x,
                kernel,
                (self._stride, self._stride),
                self._pad,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
        if self._bias:
            bias = self.get("bias", self._binit, self._depth)
            bias = jaxutils.cast_to_compute(bias)
            x += bias
        return x


class Linear(nj.Module):
    def __init__(
        self,
        units,
        act="none",
        norm="none",
        bias=True,
        outscale=1.0,
        winit="normal",
        fan="avg",
        dtype="default",
    ):
        self._units = tuple(units) if hasattr(units, "__len__") else (units,)
        self._act = get_act(act)
        self._norm = norm
        self._bias = bias and norm == "none"
        self._winit = Initializer(winit, outscale, fan, dtype)
        self._binit = Initializer("zeros", 1.0, fan, dtype)

    def __call__(self, x):
        shape = (x.shape[-1], np.prod(self._units))
        kernel = self.get("kernel", self._winit, shape)
        kernel = kernel.astype(x.dtype)
        x = x @ kernel
        if self._bias:
            bias = self.get("bias", self._binit, np.prod(self._units))
            x += bias.astype(x.dtype)
        if len(self._units) > 1:
            x = x.reshape(x.shape[:-1] + self._units)
        x = self.get("norm", Norm, self._norm)(x)
        x = self._act(x)
        return x


class Norm(nj.Module):
    def __init__(self, impl, eps=1e-3):
        self._impl = impl
        self._eps = eps

    def __call__(self, x):
        if self._impl == "none":
            return x
        elif self._impl == "layer_old":
            x = x.astype(f32)
            x = jax.nn.standardize(x, axis=-1, eps=self._eps)
            x *= self.get("scale", jnp.ones, x.shape[-1], f32)
            x += self.get("offset", jnp.zeros, x.shape[-1], f32)
            return cast(x)
        elif self._impl == "layer":
            x = x.astype(f32)
            mean = x.mean(-1)[..., None]
            mean2 = jnp.square(x).mean(-1)[..., None]
            var = jnp.maximum(0, mean2 - jnp.square(mean))
            scale = self.get("scale", jnp.ones, x.shape[-1], f32)
            offset = self.get("offset", jnp.zeros, x.shape[-1], f32)
            x = (scale * jax.lax.rsqrt(var + self._eps)) * (x - mean) + offset
            return cast(x)
        elif self._impl == "instance":
            x = x.astype(f32)
            mean = x.mean(axis=(-3, -2), keepdims=True)
            var = x.var(axis=(-3, -2), keepdims=True)
            scale = self.get("scale", jnp.ones, x.shape[-1], f32)
            offset = self.get("offset", jnp.zeros, x.shape[-1], f32)
            x = (scale * jax.lax.rsqrt(var + self._eps)) * (x - mean) + offset
            return cast(x)
        else:
            raise NotImplementedError(self._impl)


class Input:
    def __init__(self, keys=["tensor"], featdims=1):
        self.keys = tuple(keys)
        self.featdims = featdims

    def __call__(self, inputs, batchdims=2, dtype=None):
        if not isinstance(inputs, dict):
            inputs = {"tensor": inputs}
        try:
            xs = []
            for key in self.keys:
                x = inputs[key]
                if jnp.issubdtype(x.dtype, jnp.complexfloating):
                    x = jnp.concatenate([x.real, x.imag], -1)
                x = x.astype(dtype or inputs[self.keys[0]].dtype)
                x = x.reshape((*x.shape[: batchdims + self.featdims - 1], -1))
                msg = f"Invalid input ({nj.SCOPE}, {key}, {x.shape}, {x.dtype}): {{x}}"
                jaxutils.check(jnp.isfinite(x).all(), msg, x=x)
                xs.append(x)
            xs = jnp.concatenate(xs, -1)
        except (KeyError, ValueError, TypeError) as e:
            shapes = {k: v.shape for k, v in inputs.items()}
            raise ValueError(
                f"Error: {e}\n"
                f"Input shapes: {shapes}\n" + f"Requested keys: {self.keys}"
            )
        return xs


class Initializer:
    def __init__(self, dist="uniform", scale=1.0, fan="avg", dtype="default"):
        if dtype == "default":
            dtype = jaxutils.PARAM_DTYPE
        self.scale = scale
        self.dist = dist
        self.fan = fan
        self.dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype

    def __call__(self, shape):
        shape = (shape,) if isinstance(shape, (int, np.integer)) else shape
        assert isinstance(shape, tuple), (shape, type(shape))
        if self.dist == "zeros" or self.scale == 0:
            return jnp.zeros(shape, self.dtype)
        elif self.dist == "uniform":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": (fanin + fanout) / 2, "in": fanin, "out": fanout}
            limit = np.sqrt(3 * self.scale / denoms[self.fan])
            value = jax.random.uniform(nj.rng(), shape, self.dtype, -limit, limit)
        elif self.dist == "normal":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": np.mean((fanin, fanout)), "in": fanin, "out": fanout}
            std = np.sqrt(self.scale / denoms[self.fan]) / 0.87962566103423978
            value = std * jax.random.truncated_normal(nj.rng(), -2, 2, shape)
            value = value.astype(self.dtype)
        elif self.dist == "normal_complex":
            assert jnp.issubdtype(self.dtype, jnp.complexfloating), self.dtype
            fanin, fanout = self._fans(shape)
            denoms = {"avg": np.mean((fanin, fanout)), "in": fanin, "out": fanout}
            std = np.sqrt(self.scale / denoms[self.fan]) / 0.87962566103423978
            real_dtype = jnp.finfo(self.dtype).self.dtype
            value = jax.random.truncated_normal(
                nj.rng(), -2, 2, (2, *shape), real_dtype
            )
            value = value[0] + 1j * value[1]
            value *= jax.lax.convert_element_type(std, real_dtype)
        elif self.dist == "ortho":
            nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
            matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
            mat = jax.random.normal(nj.rng(), matshape, self.dtype)
            qmat, rmat = jnp.linalg.qr(mat)
            qmat *= jnp.sign(jnp.diag(rmat))
            qmat = qmat.T if nrows < ncols else qmat
            qmat = qmat.reshape(nrows, *shape[:-1])
            value = self.scale * jnp.moveaxis(qmat, 0, -1)
        else:
            raise NotImplementedError(self.dist)
        return value

    def _fans(self, shape):
        if len(shape) == 0:
            return 1, 1
        elif len(shape) == 1:
            return shape[0], shape[0]
        elif len(shape) == 2:
            return shape
        else:
            space = int(np.prod(shape[:-2]))
            return shape[-2] * space, shape[-1] * space


def get_act(name):
    if callable(name):
        return name
    elif name == "none":
        return lambda x: x
    elif name == "mish":
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    elif hasattr(jax.nn, name):
        return getattr(jax.nn, name)
    else:
        raise NotImplementedError(name)
