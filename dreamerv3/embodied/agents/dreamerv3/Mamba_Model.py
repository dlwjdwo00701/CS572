from __future__ import annotations
import math
import jax
import jax.numpy as np
from jax.nn.initializers import lecun_normal, normal
from flax import linen as nn
from dataclasses import dataclass
from typing import Union
import jax.tree_util as tree_util

# ==========================
# Utilities and Configuration
# ==========================

@dataclass
class ModelArgs:
    """Model configuration arguments."""
    d_model: int    # Model input dimensionality
    n_layer: int    # Number of layers
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

# ==========================
# Layers and Blocks
# ==========================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.d_model,))
        normed = x * jax.lax.rsqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.eps)
        return normed * weight


class ResidualBlock(nn.Module):
    """Residual block with normalization and Mamba block."""
    args: ModelArgs

    def setup(self):
        self.mixer = MambaBlock(self.args)
        self.norm = RMSNorm(self.args.d_model)

    @nn.compact
    def __call__(self, x):
        return self.mixer(self.norm(x)) + x


class MambaBlock(nn.Module):
    """Main Mamba block implementation."""
    args: ModelArgs

    def setup(self):
        self.in_proj = nn.Dense(features=self.args.d_inner * 2, kernel_init=normal(), use_bias=self.args.bias)
        self.conv1d = nn.Conv(features=self.args.d_inner,
                              kernel_size=[self.args.d_conv],
                              feature_group_count=self.args.d_inner,
                              padding=self.args.d_conv - 1,
                              use_bias=self.args.conv_bias)
        self.x_proj = nn.Dense(self.args.dt_rank + self.args.d_state * 2, use_bias=False)
        self.dt_proj = nn.Dense(self.args.d_inner, use_bias=True)
        A = np.tile(np.arange(1, self.args.d_state + 1), (self.args.d_inner, 1))
        self.A_log = self.param('A_log', lambda rng, shape: np.log(A), (self.args.d_inner, self.args.d_state))
        self.D = self.param('D', nn.initializers.ones, (self.args.d_inner,))
        self.out_proj = nn.Dense(self.args.d_model, kernel_init=normal(), use_bias=self.args.bias)

    def __call__(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        x, res = np.split(x_and_res, [self.args.d_inner], axis=-1)
        x = jax.nn.silu(self.conv1d(x)[:, :l, :])
        y = self.ssm(x)
        y = y * jax.nn.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        (b, l, d_in) = x.shape
        A = -np.exp(self.A_log)
        D = self.D
        x_dbl = self.x_proj(x)
        delta, B, C = np.split(x_dbl, [self.args.dt_rank, self.args.dt_rank + self.args.d_state], axis=-1)
        delta = jax.nn.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = np.exp(np.einsum('bld,dn->bldn', delta, A))
        deltaB_u = np.einsum('bld,bln,bld->bldn', delta, B, u)
        x = np.zeros((b, d_in, n))
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = np.einsum('bdn,bn->bd', x, C[:, i])
            ys.append(y)
        y = np.stack(ys, axis=1)
        return y + u * D


# ==========================
# Main Model
# ==========================

class Mamba(nn.Module):
    """Full Mamba Model."""
    args: ModelArgs

    def setup(self):
        self.layers = [ResidualBlock(self.args) for _ in range(self.args.n_layer)]
        self.norm_f = RMSNorm(self.args.d_model)

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x


import jax
import jax.numpy as jnp

def pad_and_mask_inputs(inputs, target_length):
    """
    inputs: [B, L, D] 형태의 입력
    target_length: 패딩 후 맞출 길이 L_target

    returns:
      padded_inputs: [B, L_target, D]
      mask: [B, L_target]  (실제 데이터에는 1, 패딩 부분에는 0)
    """
    B, L, D = inputs.shape
    if L == target_length:
        # 이미 원하는 길이면 패딩 불필요
        mask = jnp.ones((B, L), dtype=jnp.float32)
        return inputs, mask
    elif L < target_length:
        # 제로패딩
        pad_len = target_length - L
        padded_inputs = jnp.pad(inputs, ((0,0), (0,pad_len), (0,0)))
        # 실제 데이터 부분 1, 패딩 부분 0
        mask = jnp.concatenate([jnp.ones((B, L)), jnp.zeros((B, pad_len))], axis=1)
        return padded_inputs, mask
    else:
        # target_length보다 긴 경우
        raise ValueError("target_length is shorter than input length.")


# 예제 입력
rng = jax.random.PRNGKey(42)
B = 1
D = 1024

long_seq_length = 20
short_seq_length = 16

long_input = jax.random.normal(rng, (B, long_seq_length, D))
short_input = jax.random.normal(rng, (B, short_seq_length, D))

# 두 입력을 모두 길이 20으로 맞추기
target_length = 20
long_input_padded, long_mask = pad_and_mask_inputs(long_input, target_length)
short_input_padded, short_mask = pad_and_mask_inputs(short_input, target_length)

# 모델 초기화
model_args = ModelArgs(
    d_model=1024,
    n_layer=4,
    d_state=16,
    expand=2,
    dt_rank='auto',
    d_conv=4,
    conv_bias=True,
    bias=False
)
mamba_model = Mamba(args=model_args)

params = mamba_model.init(rng, long_input_padded)  # 파라미터 초기화

# 파라미터 개수 출력
num_params = sum(p.size for p in tree_util.tree_leaves(params))
print("Number of parameters:", num_params)

# 모델 적용
long_output = mamba_model.apply(params, long_input_padded)   # [1,20,54]
short_output = mamba_model.apply(params, short_input_padded) # [1,20,54]

def mse_loss(pred, target, mask):
    mask = jnp.expand_dims(mask, axis=-1)  # [B,L,1]
    error = (pred - target)**2
    error = error * mask
    return jnp.sum(error) / jnp.sum(mask)

long_target = jax.random.normal(rng, (B, long_seq_length, D))
short_target = jax.random.normal(rng, (B, short_seq_length, D))
short_target_padded = jnp.pad(short_target, ((0,0),(0,target_length-short_seq_length),(0,0)))

long_loss = mse_loss(long_output, long_target, long_mask)
short_loss = mse_loss(short_output, short_target_padded, short_mask)

print("Long loss:", long_loss)
print("Short loss:", short_loss)
