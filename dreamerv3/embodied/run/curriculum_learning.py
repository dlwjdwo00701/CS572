import re
from collections import defaultdict
from functools import partial as bind
import copy
import embodied
import numpy as np
import random

def normalisation(signal):
    min_signal = np.min(signal)
    max_signal = np.max(signal)
    denom = (max_signal - min_signal)
    return (signal - min_signal) / denom if denom != 0 else signal

def complexity_auc(signal):
    return 1 - np.trapz(signal, np.arange(len(signal))) / len(signal)

def complexity_gradient(signal):
    val = np.sum(np.gradient(signal) * np.flip(np.arange(len(signal)) + 1)) / np.sum(np.arange(len(signal)) + 1) * len(signal)
    return np.inf if val <= 0 else 1 / val

def complexity_index_percentage(signal, threshold, min_index):
    max_threshold = threshold * np.max(signal)
    idx = np.where(signal >= max_threshold)[0]
    if len(idx) == 0:
        index_corrected = len(signal)
    else:
        first_idx = np.min(idx)
        index_corrected = first_idx if first_idx >= min_index else len(signal)
    return index_corrected / len(signal)

def collect_episodes(envs, agent, batch_length=64):
    """
    32개 환경(envs)로부터 에피소드 1개씩을 동시에 수집한다.
    각 에피소드는 최대 64스텝이며, 일찍 종료되면 0패딩.

    반환되는 data 딕셔너리 각 키 설명:
    - 'action': shape (32,64,61) float32, 각 스텝별 action
    - 'is_first': shape (32,64) bool, 에피소드의 첫 스텝 여부
    - 'is_last': shape (32,64) bool, 에피소드가 해당 스텝에서 종료되었는지 여부
    - 'is_terminal': shape (32,64) bool, 환경 상에서 terminal 상태 여부
    - 'reward': shape (32,64) float32, 해당 스텝의 reward
    - 'success': shape (32,64) float32, 성공 여부(또는 메트릭) 정보
    - 'success_subtasks': shape (32,64) float32, 서브태스크 성공 정보
    - 'vector': shape (32,64,155) float32, 관측 정보 벡터
    - 'padding_start': shape (32,) int, 각 에피소드별로 0 패딩이 시작되는 스텝 인덱스
      (에피소드가 조기 종료되면 그 시점 이후로 패딩)
    """
    num_envs = len(envs)
    policy_carry = agent.init_policy(num_envs)

    obs_list = []
    for env in envs:
        # reset 시 action 0으로
        obs = env.step({"reset": True, "action": np.zeros((61,), dtype=np.float32)})
        obs_list.append(obs)
    obs = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0].keys()}

    num_steps = batch_length
    data = {
        "action": np.zeros((num_envs, num_steps, 61), dtype=np.float32),
        "is_first": np.zeros((num_envs, num_steps), dtype=bool),
        "is_last": np.zeros((num_envs, num_steps), dtype=bool),
        "is_terminal": np.zeros((num_envs, num_steps), dtype=bool),
        "reward": np.zeros((num_envs, num_steps), dtype=np.float32),
        "success": np.zeros((num_envs, num_steps), dtype=np.float32),
        "success_subtasks": np.zeros((num_envs, num_steps), dtype=np.float32),
        "vector": np.zeros((num_envs, num_steps, 155), dtype=np.float32),
    }

    done_flags = np.zeros((num_envs,), dtype=bool)
    data["is_first"][:,0] = True
    data["reward"][:,0] = obs["reward"]
    data["is_last"][:,0] = obs["is_last"]
    data["is_terminal"][:,0] = obs["is_terminal"]
    data["success"][:,0] = obs.get("success", 0.0)
    data["success_subtasks"][:,0] = obs.get("success_subtasks", 0.0)
    data["vector"][:,0] = obs["vector"]

    # 첫 액션 결정: current_index 추가
    first_obs = copy.deepcopy(obs)
    first_obs["current_index"] = np.zeros((num_envs,), dtype=int)  # step=0
    act, policy_carry = agent.policy(first_obs, policy_carry, mode="train")
    action_array = act["action"]
    data["action"][:,0] = action_array

    for t in range(1, num_steps):
        done_flags = np.logical_or(done_flags, data["is_last"][:,t-1])
        step_mask = ~done_flags
        step_indices = np.where(step_mask)[0]

        obs_list = [None]*num_envs
        if len(step_indices) > 0:
            step_actions = [{"action": data["action"][i,t-1], "reset": False} for i in range(num_envs)]
            for i in step_indices:
                obs_list[i] = envs[i].step(step_actions[i])

        for i in range(num_envs):
            if obs_list[i] is not None:
                o = obs_list[i]
                data["reward"][i,t] = o["reward"]
                data["is_last"][i,t] = o["is_last"]
                data["is_terminal"][i,t] = o["is_terminal"]
                data["success"][i,t] = o.get("success", 0.0)
                data["success_subtasks"][i,t] = o.get("success_subtasks", 0.0)
                data["vector"][i,t] = o["vector"]

        next_obs = {
            "is_first": data["is_first"][:, :t+1],
            "is_last": data["is_last"][:, :t+1],
            "is_terminal": data["is_terminal"][:, :t+1],
            "reward": data["reward"][:, :t+1],
            "success": data["success"][:, :t+1],
            "success_subtasks": data["success_subtasks"][:, :t+1],
            "vector": data["vector"][:, :t+1],
            "current_index": np.tile(np.arange(t+1)[None,:], (num_envs,1)),
        }

        print(f"next_obs : {next_obs}")
        act, policy_carry = agent.policy(next_obs, policy_carry, mode="train")
        action_array = act["action"]
        action_array = action_array * step_mask[:,None].astype(action_array.dtype)
        data["action"][:,t] = action_array

    # 패딩 시작점 계산
    padding_start = np.full((num_envs,), num_steps, dtype=int)
    for i in range(num_envs):
        last_steps = np.where(data["is_last"][i])[0]
        if len(last_steps) > 0:
            padding_start[i] = min(last_steps[0] + 1, num_steps)

    data["padding_start"] = padding_start
    return data

def collect_eval_episodes(eval_env, agent, batch_length=64, batch_size=16):
    data = {
        "action": np.zeros((batch_size, batch_length, 61), dtype=np.float32),
        "is_first": np.zeros((batch_size, batch_length), dtype=bool),
        "is_last": np.zeros((batch_size, batch_length), dtype=bool),
        "is_terminal": np.zeros((batch_size, batch_length), dtype=bool),
        "reward": np.zeros((batch_size, batch_length), dtype=np.float32),
        "success": np.zeros((batch_size, batch_length), dtype=np.float32),
        "success_subtasks": np.zeros((batch_size, batch_length), dtype=np.float32),
        "vector": np.zeros((batch_size, batch_length, 155), dtype=np.float32),
    }

    for epi in range(batch_size):
        obs = eval_env.step({"reset": True, "action": np.zeros((61,), dtype=np.float32)})
        policy_carry = agent.init_policy(1)
        data["is_first"][epi,0] = True
        data["reward"][epi,0] = obs["reward"]
        data["is_last"][epi,0] = obs["is_last"]
        data["is_terminal"][epi,0] = obs["is_terminal"]
        data["success"][epi,0] = obs.get("success", 0.0)
        data["success_subtasks"][epi,0] = obs.get("success_subtasks", 0.0)
        data["vector"][epi,0] = obs["vector"]

        # first step policy: current_index 추가
        single_obs = {k: data[k][epi,0] for k in ["is_first","is_last","is_terminal","reward","success","success_subtasks"]}
        single_obs["vector"] = data["vector"][epi,0]
        single_obs["current_index"] = np.array([0], dtype=int)
        for k,v in single_obs.items():
            if k == "vector":
                # vector는 (155,) → (1,155)
                single_obs[k] = v[np.newaxis, :]
            elif v.ndim == 0:
                single_obs[k] = v[np.newaxis]

        act, policy_carry = agent.policy(single_obs, policy_carry, mode="eval")
        data["action"][epi,0] = act["action"][0]

        done_flag = data["is_last"][epi,0]
        for t in range(1, batch_length):
            if done_flag:
                continue
            step_action = {"action": data["action"][epi,t-1], "reset": False}
            obs = eval_env.step(step_action)
            data["reward"][epi,t] = obs["reward"]
            data["is_last"][epi,t] = obs["is_last"]
            data["is_terminal"][epi,t] = obs["is_terminal"]
            data["success"][epi,t] = obs.get("success", 0.0)
            data["success_subtasks"][epi,t] = obs.get("success_subtasks", 0.0)
            data["vector"][epi,t] = obs["vector"]
            done_flag = obs["is_last"]

            next_obs = {
                "is_first": np.array([[data["is_first"][epi,t]]]),
                "is_last": np.array([[data["is_last"][epi,t]]]),
                "is_terminal": np.array([[data["is_terminal"][epi,t]]]),
                "reward": np.array([[data["reward"][epi,t]]]),
                "success": np.array([[data["success"][epi,t]]]),
                "success_subtasks": np.array([[data["success_subtasks"][epi,t]]]),
                "vector": data["vector"][epi,t:t+1],
                "current_index": np.array([[t]], dtype=int),  # eval에서도 current_index 전달
            }

            act, policy_carry = agent.policy(next_obs, policy_carry, mode="eval")
            data["action"][epi,t] = act["action"][0]

    # eval에서도 padding_start 계산 (선택사항)
    padding_start = np.full((batch_size,), batch_length, dtype=int)
    for i in range(batch_size):
        last_steps = np.where(data["is_last"][i])[0]
        if len(last_steps) > 0:
            padding_start[i] = min(last_steps[0] + 1, batch_length)
    data["padding_start"] = padding_start

    return data

def curriculum_learning(
        make_agent,
        make_train_replay,  # 사용하지 않음
        make_eval_replay,   # 사용하지 않음
        make_train_env,
        make_eval_env,
        make_logger,
        args,
        config,
        eval_config,
):
    agent = make_agent()
    logger = make_logger()

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    step = logger.step
    step.reset()
    usage = embodied.Usage(**args.usage)
    agg = embodied.Agg()
    train_episodes = defaultdict(embodied.Agg)
    train_epstats = embodied.Agg()
    eval_episodes = defaultdict(embodied.Agg)
    eval_epstats = embodied.Agg()

    batch_steps = args.batch_size * args.batch_length
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial)

    tasks = [
        {'name': 'humanoid_h1hand-stand-v0', 'difficulty': 1, 'reward_threshold': 600},
        {'name': 'humanoid_h1hand-walk-v0', 'difficulty': 2, 'reward_threshold': 600},
        {'name': 'humanoid_h1hand-maze-v0', 'difficulty': 3, 'reward_threshold': 250},
        {'name': 'humanoid_h1hand-run-v0', 'difficulty': 4, 'reward_threshold': 500},
    ]

    def log_step(tran, worker, mode, task_name, task_difficulty, train_scores, eval_scores):
        episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
        epstats = dict(train=train_epstats, eval=eval_epstats)[mode]

        if tran["is_first"]:
            episodes[worker].reset()

        episodes[worker].add("score", tran["reward"], agg="sum")
        episodes[worker].add("length", 1, agg="sum")
        episodes[worker].add("rewards", tran["reward"], agg="stack")

        if "success" in tran:
            episodes[worker].add("success", tran["success"], agg="sum")
        if "success_subtasks" in tran:
            episodes[worker].add("success_subtasks", tran["success_subtasks"], agg="max")

        for key, value in tran.items():
            if re.match(args.log_keys_sum, key):
                episodes[worker].add(key, value, agg="sum")
            if re.match(args.log_keys_avg, key):
                episodes[worker].add(key, value, agg="avg")
            if re.match(args.log_keys_max, key):
                episodes[worker].add(key, value, agg="max")

        if tran["is_last"]:
            result = episodes[worker].result()
            score = result.pop("score")
            length = result.pop("length") - 1
            logger.add(
                {
                    "score": score,
                    "length": length,
                    "task_name": task_name,
                    "task_number": task_difficulty,
                },
                prefix=f"episode_{mode}",
            )
            logger.add(
                {
                    "return": score,
                    "episode_length": length,
                    "task_name": task_name,
                    "task_number": task_difficulty,
                },
                prefix=f"results_{mode}",
            )

            if "success" in result:
                logger.add({"success": result.pop("success")}, prefix=f"results_{mode}")
            if "success_subtasks" in result:
                logger.add({"success_subtasks": result.pop("success_subtasks")}, prefix=f"results_{mode}")

            rew = result.pop("rewards")
            result["reward_rate"] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

            if mode == 'train':
                train_scores.append(score)
                if len(train_scores) > 10:
                    train_scores.pop(0)

            if mode == 'eval':
                eval_scores.append(score)
                if len(eval_scores) > 100:
                    eval_scores.pop(0)

    def task_completed(eval_returns, threshold, task_id):
        if len(eval_returns) < 5:
            return False
        mean_return = np.mean(eval_returns[-10:])
        done = mean_return >= threshold

        logger.add({
            "task_id": task_id,
            "mean_returns": mean_return,
            "reward_threshold": threshold,
            "done": done,
        }, prefix="eval_metrics")
        return done

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.step = step
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    else:
        checkpoint.save()
    should_save(step)

    train_envs = [make_train_env(config, i) for i in range(args.num_envs)]
    eval_env = make_eval_env(eval_config, 0)

    for task_info in tasks:
        task_name = task_info['name']
        task_config = copy.deepcopy(config)
        task_config = task_config.update({'task': task_name})

        task_eval_config = copy.deepcopy(eval_config)
        task_eval_config = task_eval_config.update({'task': task_name})

        task_difficulty = task_info['difficulty']
        reward_threshold = task_info['reward_threshold']
        train_scores = []
        eval_scores = []

        print(f"Start training loop on {task_name}")

        logger.add({
            "task_name": task_name,
            "task_number": task_difficulty,
            "event": "task_started",
            "timestamp": step.save()
        }, prefix="task")

        task_solved = False

        while not task_solved:
            train_data = collect_episodes(train_envs, agent, batch_length=args.batch_length)
            selected_idx = random.sample(range(32), args.batch_size)
            batch_data = {k: v[selected_idx] for k,v in train_data.items() if k != "padding_start"}

            for _ in range(should_train(step)):
                outs, new_carry, mets = agent.train(batch_data, agent.init_train(args.batch_size))
                agg.add(mets, prefix="train")

            total_env_steps = args.num_envs * args.batch_length
            step.increment(total_env_steps)

            # 에피소드별 로깅
            for env_i in range(args.num_envs):
                ep_is_last = train_data["is_last"][env_i]
                if np.any(ep_is_last):
                    last_idx = np.where(ep_is_last)[0][-1]
                else:
                    last_idx = args.batch_length - 1
                tran = {
                    "is_first": train_data["is_first"][env_i,last_idx],
                    "is_last": train_data["is_last"][env_i,last_idx],
                    "is_terminal": train_data["is_terminal"][env_i,last_idx],
                    "reward": train_data["reward"][env_i,last_idx],
                    "success": train_data["success"][env_i,last_idx],
                    "success_subtasks": train_data["success_subtasks"][env_i,last_idx],
                }
                log_step(tran, env_i, "train", task_name, task_difficulty, train_scores, [])

            if should_eval(step):
                print("Start evaluation")
                eval_data = collect_eval_episodes(eval_env, agent, batch_length=args.batch_length, batch_size=args.batch_size)
                for i in range(args.batch_size):
                    ep_is_last = eval_data["is_last"][i]
                    if np.any(ep_is_last):
                        last_idx = np.where(ep_is_last)[0][-1]
                    else:
                        last_idx = args.batch_length-1
                    tran = {
                        "is_first": eval_data["is_first"][i,last_idx],
                        "is_last": eval_data["is_last"][i,last_idx],
                        "is_terminal": eval_data["is_terminal"][i,last_idx],
                        "reward": eval_data["reward"][i,last_idx],
                        "success": eval_data["success"][i,last_idx],
                        "success_subtasks": eval_data["success_subtasks"][i,last_idx],
                    }
                    log_step(tran, i, "eval", task_name, task_difficulty, [], eval_scores)

                logger.add(eval_epstats.result(), prefix="epstats")
                logger.add(agent.report(eval_data), prefix="eval")

                if task_completed(eval_scores, reward_threshold, task_difficulty):
                    print(f"Task {task_name} completed based on criteria!")
                    logger.add({
                        "task_name": task_name,
                        "event": "task_completed",
                        "timestamp": step.save()
                    }, prefix="task")
                    task_solved = True

            if should_log(step):
                logger.add(agg.result())
                logger.add(train_epstats.result(), prefix="epstats")
                logger.add(embodied.timer.stats(), prefix="timer")
                logger.add(usage.stats(), prefix="usage")
                logger.write(fps=True)

            if should_save(step):
                checkpoint.save()

        print(f"Completed training on {task_name} task")
        checkpoint.save()

    print("\nCurriculum learning finished")
    logger.close()
    eval_env.close()
    for env in train_envs:
        env.close()
