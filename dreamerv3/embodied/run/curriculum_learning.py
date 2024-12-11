import re
from collections import defaultdict
from functools import partial as bind
import copy
import embodied
import numpy as np

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
        # No index meets the threshold; treat as failure
        index_corrected = len(signal)
    else:
        first_idx = np.min(idx)
        index_corrected = first_idx if first_idx >= min_index else len(signal)
    return index_corrected / len(signal)

def curriculum_learning(
    make_agent,
    make_train_replay,
    make_eval_replay,
    make_train_env,
    make_eval_env,
    make_logger,
    args,
    config,
    eval_config,
):
    agent = make_agent()
    train_replay = make_train_replay()
    eval_replay = make_eval_replay()
    logger = make_logger()

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    step = logger.step
    step.reset()
    usage = embodied.Usage(**args.usage)
    agg = embodied.Agg()
    train_episodes = defaultdict(embodied.Agg)
    train_epstats = embodied.Agg()  # aggregates episodic returns
    eval_episodes = defaultdict(embodied.Agg)
    eval_epstats = embodied.Agg()

    batch_steps = args.batch_size * args.batch_length
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial)

    # Example tasks
    tasks = [
        {'name': 'humanoid_h1hand-stand-v0', 'difficulty': 1, 'reward_threshold': 600},
        {'name': 'humanoid_h1hand-walk-v0', 'difficulty': 2, 'reward_threshold': 600},
        {'name': 'humanoid_h1hand-maze-v0', 'difficulty': 3, 'reward_threshold': 250},
        {'name': 'humanoid_h1hand-run-v0', 'difficulty': 4, 'reward_threshold': 500},
    ]
    # tasks = [
    #     {'name': 'humanoid_h1hand-stand-v0', 'difficulty': 1, 'reward_threshold': 5},
    #     {'name': 'humanoid_h1hand-walk-v0', 'difficulty': 2, 'reward_threshold': 5},
    #     {'name': 'humanoid_h1hand-maze-v0', 'difficulty': 3, 'reward_threshold': 5},
    #     {'name': 'humanoid_h1hand-run-v0', 'difficulty': 4, 'reward_threshold': 5},
    # ]
    num_tasks = len(tasks)

    def log_step(tran, worker, mode, task_name, task_difficulty, train_scores, eval_scores):
        episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
        epstats = dict(train=train_epstats, eval=eval_epstats)[mode]

        episode = episodes[worker]
        if tran["is_first"]:
            episode.reset()

        episode.add("score", tran["reward"], agg="sum")
        episode.add("length", 1, agg="sum")
        episode.add("rewards", tran["reward"], agg="stack")

        if "success" in tran:
            episode.add("success", tran["success"], agg="sum")
        if "success_subtasks" in tran:
            episode.add("success_subtasks", tran["success_subtasks"], agg="max")

        if worker < args.log_video_streams:
            for key in args.log_keys_video:
                if key in tran:
                    episode.add(f"policy_{key}", tran[key], agg="stack")

        for key, value in tran.items():
            if re.match(args.log_keys_sum, key):
                episode.add(key, value, agg="sum")
            if re.match(args.log_keys_avg, key):
                episode.add(key, value, agg="avg")
            if re.match(args.log_keys_max, key):
                episode.add(key, value, agg="max")

        if tran["is_last"]:
            result = episode.result()
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
            if worker < args.log_video_streams:
                for key in args.log_keys_video:
                    if key in tran:
                        logger.add({"video": result[f"policy_{key}"]}, prefix=f"results_{mode}")
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

    # # Function to decide if task is completed based on complexity metrics
    # def task_completed_victorin(eval_returns):
    #     # eval_returns is a list of episode returns from the evaluation.
    #     if len(eval_returns) < 5:
    #         # If insufficient data, don't mark as completed
    #         return False

    #     # Normalize returns
    #     norm_returns = normalisation(np.array(eval_returns))
    #     auc_comp = complexity_auc(norm_returns)
    #     grad_comp = complexity_gradient(norm_returns)
    #     idx_comp = complexity_index_percentage(norm_returns, 0.8, 50)

    #     # Create a final score or criterion
    #     # For example, let's say we consider the task done if:
    #     # 1) AUC complexity is below 0.5 (hypothetical)
    #     # 2) Gradient complexity is below 2.0 (hypothetical)
    #     # 3) Index complexity < 0.7 (hypothetical)
    #     # Adjust these thresholds based on your own experiments.
    #     done = (auc_comp < 0.5) and (grad_comp < 2.0) and (idx_comp < 0.7)
    #     logger.add({
    #         # "task_name": task_name,
    #         "norm_returns": norm_returns,
    #         "auc_comp": auc_comp,
    #         "grad_comp": grad_comp,
    #         "idx_comp": idx_comp,
    #         "done": done,
    #         "timestamp": step.save()
    #     }, prefix="eval_metrics")
    #     return done

    def task_completed(eval_returns, threshold, task_id):
        # Simple criterion: if average of last 10 evaluation returns exceeds threshold
        if len(eval_returns) < 5:
            return False
        mean_return = np.mean(eval_returns[-10:])
        done = mean_return >= threshold

        logger.add({    
            # "task_name": task_name,
            "task_id": task_id,
            "mean_returns": mean_return,
            "reward_threshold": threshold,
            "done": done,
        }, prefix="eval_metrics")
        return done
    
    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.step = step
    checkpoint.agent = agent

    checkpoint.train_replay = train_replay
    checkpoint.eval_replay = eval_replay

    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    else:
        checkpoint.save()
    should_save(step)  # Register that we just saved.
    
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

        fns = [bind(make_train_env, task_config, i) for i in range(args.num_envs)]
        train_driver = embodied.Driver(fns, args.driver_parallel)
        train_driver.on_step(lambda tran, _: step.increment())
        train_driver.on_step(train_replay.add)

        # fns_eval = [bind(make_eval_env, task_eval_config, i) for i in range(args.num_envs)]
        # I DONT THINK WE NEED MORE THAN 1 ENV FOR EVALUATION
        fns_eval = [bind(make_eval_env, task_eval_config, i) for i in range(1)]
        eval_driver = embodied.Driver(fns_eval, args.driver_parallel)
        eval_driver.on_step(eval_replay.add)

        train_driver.on_step(bind(log_step, mode="train", task_name=task_name, task_difficulty=task_difficulty, train_scores=train_scores, eval_scores=[]))
        eval_driver.on_step(bind(log_step, mode="eval", task_name=task_name, task_difficulty=task_difficulty, train_scores=[], eval_scores=eval_scores))

        train_dataset = agent.dataset(
            embodied.Batch([train_replay.dataset] * args.batch_size)
        )
        eval_dataset = agent.dataset(
            embodied.Batch([eval_replay.dataset] * args.batch_size)
        )
        carry = [agent.init_train(args.batch_size)]

        def train_step(tran, worker):
            if len(train_replay) < args.batch_size or step < args.train_fill:
                return
            for _ in range(should_train(step)):
                with embodied.timer.section("dataset_next"):
                    batch = next(train_dataset)
                outs, carry[0], mets = agent.train(batch, carry[0])
                agg.add(mets, prefix="train")

        train_driver.on_step(train_step)
        
        # checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
        # checkpoint.step = step
        # checkpoint.agent = agent

        # checkpoint.train_replay = train_replay
        # checkpoint.eval_replay = eval_replay

        # if args.from_checkpoint:
        #     checkpoint.load(args.from_checkpoint)
        # else:
        #     checkpoint.save()
        # should_save(step)  # Register that we just saved.

        train_policy = lambda *args: agent.policy(
            *args, mode="explore" if should_expl(step) else "train"
        )
        eval_policy = lambda *args: agent.policy(*args, mode="eval")
        train_driver.reset(agent.init_policy)

        print(f"Start training loop on {task_name}")

        # Log the start of a new task
        logger.add({
            "task_name": task_name,
            "task_number": task_difficulty,
            "event": "task_started",
            "timestamp": step.save()
        }, prefix="task")

        # Instead of fixed steps, we will rely on evaluations to determine completion.
        # We run training steps until an evaluation decides the task is completed.
        task_solved = False

        while not task_solved:
            print("STEPS: ", step)
            # print("TRAIN REPLAY: ", train_replay.stats())
            # print("EVAL REPLAY: ", eval_replay.stats())

            if should_eval(step):
                print("Start evaluation")
                eval_driver.reset(agent.init_policy)
                eval_driver(eval_policy, episodes=args.eval_eps)
                # Extract evaluation returns for completion metric
                # eval_results = eval_epstats.result(reset=False)
                # eval_returns = eval_results.get("score", [])

                logger.add(eval_epstats.result(), prefix="epstats")

 

                if len(eval_replay):
                    logger.add(agent.report(next(eval_dataset)), prefix="eval")



                # Compute complexity metrics and decide if done
                if task_completed(eval_scores, reward_threshold, task_difficulty):
                    print(f"Task {task_name} completed based on complexity metrics!")
                    logger.add({
                        "task_name": task_name,
                        "event": "task_completed",
                        "timestamp": step.save()
                    }, prefix="task")
                    task_solved = True

            if task_solved:
                break

            # Continue training if not completed
            train_driver(train_policy, steps=10)

            if should_log(step):
                logger.add(agg.result())
                logger.add(train_epstats.result(), prefix="epstats")
                if len(train_replay):
                    logger.add(agent.report(next(train_dataset)), prefix="report")
                logger.add(embodied.timer.stats(), prefix="timer")
                logger.add(train_replay.stats(), prefix="replay")
                logger.add(usage.stats(), prefix="usage")
                logger.write(fps=True)

            if should_save(step):
                checkpoint.save()

        print(f"Completed training on {task_name} task")
        checkpoint.save()
        # Close environments and drivers
        train_driver.close()
        eval_driver.close()

    print("\nCurriculum learning finished")
    logger.close()
    train_driver.close()
    eval_driver.close()
    # train_replay.close()
    # eval_replay.close()
