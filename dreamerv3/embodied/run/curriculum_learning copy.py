import re
from collections import defaultdict
from functools import partial as bind
import copy
import embodied
import numpy as np

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
    train_epstats = embodied.Agg()  # aggregates the episodic returns
    eval_episodes = defaultdict(embodied.Agg)
    eval_epstats = embodied.Agg()

    batch_steps = args.batch_size * args.batch_length
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial)

    # DUMMY TASKS FOR DEBUGGING
    tasks = [
        {'name': 'humanoid_h1hand-stand-v0', 'difficulty': 1, 'reward_threshold': 800},
        {'name': 'humanoid_h1hand-walk-v0', 'difficulty': 2, 'reward_threshold': 1000},
        {'name': 'humanoid_h1hand-run-v0', 'difficulty': 3, 'reward_threshold': 1500},
        {'name': 'humanoid_h1hand-maze-v0', 'difficulty': 4, 'reward_threshold': 2000},
        # {'name': 'humanoid_h1hand-sit_simple-v0', 'difficulty': 2, 'reward_threshold': 1000}, # does it load the chair??
        # {'name': 'humanoid_h1hand-slide-v0', 'difficulty': 3, 'reward_threshold': 1500},
        # {'name': 'humanoid_h1hand-stair-v0', 'difficulty': 3, 'reward_threshold': 1500},
        # {'name': 'humanoid_h1hand-hurdle-v0', 'difficulty': 3, 'reward_threshold': 1500},
        # {'name': 'humanoid_h1hand-sit_hard-v0', 'difficulty': 4, 'reward_threshold': 2000}, # diff obs space
        # {'name': 'humanoid_h1hand-balance_simple-v0', 'difficulty': 2, 'reward_threshold': 1000}, # diff obs space
        # {'name': 'humanoid_h1hand-balance_hard-v0', 'difficulty': 4, 'reward_threshold': 2000}, # diff obs space
        # {'name': 'humanoid_h1strong-highbar_simple-v0', 'difficulty': 5, 'reward_threshold': 2500}, # doesnt exist??
        # {'name': 'humanoid_h1strong-highbar_hard-v0', 'difficulty': 5, 'reward_threshold': 2500},# doesnt exist??
    ]
    # TODO: ADD WANDB LOG FOR TASK SWITCHING
    # TODO: ADD VICTORIN"S EVALUATION METRIC
    num_tasks = len(tasks)

    for task_info in tasks:
        task_name = task_info['name']
        task_config = copy.deepcopy(config)
        task_config = task_config.update({'task': task_name})

        task_eval_config = copy.deepcopy(eval_config)
        task_eval_config = task_eval_config.update({'task': task_name})
        
        score_threshold = task_info['reward_threshold']
        task_difficulty = task_info['difficulty']
        recent_scores = []

        
        fns = [bind(make_train_env, task_config, i) for i in range(args.num_envs)]
        train_driver = embodied.Driver(fns, args.driver_parallel)
        train_driver.on_step(lambda tran, _: step.increment())
        train_driver.on_step(train_replay.add)

        fns_eval = [bind(make_eval_env, task_eval_config, i) for i in range(args.num_envs)]
        eval_driver = embodied.Driver(fns_eval, args.driver_parallel)
        eval_driver.on_step(eval_replay.add)

        # TODO: UPDATE LOGGER SO THAT I CAN ALSO VISUALIZE EACH TASK REWARD SEPERATELY
        @embodied.timer.section("log_step")
        def log_step(tran, worker, mode):
            nonlocal recent_scores
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
                        "task_name": task_name,  # Include task name
                        "task_number": task_difficulty,
                    },
                    prefix="episode",
                )
                logger.add(
                    {
                        "return": score,
                        "episode_length": length,
                        "task_name": task_name,  # Include task name
                        "task_number": task_difficulty,
                    },
                    prefix="results",
                )
                if worker < args.log_video_streams:
                    for key in args.log_keys_video:
                        if key in tran:
                            logger.add({"video": result[f"policy_{key}"]}, prefix="results")
                if "success" in result:
                    logger.add({"success": result.pop("success")}, prefix="results")
                if "success_subtasks" in result:
                    logger.add({"success_subtasks": result.pop("success_subtasks")}, prefix="results")
                rew = result.pop("rewards")
                result["reward_rate"] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
                epstats.add(result)

                if mode == 'train':
                    recent_scores.append(score)
                    if len(recent_scores) > 10:
                        recent_scores.pop(0)

        train_driver.on_step(bind(log_step, mode="train"))
        eval_driver.on_step(bind(log_step, mode="eval"))

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

        checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
        checkpoint.step = step
        checkpoint.agent = agent

        checkpoint.train_replay = train_replay
        checkpoint.eval_replay = eval_replay

        if args.from_checkpoint:
            checkpoint.load(args.from_checkpoint)
        else:
            checkpoint.save()
        # checkpoint.load_or_save()
        should_save(step)  # Register that we just saved.

        train_policy = lambda *args: agent.policy(
            *args, mode="explore" if should_expl(step) else "train"
        )
        eval_policy = lambda *args: agent.policy(*args, mode="eval")
        train_driver.reset(agent.init_policy)

        def task_completion_criteria(steps_taken, num_tasks):
            completed = steps_taken >= int(args.steps / num_tasks)
            if completed is True:
                print("TASK COMPLETEED")
            return completed

        print(f"Start training loop on {task_name}")

        # Initialize the starting step for the current task
        task_start_step = step.save()
        steps_taken = 0

        # Log the start of a new task
        logger.add({
            "task_name": task_name,
            "task_number": task_difficulty,
            "event": "task_started",
            "timestamp": task_start_step
        }, prefix="task")

        while not task_completion_criteria(steps_taken, num_tasks):
            print("STEPS: ", step)
            if should_eval(step):
                print("Start evaluation")
                eval_driver.reset(agent.init_policy)
                eval_driver(eval_policy, episodes=args.eval_eps)
                logger.add(eval_epstats.result(), prefix="epstats")
                if len(eval_replay):
                    logger.add(agent.report(next(eval_dataset)), prefix="eval")

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

            # Update steps_taken for the current task
            steps_taken = step.save() - task_start_step

            # Check if the task's reward threshold is met
            # if len(recent_scores) >= 10 and np.mean(recent_scores[-10:]) >= score_threshold:
            # if len(recent_scores) >= 10:
            #     print(f"Task {task_name} completed with average score {np.mean(recent_scores[-10:])}")
            #     logger.add({
            #         "task_name": task_name,
            #         "average_score": np.mean(recent_scores[-10:]),
            #         "reward_threshold": score_threshold,
            #         "event": "task_completed",
            #         "timestamp": step
            #     }, prefix="task")
            #     break  # Move to the next task


        print(f"Completed training on {task_name} task")
        checkpoint.save()

    print("\nCurriculum learning finished")
    logger.close()
