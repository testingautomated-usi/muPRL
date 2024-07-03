import numpy as np
import statsmodels.stats.proportion as smp
from log import Log, close_loggers
from randomness_utils import set_random_seed

from test_generation.test_configurator import TestConfigurator
from test_generation.testing_args import TestingArgs

if __name__ == "__main__":
    args = TestingArgs().parse()

    all_values = []

    for i in range(args.num_runs_experiments):
        logger = Log("generate")
        logger.info("Args: {}".format(args))

        if args.seed == -1:
            args.seed = np.random.randint(2**32 - 1)

        # also set when instantiating the algorithm
        set_random_seed(seed=args.seed)

        test_configurator = TestConfigurator(
            env_name=args.env_name,
            render=args.render,
            seed=args.seed,
            num_envs=1,
            folder=args.folder,
            algo=args.algo,
            exp_id=args.exp_id,
            testing_policy_for_training_name=args.testing_policy_for_training_name,
            testing_strategy_name=args.testing_strategy_name,
            model_checkpoint=args.model_checkpoint,
            training_progress_filter=args.training_progress_filter,
            layers=args.layers,
            num_episodes=args.num_episodes,
            exp_name="{}-trial".format(i),
            num_runs_each_env_config=args.num_runs_each_env_config,
            parallelize=args.parallelize,
            budget=args.budget,
            sampling_size=args.sampling_size,
            register_environment=args.register_env,
            **args.wrapper_kwargs,
        )

        num_experiments = 0
        num_failures = 0
        previous_env_config = None
        map_env_config_failure_prob = dict()
        map_env_config_min_fitness = dict()
        num_trials = 0
        episode_num = 0
        min_fitness_values = []

        while num_experiments < args.num_runs_each_env_config * args.num_episodes:
            if (
                num_experiments % args.num_runs_each_env_config == 0
                and num_experiments != 0
            ):
                map_env_config_failure_prob[previous_env_config.get_str()] = (
                    num_failures / args.num_runs_each_env_config,
                    smp.proportion_confint(
                        count=num_failures,
                        nobs=args.num_runs_each_env_config,
                        method="wilson",
                    ),
                )
                logger.info(
                    "Failure probability for env config {}: {}".format(
                        previous_env_config.get_str(),
                        map_env_config_failure_prob[previous_env_config.get_str()],
                    )
                )
                num_failures = 0
                episode_num = 0
                num_trials += 1

            failure, env_config, fitness_values = test_configurator.test_single_episode(
                episode_num=episode_num, num_trials=num_trials
            )
            logger.debug("Env config: {}".format(env_config.get_str()))
            previous_env_config = env_config
            if failure:
                num_failures += 1
            num_experiments += 1
            episode_num += 1
            logger.debug(
                "Num experiments: {}/{}".format(
                    num_experiments, args.num_runs_each_env_config * args.num_episodes
                )
            )

            if len(fitness_values) > 0:
                min_fitness_values.append(min(fitness_values))
                logger.debug(f"Min fitness value: {min(fitness_values)}")

        if len(map_env_config_failure_prob) > 0:
            map_env_config_failure_prob[previous_env_config.get_str()] = (
                num_failures / args.num_runs_each_env_config,
                smp.proportion_confint(
                    count=num_failures,
                    nobs=args.num_runs_each_env_config,
                    method="wilson",
                ),
            )

        if len(map_env_config_min_fitness) > 0:
            map_env_config_min_fitness[previous_env_config.get_str()] = np.mean(
                min_fitness_values
            )
            min_fitness_values.clear()

        values = []

        num_failures = 0
        for key, value in map_env_config_failure_prob.items():
            if value[0] > 0.5:
                num_failures += 1
                logger.info(
                    "FAIL - Failure probability for env config {}: {}".format(
                        key, value
                    )
                )
            else:
                logger.info(
                    "Failure probability for env config {}: {}".format(key, value)
                )
            values.append(value[0])

        all_values.extend(values)
        if len(values) > 0:
            logger.info(
                "Failure probabilities: {}, Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}".format(
                    values,
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                )
            )

        if len(map_env_config_min_fitness) > 0:
            mean_fitness_values = [
                mean_fitness_value
                for mean_fitness_value in map_env_config_min_fitness.values()
            ]
            logger.info(
                "Min fitness values: Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}, Values: {}".format(
                    np.mean(mean_fitness_values),
                    np.std(mean_fitness_values),
                    np.min(mean_fitness_values),
                    np.max(mean_fitness_values),
                    mean_fitness_values,
                )
            )

        logger.info("{}".format(map_env_config_failure_prob.keys()))

        logger.info(
            "Number of evaluation predictions (i.e. number of times a model was used to make predictions): {}".format(
                test_configurator.get_num_evaluation_predictions()
            )
        )
        test_configurator.close_env()

        close_loggers()

    if len(all_values) > 0:
        num_total_failures = len(list(filter(lambda v: v > 0.5, all_values)))
        print(
            f"Considering {args.num_runs_experiments} "
            f"experiments the {args.testing_strategy_name} method generated: "
            f"{num_total_failures} failures / {len(all_values)} episodes"
        )
        print(
            "Mean: {:.2f}, Std: {:.2f}, Min: {}, Max: {}".format(
                np.mean(all_values),
                np.std(all_values),
                np.min(all_values),
                np.max(all_values),
            )
        )
