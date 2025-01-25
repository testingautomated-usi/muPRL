import statistics
from argparse import ArgumentParser
from rliable import library as rly

import numpy as np
from config import CARTPOLE_ENV_NAME, ENV_NAMES
from mutants.utils import find_all_mutation_operators
from mutation_utils import (
    create_results_folders,
    get_agent_stats,
    get_operator_trained_confs,
    write_to_csv,
)
from properties import SENSITIVITY_MS
from stats import is_diff_sts_fisher, probability_of_improvement
from test_generation.utils.env_utils import ALGOS


def triviality_check(num_runs, original_results, mutant_results):
    num_original_success = 0
    num_mutant_fail = 0

    for i in range(num_runs):
        original_results_i = np.array(original_results[i])
        mutant_results_i = np.array(mutant_results[i])
        orig_success_idx = np.where(original_results_i == 0)[0]

        mutant_subset = mutant_results_i[orig_success_idx]

        mutant_fail_idx = np.where(mutant_subset == 1)[0]

        num_original_success += len(orig_success_idx)
        num_mutant_fail += len(mutant_fail_idx)

    if num_original_success == 0:
        is_trivial = False
    else:
        is_trivial = round(num_mutant_fail / num_original_success, 2) > 0.9

    return is_trivial


def killability_check(num_runs, original_stats, mutant_stats, bootstrap):
    killed_pairs = 0
    num_runs_original_weaker = 0
    failure_rates_mutant = []
    success_rates_original = []
    success_rates_mutant = []
    num_episodes = []
    for i in range(num_runs):
        failure_rate_mutant = 1 - mutant_stats[i][0] / (
            mutant_stats[i][0] + mutant_stats[i][1]
        )
        num_episodes.append(original_stats[i][0] + original_stats[i][1])
        if bootstrap:
            if original_stats[i][1] > mutant_stats[i][1]:
                # not counting the case where the mutant is killed because the original
                # configuration is weak
                num_runs_original_weaker += 1
                continue
            success_rate_original = round(
                original_stats[i][0] / (original_stats[i][0] + original_stats[i][1]), 3
            )
            success_rate_mutant = round(
                mutant_stats[i][0] / (mutant_stats[i][0] + mutant_stats[i][1]), 3
            )
            success_rates_original.append(success_rate_original)
            success_rates_mutant.append(success_rate_mutant)
        else:
            is_killed_pair = is_diff_sts_fisher([original_stats[i], mutant_stats[i]])
            if is_killed_pair and original_stats[i][1] > mutant_stats[i][1]:
                # not counting the case where the mutant is killed because the original
                # configuration is weak
                num_runs_original_weaker += 1
                continue
            killed_pairs += int(is_killed_pair)
        failure_rates_mutant.append(failure_rate_mutant)

    if bootstrap:
        all_pairs = dict()
        all_pairs["original_mutant"] = (
            np.array([success_rates_original]).T,
            np.array([success_rates_mutant]).T,
        )
        average_probabilities, average_prob_cis = rly.get_interval_estimates(
            all_pairs, probability_of_improvement, reps=2000
        )
        point_estimate = average_probabilities["original_mutant"]
        low_ci = average_prob_cis["original_mutant"][0][0]
        if point_estimate > 0.5 and low_ci > 0.5:
            is_killed_conf = True
        else:
            is_killed_conf = False

        killed_pairs = np.where(
            np.array(success_rates_original) > np.array(success_rates_mutant)
        )[0].shape[0]
    else:
        if killed_pairs >= round((num_runs - num_runs_original_weaker) / 2):
            is_killed_conf = True
        else:
            is_killed_conf = False

    return (
        is_killed_conf,
        round(killed_pairs / (num_runs - num_runs_original_weaker), 2),
        failure_rates_mutant,
    )


def check_killability_triviality(
    subj_path, do_triviality, algo, env, num_runs, threshold_failure, bootstrap
):
    get_raw_results = do_triviality
    original_results = get_agent_stats(
        algo,
        env,
        "replay",
        "original",
        None,
        None,
        num_runs,
        get_raw_results,
        threshold_failure,
    )

    killable_operators = []

    killability_report_operator = [["operator", "is_killable"]]
    killability_report_conf = [["operator", "conf", "is_killable", "killability_rate"]]
    triviality_report = [["operator", "conf", "is_trivial"]]

    for operator in find_all_mutation_operators():
        is_operator_killable = False

        operator = {"name": operator}

        confs = get_operator_trained_confs(algo, env, operator["name"])

        # operator might not be applicable to a certain algorithm; in that case skip.
        if len(confs) > 0:
            print(f"Check killability for operator {operator}")
            for conf in confs:
                try:
                    mutant_results = get_agent_stats(
                        algo,
                        env,
                        "replay",
                        "mutant",
                        operator["name"],
                        conf,
                        num_runs,
                        get_raw_results,
                        threshold_failure,
                    )
                    is_conf_killable, killability_rate, _ = killability_check(
                        num_runs, original_results[0], mutant_results[0], bootstrap
                    )
                    killability_report_conf.append(
                        [operator["name"], conf, is_conf_killable, killability_rate]
                    )

                    if do_triviality:
                        is_trivial = triviality_check(
                            num_runs, original_results[1], mutant_results[1]
                        )
                        triviality_report.append([operator["name"], conf, is_trivial])

                    if is_conf_killable:
                        is_operator_killable = True
                except Exception as ex:
                    print(ex)

            for report in killability_report_conf:
                if report[0] == operator["name"]:
                    if is_operator_killable:
                        killable_operators.append(operator)
                    killability_report_operator.append(
                        [operator["name"], is_operator_killable]
                    )
                    break

    write_to_csv(killability_report_operator, subj_path, "killable_operators.csv")
    write_to_csv(killability_report_conf, subj_path, "killable_configs.csv")
    write_to_csv(triviality_report, subj_path, "trivial_configs.csv")

    return killable_operators, killability_report_conf, triviality_report


def calculate_mutation_score(
    killable_operators,
    triviality_report,
    test_datasets,
    do_sensitivity,
    subj_path,
    algo,
    env,
    num_runs,
    threshold_failure,
    bootstrap,
):
    if len(killable_operators) != 0:
        get_raw_results = False

        mut_scores = []
        mut_scores_kr = []
        sensitivity_report = []
        sensitivity_report_fr = []

        for dataset in test_datasets:
            data_mut_scores = []
            data_mut_scores_kr = []
            data_mut_scores_fr = []
            data_mut_scores_report = [["operator", "MS", "KR_MS", "FR_MS"]]
            data_killed_config_report = [
                [
                    "operator",
                    "conf",
                    "is_killed",
                    "killability_rate",
                    "failure_rates_mutant",
                ]
            ]
            original_results = get_agent_stats(
                algo,
                env,
                dataset,
                "original",
                None,
                None,
                num_runs,
                get_raw_results,
                threshold_failure,
            )

            for operator in killable_operators:
                print(
                    f"Compute mutation score for {operator} with {dataset} test generator"
                )
                confs = get_operator_trained_confs(algo, env, operator["name"])

                filter_trivial_mutants = filter(
                    lambda item: item[0] == operator["name"] and item[2],
                    triviality_report,
                )
                trivial_confs = list(map(lambda item: item[1], filter_trivial_mutants))

                confs = list(filter(lambda conf: conf not in trivial_confs, confs))

                killable_confs = []
                killability_rates = []
                failure_rates_mutants = []

                for conf in confs:
                    try:
                        mutant_results = get_agent_stats(
                            algo,
                            env,
                            dataset,
                            "mutant",
                            operator["name"],
                            conf,
                            num_runs,
                            get_raw_results,
                            threshold_failure,
                        )
                        is_conf_killable, killability_rate, failure_rates_mutant = (
                            killability_check(
                                num_runs,
                                original_results[0],
                                mutant_results[0],
                                bootstrap,
                            )
                        )
                        killable_confs.append(int(is_conf_killable))
                        killability_rates.append(killability_rate)
                        failure_rates_mutants.append(np.mean(failure_rates_mutant))
                        data_killed_config_report.append(
                            [
                                operator["name"],
                                conf,
                                is_conf_killable,
                                killability_rate,
                                np.mean(failure_rates_mutant),
                            ]
                        )
                    except Exception as ex:
                        print(ex)

                if len(killable_confs) > 0:
                    ms_operator = statistics.mean(killable_confs)
                    ms_operator_kr = statistics.mean(killability_rates)
                    ms_operator_fr = statistics.mean(failure_rates_mutants)
                    data_mut_scores.append(ms_operator)
                    data_mut_scores_kr.append(ms_operator_kr)
                    data_mut_scores_fr.append(ms_operator_fr)
                    data_mut_scores_report.append(
                        [operator["name"], ms_operator, ms_operator_kr, ms_operator_fr]
                    )

            if len(data_mut_scores) > 0:
                dataset_ms = statistics.mean(data_mut_scores)
                dataset_ms_kr = statistics.mean(data_mut_scores_kr)
                dataset_ms_fr = statistics.mean(data_mut_scores_fr)

                data_mut_scores_report.append(
                    ["total", dataset_ms, dataset_ms_kr, dataset_ms_fr]
                )
                write_to_csv(
                    data_mut_scores_report,
                    subj_path,
                    "mutation_scores_" + str(dataset) + ".csv",
                )

                write_to_csv(
                    data_killed_config_report,
                    subj_path,
                    "killability_" + str(dataset) + ".csv",
                )

                mut_scores.append(dataset_ms)
                mut_scores_kr.append(dataset_ms_kr)
                if SENSITIVITY_MS == "KR":
                    sensitivity_report.append([dataset, dataset_ms_kr])
                else:
                    sensitivity_report.append([dataset, dataset_ms])

                sensitivity_report_fr.append([dataset, dataset_ms_fr])

        if len(test_datasets) > 1 and do_sensitivity and len(mut_scores) > 1:
            if SENSITIVITY_MS == "KR":
                mut_scores_sensitivity = mut_scores_kr
            else:
                mut_scores_sensitivity = mut_scores

            if mut_scores_sensitivity[0] >= mut_scores_sensitivity[1]:
                sensitivity = 0
            else:
                max_score = max(mut_scores_sensitivity[0], mut_scores_sensitivity[1])
                sensitivity = round(
                    (mut_scores_sensitivity[1] - mut_scores_sensitivity[0]) / max_score,
                    3,
                )
                print(
                    f"{mut_scores_sensitivity[0]} vs {mut_scores_sensitivity[1]}, sensitivity: {sensitivity}"
                )

            sensitivity_report.append(["sensitivity", sensitivity])
            write_to_csv(
                sensitivity_report,
                subj_path,
                "sensitivity_" + "_".join(test_datasets) + ".csv",
            )

            write_to_csv(
                sensitivity_report_fr,
                subj_path,
                "sensitivity_fr_" + "_".join(test_datasets) + ".csv",
            )
    else:
        raise Exception("No killable operators")


algos_list = list(ALGOS.keys())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--triv",
        help="triviality",
        dest="do_triviality",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--ms", help="ms", dest="do_ms", type=int, default=1, choices=[0, 1]
    )
    parser.add_argument(
        "--sens",
        help="sensitivity",
        dest="do_sensitivity",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        default="ppo",
        type=str,
        required=False,
        choices=algos_list,
    )
    parser.add_argument(
        "--env",
        type=str,
        default=CARTPOLE_ENV_NAME,
        choices=ENV_NAMES,
        help="environment ID",
    )
    parser.add_argument(
        "--num-runs", type=int, help="Number of runs for the agent", default=1
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Use bootstrap for confidence intervals (using the U statistic of the Mann-Whitney test instead of the Fisher's test) for killability analysis",
        default=False,
    )
    # FIXME: duplicated with testing_args and train.py (test_generation)
    parser.add_argument(
        "--threshold-failure",
        type=float,
        help="Threshold of failure probability after which the episode is considered a failure (this holds for deterministic and non-deterministic environments)",
        default=0.5,
    )
    args = parser.parse_args()

    subj_path, raw_res_path = create_results_folders(args.algo, args.env)
    killable_operators, _, triviality_report = check_killability_triviality(
        subj_path,
        args.do_triviality,
        args.algo,
        args.env,
        args.num_runs,
        args.threshold_failure,
        args.bootstrap,
    )

    if args.do_ms:
        calculate_mutation_score(
            killable_operators,
            triviality_report,
            ["weaker", "strong"],
            args.do_sensitivity,
            subj_path,
            args.algo,
            args.env,
            args.num_runs,
            args.threshold_failure,
            args.bootstrap,
        )
