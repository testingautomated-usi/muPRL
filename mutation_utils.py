import csv
import json
import os

import numpy as np
from properties import RESULTS_ROOT, ROOT_DIR


def create_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(name=folder, exist_ok=True)
        except OSError as e:
            print("Unable to create folder for mutated models:" + str(e))
            raise Exception()


def create_results_folders(algo, env):
    create_folder(RESULTS_ROOT)

    algo_path = os.path.join(RESULTS_ROOT, algo)  # ALGO
    create_folder(algo_path)

    subj_path = os.path.join(algo_path, env)  # SUBJ
    create_folder(subj_path)

    raw_res_path = os.path.join(subj_path, "raw_results")
    create_folder(raw_res_path)

    return subj_path, raw_res_path


def write_to_csv(rows, dir, file_name):

    out_csv_file = os.path.join(dir, file_name)

    with open(out_csv_file, "w") as file:
        writer = csv.writer(
            file,
            delimiter=",",
            lineterminator="\n",
        )
        writer.writerows(rows)


def get_operator_trained_confs(algo, env, operator_name):
    path = os.path.join(ROOT_DIR, "logs", algo)
    conf_list = [
        item
        for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item))
        and env in item
        and operator_name in item
    ]
    conf_list = [
        item.replace(env + "_" + "mutant" + "_" + operator_name + "_", "")
        for item in conf_list
    ]
    return conf_list


def get_agent_stats(
    algo,
    env,
    data_type,
    agent_type,
    operator,
    conf,
    num_runs,
    get_raw_results,
    threshold_failure,
):
    path = os.path.join(ROOT_DIR, "logs", algo, str(env) + "_" + str(agent_type))
    if agent_type == "mutant":
        path = "_".join((path, str(operator), str(conf)))

    results_stats = []
    results_raw = []
    results = []

    if not os.path.isdir(path):
        raise Exception(f"Path {path} is not a directory")
    else:
        for i in range(num_runs):
            stats_folder = os.path.join(path, "_".join(("run", str(i))))
            if data_type == "strong" and agent_type == "original":
                stats_file = [
                    filename
                    for filename in os.listdir(stats_folder)
                    if filename.endswith(
                        str(operator)
                        + "-"
                        + str(conf)
                        + "-"
                        + str(data_type)
                        + "-statistics.json"
                    )
                ]
                if len(stats_file) == 0:
                    stats_file = [
                        filename
                        for filename in os.listdir(stats_folder)
                        if filename.endswith(str(data_type) + "-statistics.json")
                    ]
            else:
                stats_file = [
                    filename
                    for filename in os.listdir(stats_folder)
                    if filename.endswith(str(data_type) + "-statistics.json")
                ]

            if len(stats_file) == 0:
                raise Exception(
                    f"No statistics file that ends with {str(data_type)}-statistics.json in {stats_folder}"
                )

            with open(os.path.join(stats_folder, stats_file[0])) as f:
                stats = json.load(f)

            failure_probabilities = (
                (np.asarray(stats["failure_probabilities"]) > threshold_failure)
                .astype(int)
                .tolist()
            )
            num_failures = failure_probabilities.count(1)
            num_success = failure_probabilities.count(0)

            results_stats.append([num_success, num_failures])
            results_raw.append(failure_probabilities)

        results.append(results_stats)
        if get_raw_results:
            results.append(results_raw)

    return results
