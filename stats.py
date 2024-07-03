#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from typing import List, Tuple
from bisect import bisect_left
import numpy as np
import pandas as pd
import properties
import statsmodels.api as sm
import statsmodels.stats.power as pw
from patsy import dmatrices
from scipy.stats import fisher_exact, wilcoxon, mannwhitneyu, rankdata
from scipy.stats.contingency import odds_ratio


# calculates cohen's kappa value
def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        (
            (nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2
            + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2
        )
        / dof
    )
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return result


# calculates whether two accuracy arrays are statistically different according to GLM
def is_diff_sts(orig_accuracy_list, accuracy_list, threshold=0.05):
    print("_" * 50)
    print(orig_accuracy_list)
    print("_" * 50)
    print((accuracy_list))
    print("_" * 50)

    if properties.STAT_TEST == "WLX":
        p_value = p_value_wilcoxon(orig_accuracy_list, accuracy_list)
    elif properties.STAT_TEST == "GLM":
        p_value = p_value_glm(orig_accuracy_list, accuracy_list)

    effect_size = cohen_d(orig_accuracy_list, accuracy_list)

    # if properties.MODEL_TYPE == 'regression':
    #     is_sts = ((p_value < threshold) and effect_size <= -0.5)
    # else:
    #     is_sts = ((p_value < threshold) and effect_size >= 0.5)

    is_sts = (p_value < threshold) and (effect_size <= -0.5 or effect_size >= 0.5)

    return is_sts, p_value, effect_size


def p_value_wilcoxon(orig_accuracy_list, accuracy_list):
    w, p_value_w = wilcoxon(orig_accuracy_list, accuracy_list, mode="approx")

    return p_value_w


def p_value_glm(orig_accuracy_list, accuracy_list):
    list_length = len(orig_accuracy_list)

    zeros_list = [0] * list_length
    ones_list = [1] * list_length
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list

    data = {"Acc": acc_lists, "Mod": mod_lists}
    df = pd.DataFrame(data)

    response, predictors = dmatrices("Acc ~ Mod", df, return_type="dataframe")
    glm = sm.GLM(response, predictors)
    glm_results = glm.fit()
    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value_g = float(pv)

    return p_value_g


def power(orig_accuracy_list, mutation_accuracy_list):
    eff_size = cohen_d(orig_accuracy_list, mutation_accuracy_list)
    pow = pw.FTestAnovaPower().solve_power(
        effect_size=eff_size,
        nobs=len(orig_accuracy_list) + len(mutation_accuracy_list),
        alpha=0.05,
    )
    return pow


def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray) -> float:
    # Copyright 2021 The Rliable Authors.
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #      http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """Overall Probability of imporvement of algorithm `X` over `Y`.

    Args:
      scores_x: A matrix of size (`num_runs_x` x `num_tasks`) where scores_x[n][m]
        represent the score on run `n` of task `m` for algorithm `X`.
      scores_y: A matrix of size (`num_runs_y` x `num_tasks`) where scores_x[n][m]
        represent the score on run `n` of task `m` for algorithm `Y`.
    Returns:
        P(X_m > Y_m) averaged across tasks.
    """
    num_tasks = scores_x.shape[1]
    task_improvement_probabilities = []
    num_runs_x, num_runs_y = scores_x.shape[0], scores_y.shape[0]
    assert (
        num_runs_x == num_runs_y
    ), "Number of runs should be equal for both algorithms"
    for task in range(num_tasks):
        if np.array_equal(scores_x[:, task], scores_y[:, task]):
            task_improvement_prob = 0.5
        else:
            task_improvement_prob = len(
                np.where(scores_x[:, task] > scores_y[:, task])[0]
            ) + 0.5 * len(np.where(scores_x[:, task] == scores_y[:, task])[0])
            task_improvement_prob /= num_runs_x
        task_improvement_probabilities.append(task_improvement_prob)
    return np.mean(task_improvement_probabilities)


def is_diff_sts_fisher(table, threshold=0.05):
    res = fisher_exact(table, alternative="two-sided")
    # print(res)
    # changes with newer versions
    # is_sts = (res.pvalue < threshold)
    is_sts = res[1] < threshold
    return is_sts


def compute_odds_ratio(table):
    return odds_ratio(table=table).statistic


def vargha_delaney(a: List[float], b: List[float]) -> Tuple[float, str]:
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param a: a numeric list
    :param b: another numeric list
    :returns the value estimate and the magnitude
    """
    m = len(a)
    n = len(b)

    assert m == n, "The two list must be of the same length: {}, {}".format(m, n)

    r = rankdata(a + b)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (
        2 * n * m
    )  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


# paired
def is_diff_sts_wilc(a: List[float], b: List[float], threshold: bool = 0.05) -> bool:
    if np.array_equal(a, b):
        return False
    res = wilcoxon(x=a, y=b)
    is_sts = res[1] < threshold
    return bool(is_sts)


# non-paired
def is_diff_sts_mann(
    original: List[float], mutant: List[float], threshold: bool = 0.05
) -> bool:
    if np.array_equal(original, mutant):
        return False
    res = mannwhitneyu(x=original, y=mutant)
    is_sts = res[1] < threshold
    return bool(is_sts)
