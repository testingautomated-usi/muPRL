#!/bin/bash

env_name=
algo_name=
wrapper_kwargs=
mode="replay"
delete_mutants="false"
mutant_name=
# the following parameters will be ignored in replay mode
num_episodes=1
seed=0
testing_strategy_name=nn
testing_policy_for_training_name=mlp
num_layers=2
filter=
testing_configurations="false"
parallelize="false"
mutant_configuration=
num_cpus=-1
run_num=-1


while [ $# -gt 0 ] ; do
  case $1 in
    -a | --algo-name) algo_name="$2" ;;
    -e | --env-name) env_name="$2" ;;
    -d | --mode) mode="$2" ;;
    -m | --mutant-name) mutant_name="$2" ;;
    -n | --num-episodes) num_episodes="$2" ;;
    -s | --seed) seed="$2" ;;
    -l | --num-layers) num_layers="$2" ;;
    -f | --filter) filter="$2" ;;
    -t | --testing-configurations) testing_configurations="$2" ;;
    -p | --parallelize) parallelize="$2" ;;
    -u | --mutant-configuration) mutant_configuration="$2" ;;
    -c | --num-cpus) num_cpus="$2" ;;
    -o | --delete-mutants) delete_mutants="$2" ;;
    -r | --run-num) run_num="$2" ;;
  esac
  shift
done

sampling_size=500
if [[ "$env_name" == "CartPole-v1" ]]; then
  wrapper_kwargs="timeout_steps:500 fail_on_timeout:False"
  num_runs_each_env_config=1
elif [[ "$env_name" == "Humanoid-v4" ]]; then
  num_runs_each_env_config=3
elif [[ "$env_name" == "parking-v0" ]]; then
  num_runs_each_env_config=1
elif [[ "$env_name" == "LunarLander-v2" ]]; then
  wrapper_kwargs="timeout_steps:1000 fail_on_timeout:True"
  num_runs_each_env_config=3
else
  echo Env name "$env_name" not supported
  exit 1
fi

if [[ "$mode" == "replay" || "$mode" == "weak" || "$mode" == "weaker" ]]; then
  if [[ "$parallelize" == "true" ]]; then
    if [[ "$delete_mutants" == "true" ]]; then
      python -m test_generation.test_agent --algo "$algo_name" --env-name "$env_name" \
        --num-runs-each-env-config "$num_runs_each_env_config" \
        --register-env --wrapper-kwargs $wrapper_kwargs \
        --training-type mutant --mutant-name "$mutant_name" --mutant-configuration "$mutant_configuration" \
        --mode "$mode" --num-episodes "$num_episodes" --seed "$seed" \
        --parallelize --num-cpus "$num_cpus" --delete-mutants-after-replay
    else
      python -m test_generation.test_agent --algo "$algo_name" --env-name "$env_name" \
          --num-runs-each-env-config "$num_runs_each_env_config" \
          --register-env --wrapper-kwargs $wrapper_kwargs \
          --training-type mutant --mutant-name "$mutant_name" --mutant-configuration "$mutant_configuration" \
          --mode "$mode" --num-episodes "$num_episodes" --seed "$seed" \
          --parallelize --num-cpus "$num_cpus"
    fi
  else
    if [[ "$delete_mutants" == "true" ]]; then
      python -m test_generation.test_agent --algo "$algo_name" --env-name "$env_name" \
        --num-runs-each-env-config "$num_runs_each_env_config" \
        --register-env --wrapper-kwargs $wrapper_kwargs \
        --training-type mutant --mutant-name "$mutant_name" --mutant-configuration "$mutant_configuration" \
        --mode "$mode" --num-episodes "$num_episodes" --seed "$seed" --delete-mutants-after-replay --run-num "$run_num"
    else
      python -m test_generation.test_agent --algo "$algo_name" --env-name "$env_name" \
        --num-runs-each-env-config "$num_runs_each_env_config" \
        --register-env --wrapper-kwargs $wrapper_kwargs \
        --training-type mutant --mutant-name "$mutant_name" --mutant-configuration "$mutant_configuration" \
        --mode "$mode" --num-episodes "$num_episodes" --seed "$seed" --run-num "$run_num"
    fi
  fi
elif [[ $mode == "strong" ]]; then
  if [[ $testing_configurations == "true" ]]; then
    # we just remove the filter parameter
    if [[ "$parallelize" == "true" ]]; then
      python -m test_generation.test_agent --algo "$algo_name" --env-name "$env_name" \
        --num-runs-each-env-config "$num_runs_each_env_config" \
        --register-env --wrapper-kwargs $wrapper_kwargs \
        --training-type mutant --mutant-name "$mutant_name" \
        --mode "$mode" --num-episodes "$num_episodes" --seed "$seed" \
        --testing-strategy-name "$testing_strategy_name" \
        --testing-policy-for-training-name "$testing_policy_for_training_name" \
        --layers "$num_layers" --sampling-size "$sampling_size" --parallelize \
        --mutant-configuration "$mutant_configuration" --num-cpus "$num_cpus"
    else
      python -m test_generation.test_agent --algo "$algo_name" --env-name "$env_name" \
        --num-runs-each-env-config "$num_runs_each_env_config" \
        --register-env --wrapper-kwargs $wrapper_kwargs \
        --training-type mutant --mutant-name "$mutant_name" \
        --mode "$mode" --num-episodes "$num_episodes" --seed "$seed" \
        --testing-strategy-name "$testing_strategy_name" \
        --testing-policy-for-training-name "$testing_policy_for_training_name" \
        --layers "$num_layers" --sampling-size "$sampling_size" \
        --mutant-configuration "$mutant_configuration" --run-num "$run_num"
    fi
  else
    if [[ "$parallelize" == "true" ]]; then
      python -m test_generation.test_agent --algo "$algo_name" --env-name "$env_name" \
        --num-runs-each-env-config "$num_runs_each_env_config" \
        --register-env --wrapper-kwargs $wrapper_kwargs \
        --training-type mutant --mutant-name "$mutant_name" \
        --mode "$mode" --num-episodes "$num_episodes" --seed "$seed" \
        --testing-strategy-name "$testing_strategy_name" \
        --testing-policy-for-training-name "$testing_policy_for_training_name" \
        --layers "$num_layers" --training-progress-filter "$filter" --sampling-size "$sampling_size" \
        --parallelize --mutant-configuration "$mutant_configuration" --num-cpus "$num_cpus"
    else
      python -m test_generation.test_agent --algo "$algo_name" --env-name "$env_name" \
        --num-runs-each-env-config "$num_runs_each_env_config" \
        --register-env --wrapper-kwargs $wrapper_kwargs \
        --training-type mutant --mutant-name "$mutant_name" \
        --mode "$mode" --num-episodes "$num_episodes" --seed "$seed" \
        --testing-strategy-name "$testing_strategy_name" \
        --testing-policy-for-training-name "$testing_policy_for_training_name" \
        --layers "$num_layers" --training-progress-filter "$filter" --sampling-size "$sampling_size" \
        --mutant-configuration "$mutant_configuration" --run-num "$run_num"
    fi
  fi
else
  echo Unknown mode: "$mode"
  exit 1
fi

