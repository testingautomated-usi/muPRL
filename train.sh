#!/bin/bash

env_name=
algo_name=
num_runs=10
wrapper_kwargs=
type=original
mutant_name=
search_budget=1
parallelize="false"
eval_episodes=50
num_cpus=-1
mock="false"
seed=-1
search_iteration=-1
run_num=-1

# Tensorboard is disabled by default, as it takes quite some disk space
# for debugging purposes just add "--tensorboard-log logs/tensorboard" as
# a parameter of the train_agent script in the commands below


while [ $# -gt 0 ] ; do
  case $1 in
    -a | --algo-name) algo_name="$2" ;;
    -e | --env-name) env_name="$2" ;;
    -n | --num-runs) num_runs="$2" ;;
    -t | --type) type="$2" ;;
    -m | --mutant-name) mutant_name="$2" ;;
    -s | --search-budget) search_budget="$2" ;;
    -p | --parallelize) parallelize="$2" ;;
    -c | --num-cpus) num_cpus="$2" ;;
    -k | --mock) mock="$2" ;;
    -d | --seed) seed="$2" ;;
    -i | --search-iteration) search_iteration="$2" ;;
    -r | --run-num) run_num="$2" ;;
  esac
  shift
done

if [[ "$env_name" == "CartPole-v1" ]]; then
  wrapper_kwargs="timeout_steps:500 fail_on_timeout:False"
  eval_episodes=100
elif [[ "$env_name" == "Humanoid-v4" ]]; then
  eval_episodes=100
elif [[ "$env_name" == "parking-v0" ]]; then
  eval_episodes=100
elif [[ "$env_name" == "LunarLander-v2" ]]; then
  wrapper_kwargs="timeout_steps:1000 fail_on_timeout:True"
  eval_episodes=100
else
  echo Env name "$env_name" not supported
  exit 1
fi

# do not quote the wrapper_kwargs variable
if [[ "$type" == "original" ]]; then
  if [[ "$parallelize" == "true" ]]; then
    python train_agent.py --algo "$algo_name" --env "$env_name" --eval-env --eval-episodes "$eval_episodes" \
      --log-folder logs --yaml-file hyperparams/"$algo_name".yml \
      --register-env --log-success --test-generation \
      --wrapper-kwargs $wrapper_kwargs \
      --training-type original --num-runs "$num_runs" --parallelize \
      --num-cpus "$num_cpus" --seed "$seed"
  else
    if [[ "$mock" == "true" ]]; then
      python train_agent.py --algo "$algo_name" --env "$env_name" --progress --eval-env --eval-episodes "$eval_episodes" \
        --log-folder logs --yaml-file hyperparams/"$algo_name".yml \
        --register-env --log-success --test-generation \
        --wrapper-kwargs $wrapper_kwargs \
        --training-type original --num-runs "$num_runs" --seed "$seed" \
        --mock --run-num "$run_num"
    else
      python train_agent.py --algo "$algo_name" --env "$env_name" --progress --eval-env --eval-episodes "$eval_episodes" \
        --log-folder logs --yaml-file hyperparams/"$algo_name".yml \
        --register-env --log-success --test-generation \
        --wrapper-kwargs $wrapper_kwargs \
        --training-type original --num-runs "$num_runs" --seed "$seed" \
        --run-num "$run_num"
    fi
  fi
elif [[ "$type" == "mutant" ]]; then
  if [[ "$parallelize" == "true" ]]; then
    python train_agent.py --algo "$algo_name" --env "$env_name" --eval-env --eval-episodes "$eval_episodes" \
      --log-folder logs --yaml-file hyperparams/"$algo_name".yml \
      --register-env --log-success --test-generation \
      --wrapper-kwargs $wrapper_kwargs \
      --training-type mutant --mutant-name "$mutant_name" \
      --search-budget "$search_budget" --parallelize \
      --num-cpus "$num_cpus" --seed "$seed" --search-iteration "$search_iteration"
  else
    if [[ "$mock" == "true" ]]; then
      python train_agent.py --algo "$algo_name" --env "$env_name" --progress --eval-env --eval-episodes "$eval_episodes" \
        --log-folder logs --yaml-file hyperparams/"$algo_name".yml \
        --register-env --log-success --test-generation \
        --wrapper-kwargs $wrapper_kwargs \
        --training-type mutant --mutant-name "$mutant_name" \
        --search-budget "$search_budget" --seed "$seed" --search-iteration "$search_iteration" \
        --mock --run-num "$run_num"
    else
      python train_agent.py --algo "$algo_name" --env "$env_name" --progress --eval-env --eval-episodes "$eval_episodes" \
        --log-folder logs --yaml-file hyperparams/"$algo_name".yml \
        --register-env --log-success --test-generation \
        --wrapper-kwargs $wrapper_kwargs \
        --training-type mutant --mutant-name "$mutant_name" \
        --search-budget "$search_budget" --seed "$seed" --search-iteration "$search_iteration" \
        --run-num "$run_num"
    fi
  fi
else
  echo Type "$type" not supported
  exit 1
fi


