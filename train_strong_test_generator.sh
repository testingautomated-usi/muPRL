#!/bin/bash

env_name=
algo_name=
seed=0
filter=30
num_epochs=500
learning_rate=3e-4
batch_size=64
early_stopping_patience=50
num_layers=2
test_split=0.2
mutant_name=
testing_configurations="false"
undersample=0.0
percentage_failure_discard=0.9
mutant_configuration=


while [ $# -gt 0 ] ; do
  case $1 in
    -a | --algo-name) algo_name="$2" ;;
    -e | --env-name) env_name="$2" ;;
    -s | --seed) seed="$2" ;;
    -f | --filter) filter="$2" ;;
    -n | --num-epochs) num_epochs="$2" ;;
    -l | --learning-rate) learning_rate="$2" ;;
    -b | --batch-size) batch_size="$2" ;;
    -p | --early-stopping-patience) early_stopping_patience="$2" ;;
    -y | --num-layers) num_layers="$2" ;;
    -m | --mutant-name) mutant_name="$2" ;;
    -t | --testing-configurations) testing_configurations="$2" ;;
    -u | --undersample) undersample="$2" ;;
    -c | --mutant-configuration) mutant_configuration="$2" ;;
    -d | --percentage-failure-discard) percentage_failure_discard="$2" ;;
    -k | --test-split) test_split="$2" ;;
  esac
  shift
done

if [ -z "$mutant_name" ]; then
  if [[ "$testing_configurations" == "true" ]]; then
    # we just remove the filter parameter
    python -m test_generation.train --algo "$algo_name" --env-name "$env_name" \
      --testing-policy-for-training-name mlp --test-split 0.2 \
      --seed "$seed" \
      --n-epochs "$num_epochs" \
      --learning-rate "$learning_rate" \
      --batch-size "$batch_size" \
      --weight-loss \
      --patience "$early_stopping_patience" \
      --layers "$num_layers" \
      --train-from-multiple-runs \
      --undersample "$undersample" \
      --test-split "$test_split" \
      --percentage-failure-discard "$percentage_failure_discard"
  else
    python -m test_generation.train --algo "$algo_name" --env-name "$env_name" \
      --testing-policy-for-training-name mlp --test-split 0.2 \
      --seed "$seed" --training-progress-filter "$filter" \
      --n-epochs "$num_epochs" \
      --learning-rate "$learning_rate" \
      --batch-size "$batch_size" \
      --weight-loss \
      --patience "$early_stopping_patience" \
      --layers "$num_layers" \
      --train-from-multiple-runs \
      --test-split "$test_split" \
      --undersample "$undersample"
  fi
else
  if [[ "$testing_configurations" == "true" ]]; then
    # we just remove the filter parameter
    if [ -n "$mutant_configuration" ]; then
      python -m test_generation.train --algo "$algo_name" --env-name "$env_name" \
        --testing-policy-for-training-name mlp --test-split 0.2 \
        --seed "$seed" \
        --n-epochs "$num_epochs" \
        --learning-rate "$learning_rate" \
        --batch-size "$batch_size" \
        --weight-loss \
        --patience "$early_stopping_patience" \
        --layers "$num_layers" \
        --train-from-multiple-runs \
        --mutant-name "$mutant_name" \
        --mutant-configuration "$mutant_configuration" \
        --undersample "$undersample" \
        --test-split "$test_split" \
        --percentage-failure-discard "$percentage_failure_discard"
    else
      python -m test_generation.train --algo "$algo_name" --env-name "$env_name" \
        --testing-policy-for-training-name mlp --test-split 0.2 \
        --seed "$seed" \
        --n-epochs "$num_epochs" \
        --learning-rate "$learning_rate" \
        --batch-size "$batch_size" \
        --weight-loss \
        --patience "$early_stopping_patience" \
        --layers "$num_layers" \
        --train-from-multiple-runs \
        --mutant-name "$mutant_name" \
        --undersample "$undersample" \
        --test-split "$test_split" \
        --percentage-failure-discard "$percentage_failure_discard"
    fi
  else
    if [ -n "$mutant_configuration" ]; then
      python -m test_generation.train --algo "$algo_name" --env-name "$env_name" \
        --testing-policy-for-training-name mlp --test-split 0.2 \
        --seed "$seed" --training-progress-filter "$filter" \
        --n-epochs "$num_epochs" \
        --learning-rate "$learning_rate" \
        --batch-size "$batch_size" \
        --weight-loss \
        --patience "$early_stopping_patience" \
        --layers "$num_layers" \
        --train-from-multiple-runs \
        --mutant-name "$mutant_name" \
        --mutant-configuration "$mutant_configuration" \
        --test-split "$test_split" \
        --undersample "$undersample"
    else
      python -m test_generation.train --algo "$algo_name" --env-name "$env_name" \
        --testing-policy-for-training-name mlp --test-split 0.2 \
        --seed "$seed" --training-progress-filter "$filter" \
        --n-epochs "$num_epochs" \
        --learning-rate "$learning_rate" \
        --batch-size "$batch_size" \
        --weight-loss \
        --patience "$early_stopping_patience" \
        --layers "$num_layers" \
        --train-from-multiple-runs \
        --mutant-name "$mutant_name" \
        --test-split "$test_split" \
        --undersample "$undersample"
    fi
  fi
fi





