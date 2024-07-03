#!/bin/bash

function find_mutant_configurations() {
    local algo_name_local=$1
    local env_name_local=$2
    local mutant_name_local=$3
    echo $(find logs/$algo_name_local -name "$env_name_local"_mutant_"$mutant_name_local*" | rev | cut -d"_" -f 1 | rev);
}

function train_predictor() {
    local algo_name_local=$1
    local env_name_local=$2
    local mutant_name_local=$3
    local num_layers_local=$4
    local predictor_type_local=$5
    local undersample=$6
    if [[ "$predictor_type_local" == "individual" ]]; then
        # trick to get the last match
        for mutant_conf in $(find_mutant_configurations $algo_name_local $env_name_local $mutant_name_local); do
            ./train_strong_test_generator.sh --env-name $env_name_local --algo-name $algo_name_local --seed 0 --num-epochs 500 --learning-rate 3e-4 \
                --batch-size 256 --early-stopping-patience 10 --num-layers $num_layers_local --testing-configurations true \
                --mutant-name $mutant_name_local --mutant-configuration $mutant_conf --undersample 0.4
        done
    elif [[ "$predictor_type_local" == "all" ]]; then
        ./train_strong_test_generator.sh --env-name $env_name_local --algo-name $algo_name_local --seed 0 --num-epochs 500 --learning-rate 3e-4 \
            --batch-size 256 --early-stopping-patience 10 --num-layers $num_layers_local --testing-configurations true \
            --mutant-name $mutant_name_local --undersample $undersample

    else
        echo Unknown predictor type "$predictor_type_local"
    fi
    
}

function find_killable_operators() {
    local algo_name_local=$1
    local env_name_local=$2
    local killable_operators_local=()
    # ignore the first line in the file
    for operator_comma_killable in $(tail -n +2 mutation_analysis_results/$algo_name_local/$env_name_local/killable_operators.csv); do
        if [[ $(echo $operator_comma_killable | cut -d"," -f 2) == "True" ]]; then
            killable_operators_local+=( $(echo $operator_comma_killable | cut -d"," -f 1) )
        fi
    done
    echo ${killable_operators_local[@]}
}

algo_name=
env_name=
mode=
predictor_type=

num_layers=
seed=
num_episodes=

undersample=0.0


while [ $# -gt 0 ] ; do
  case $1 in
    -a | --algo-name) algo_name="$2" ;;
    -e | --env-name) env_name="$2" ;;
    -d | --mode) mode="$2" ;;
    -p | --predictor-type) predictor_type="$2" ;;
  esac
  shift
done

seed=0
if [[ $env_name == "CartPole-v1" ]]; then
    num_layers=2
elif [[ $env_name == "parking-v0" || $env_name == "LunarLander-v2" ]]; then
    num_layers=3
elif [[ $env_name == "Humanoid-v4" ]]; then
    num_layers=3
    # In Humanoid, the number of failures is low w.r.t. the number of successes
    undersample=0.05
else
    echo Unknown env name "$env_name"
fi

if [[ "$mode" == "weak" ]]; then
    num_episodes=200
elif [[ "$mode" == "weaker" ]]; then
    num_episodes=50
elif [[ "$mode" == "strong" ]]; then
    num_episodes=100
else
    echo Unknown mode "$mode"
fi

killable_operators=$(find_killable_operators $algo_name $env_name)
echo Killable operators for $algo_name: $killable_operators

if [[ "$mode" == "weak" || "$mode" == "weaker" ]]; then
    for killable_operator in $killable_operators; do
        if [[ "$mode" == "weak" ]]; then
            ./test.sh --env-name $env_name --mode weak --mutant-name $killable_operator --num-episodes $num_episodes --seed $seed \
                --algo-name $algo_name --parallelize true
        elif [[ "$mode" == "weaker" ]]; then
            ./test.sh --env-name $env_name --mode weaker --mutant-name $killable_operator --num-episodes $num_episodes --seed $seed \
                --algo-name $algo_name --parallelize true
        fi
    done 
elif [[ "$mode" == "strong" ]]; then
    for killable_operator in $killable_operators; do
        # train predictor even if predictor_type = "individual", as it is used as a fallback in case the individual
        # predictor cannot be trained due to a low number of failures
        train_predictor $algo_name $env_name $killable_operator $num_layers "all" $undersample
        if [[ "$predictor_type" == "individual" ]]; then
            train_predictor $algo_name $env_name $killable_operator $num_layers "individual" $undersample
            for mutant_configuration in $(find_mutant_configurations $algo_name $env_name $killable_operator); do
                ./test.sh --env-name $env_name --mode strong --mutant-name $killable_operator --num-episodes $num_episodes --seed $seed --num-layers $num_layers --algo-name $algo_name \
                    --testing-configurations true --parallelize true --mutant-configuration $mutant_configuration
            done
        elif [[ "$predictor_type" == "all" ]]; then
            ./test.sh --env-name $env_name --mode strong --mutant-name $killable_operator --num-episodes $num_episodes --seed $seed --num-layers $num_layers --algo-name $algo_name \
                    --testing-configurations true --parallelize true
        else
            echo Unknown predictor type "$predictor_type"
        fi
    done
else
    echo Unknown mode "$mode"
fi
