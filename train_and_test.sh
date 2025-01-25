#!/bin/bash

env_name="CartPole-v1"
algo_name="ppo"
env_name_lower="${env_name,,}"
parallelize=false
search_budget=5
num_runs=10
# only one mutant for simplicity (gamma)
mutant_name="gamma"

start_whole=$(date +%s)

echo "Training the original $algo_name agent on the $env_name environment"
start=$(date +%s)
if [[ $parallelize == "true" ]]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false && ./train.sh --env-name $env_name --algo-name $algo_name \
        --type original --num-runs $num_runs --parallelize true \
        > logs_"$env_name_lower"_"$algo_name"_train_original_agent.txt \
        2> errors_"$env_name_lower"_"$algo_name"_train_original_agent.txt
else
    ./train.sh --env-name $env_name --algo-name $algo_name --type original --num-runs $num_runs \
        > logs_"$env_name_lower"_"$algo_name"_train_original_agent.txt \
        2> errors_"$env_name_lower"_"$algo_name"_train_original_agent.txt
fi
end=$(date +%s)
echo "Time elapsed training $algo_name on the $env_name environment: $((end-start)) seconds"

echo "Training the $mutant_name mutant of the $algo_name agent on the $env_name environment"
start=$(date +%s)
if [[ $parallelize == "true" ]]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false && ./train.sh --env-name $env_name --algo-name $algo_name \
        --type mutant --mutant-name $mutant_name --search-budget $search_budget --parallelize true \
        > logs_"$env_name_lower"_"$algo_name"_train_"$mutant_name"_mutant.txt \
        2> errors_"$env_name_lower"_"$algo_name"_train_"$mutant_name"_mutant.txt
else
    ./train.sh --env-name $env_name --algo-name $algo_name --type mutant --mutant-name $mutant_name \
        --search-budget $search_budget > logs_"$env_name_lower"_"$algo_name"_train_"$mutant_name"_mutant.txt \
        2> errors_"$env_name_lower"_"$algo_name"_train_"$mutant_name"_mutant.txt
fi
end=$(date +%s)
echo "Time elapsed training the $mutant_name mutant of the $algo_name agent on the $env_name environment: $((end-start)) seconds"

echo "Replaying the training configurations for the original $algo_name agent and its $mutant_name mutant"
start=$(date +%s)
if [[ $parallelize == "true" ]]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false && ./test.sh --env-name $env_name --algo-name $algo_name \
        --mode replay --mutant-name gamma --parallelize true > logs_"$env_name_lower"_"$algo_name"_replay.txt \
        2> errors_"$env_name_lower"_"$algo_name"_replay.txt
else
    ./test.sh --env-name $env_name --algo-name $algo_name --mode replay \
        --mutant-name gamma --delete-mutants true \
        > logs_"$env_name_lower"_"$algo_name"_replay.txt \
        2> errors_"$env_name_lower"_"$algo_name"_replay.txt
fi
end=$(date +%s)
echo "Time elapsed replaying the $mutant_name mutant of the $algo_name agent for the $env_name environment: $((end-start)) seconds"

echo "Evaluating the killability and triviality of the $mutant_name mutant of the $algo_name agent on the $env_name environment"
start=$(date +%s)
python evaluate.py --env $env_name --algo $algo_name --triv 1 --ms 0 --sens 0 --num-runs $num_runs \
    > logs_"$env_name_lower"_"$algo_name"_evaluate_killability_and_triviality.txt \
    2> errors_"$env_name_lower"_"$algo_name"_evaluate_killability_and_triviality.txt
end=$(date +%s)
echo "Time elapsed evaluating the killability and triviality of the $mutant_name mutant for the $env_name environment: $((end-start)) seconds"
echo "--- Killable mutants ---"
cat mutation_analysis_results/$algo_name/$env_name/killable_configs.csv
echo "--- Trivial mutants ---"
cat mutation_analysis_results/$algo_name/$env_name/trivial_configs.csv

echo "Building the weaker test generator for the $mutant_name mutant of the $algo_name agent on the $env_name environment"
start=$(date +%s)
if [[ $parallelize == "true" ]]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false && ./test.sh --env-name $env_name --algo-name $algo_name \
        --mode weak --mutant-name $mutant_name --num-episodes 200 --seed 0 --parallelize true \
        > logs_"$env_name_lower"_"$algo_name"_build_weaker_test_generator_"$mutant_name".txt \
        2> errors_"$env_name_lower"_"$algo_name"_build_weaker_test_generator_"$mutant_name".txt
else
    ./test.sh --env-name $env_name --algo-name $algo_name --mode weak --mutant-name $mutant_name \
        --num-episodes 200 --seed 0 \
        > logs_"$env_name_lower"_"$algo_name"_build_weaker_test_generator_"$mutant_name".txt \
        2> errors_"$env_name_lower"_"$algo_name"_build_weaker_test_generator_"$mutant_name".txt
fi
python -m test_generation.build_weaker_test_set --env-name $env_name --algo $algo_name \
    --num-configurations-to-select 50 \
    > logs_"$env_name_lower"_"$algo_name"_build_weaker_test_generator_"$mutant_name"_select_configs.txt \
    2> errors_"$env_name_lower"_"$algo_name"_build_weaker_test_generator_"$mutant_name"_select_configs.txt
end=$(date +%s)
echo "Time elapsed building the weaker test generator for the $mutant_name mutant on the $env_name environment: $((end-start)) seconds"

echo "Running the weaker test generator for the $mutant_name mutant of the $algo_name agent on the $env_name environment"
start=$(date +%s)
./test.sh --env-name $env_name --algo-name $algo_name --mode weaker --mutant-name $mutant_name --num-episodes 50 \
  > logs_"$env_name_lower"_"$algo_name"_run_weaker_test_generator_"$mutant_name".txt \
  2> errors_"$env_name_lower"_"$algo_name"_run_weaker_test_generator_"$mutant_name".txt
end=$(date +%s)
echo "Time elapsed running the weaker test generator for the $mutant_name mutant on the $env_name environment: $((end-start)) seconds"

echo "Building the strong test generator for the $mutant_name mutant of the $algo_name agent on the $env_name environment"
start=$(date +%s)
./train_strong_test_generator.sh --env-name $env_name --algo-name $algo_name --seed 0 --num-epochs 500 \
    --learning-rate 3e-4 --batch-size 256 --early-stopping-patience 10 --num-layers 3 \
    --testing-configurations true --mutant-name $mutant_name \
    > logs_"$env_name_lower"_"$algo_name"_build_strong_test_generator_"$mutant_name".txt \
    2> errors_"$env_name_lower"_"$algo_name"_build_strong_test_generator_"$mutant_name".txt
end=$(date +%s)
echo "Time elapsed building the strong test generator for the $mutant_name mutant on the $env_name environment: $((end-start)) seconds"

echo "Running the strong test generator for the $mutant_name mutant of the $algo_name agent on the $env_name environment"
start=$(date +%s)
if [[ $parallelize == "true" ]]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false && ./test.sh --env-name $env_name --algo-name $algo_name --mode strong --mutant-name $mutant_name \
        --num-episodes 100 --seed 0 --num-layers 3 --testing-configurations true --parallelize true \
        > logs_"$env_name_lower"_"$algo_name"_run_strong_test_generator_"$mutant_name".txt \
        2> errors_"$env_name_lower"_"$algo_name"_run_strong_test_generator_"$mutant_name".txt
else
    ./test.sh --env-name $env_name --algo-name $algo_name --mode strong --mutant-name $mutant_name \
        --num-episodes 100 --seed 0 --num-layers 3 --testing-configurations true \
        > logs_"$env_name_lower"_"$algo_name"_run_strong_test_generator_"$mutant_name".txt \
        2> errors_"$env_name_lower"_"$algo_name"_run_strong_test_generator_"$mutant_name".txt
fi
end=$(date +%s)
echo "Time elapsed running the strong test generator for the $mutant_name mutant on the $env_name environment: $((end-start)) seconds"

echo "Evaluating the mutation score and sensitivity of the $mutant_name mutant of the $algo_name agent on the $env_name environment"
start=$(date +%s)
python evaluate.py --env CartPole-v1 --algo ppo --triv 1 --ms 1 --sens 1 --num-runs $num_runs \
    > logs_"$env_name_lower"_"$algo_name"_evaluate_mutation_score_and_sensitivity.txt \
    2> errors_"$env_name_lower"_"$algo_name"_evaluate_mutation_score_and_sensitivity.txt
end=$(date +%s)
echo "Time elapsed evaluating the mutation score and sensitivity of the $mutant_name mutant on the $env_name environment: $((end-start)) seconds"
echo "--- Mutation scores weak test generator ---"
cat mutation_analysis_results/$algo_name/$env_name/mutation_scores_weaker.csv
echo "--- Mutation scores strong test generator ---"
cat mutation_analysis_results/$algo_name/$env_name/mutation_scores_strong.csv
echo "--- Sensitivity ---"
cat mutation_analysis_results/$algo_name/$env_name/sensitivity_weaker_strong.csv

end_whole=$(date +%s)
echo "Time elapsed for the whole process: $((end_whole-start_whole)) seconds"
