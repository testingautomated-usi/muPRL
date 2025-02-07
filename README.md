# 1. Setting up the environment

## 1.1 Download the docker image

If you do not have an NVIDIA GPU type:

```commandline

docker run -it -u ${UID} --rm --mount type=bind,source="$(pwd)",target=/home/muPRL --workdir /home/muPRL --name muPRL-container dockercontainervm/muprl-cpu:0.1.0

```

The command will download the container image `dockercontainervm/muprl-cpu:0.1.0` that should be around 2.6 GB.

If you have an NVIDIA GPU, make sure to install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and type:

```commandline

docker run --gpus all -it -u ${UID} --rm --mount type=bind,source="$(pwd)",target=/home/muPRL --workdir /home/muPRL --name muPRL-container dockercontainervm/muprl-gpu:0.1.0

```

The command will download the container image `dockercontainervm/muprl-gpu:0.1.0` that should be around 8 GB.

## 1.2 (Optional): Build the docker container instead of step 1.1

If you do not have an NVIDIA GPU type:

```commandline

docker build -f Dockerfile_CPU -t muprl:latest .
docker run -it -u ${UID} --rm --mount type=bind,source="$(pwd)",target=/home/muPRL --workdir /home/muPRL --name muPRL-container muprl:latest

```

The image size should be around 2.6 GB.

If you have an NVIDIA GPU, make sure to install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and type:

```commandline

docker build -f Dockerfile_GPU -t muprl:latest .
docker run --gpus all -it -u ${UID} --rm --mount type=bind,source="$(pwd)",target=/home/muPRL --workdir /home/muPRL --name muPRL-container muprl:latest

```

The image size should be around 8 GB.

## 1.3 (Optional): Use VSCode Devcontainer instead of step 1.1

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install DevContainer Extension;
- In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command;
- You will be prompted with two options: CPU dev container or GPU dev container. Choose the one you want to run.

# 2. Mutation testing pipeline

An example with the CartPole environment (the testing machine has a QUADRO GPU with 8GB of memory). For simplicity, we only execute five runs (instead of ten) and sample three mutant configurations (instead of five):

# 2.1 Train original model

```commandline

./train.sh --env-name CartPole-v1 --algo-name ppo --type original --num-runs 5

```

The previous command trains five cartpole models sequentially (the hyperparameters are in hyperparams/ppo.yml). To train in parallel pass `--parallelize true`; if you are training on the GPU, prepend the command with `export XLA_PYTHON_CLIENT_PREALLOCATE=false` such that Jax allocates the GPU memory dynamically (otherwise, by default Jax allocates 75% of the GPU to a single process, resulting in an OOM error). With `--parallelize=true` it takes ~7 minutes.

# 2.2 Train the gamma mutant

``` commandline

./train.sh --env-name CartPole-v1 --algo-name ppo --type mutant --mutant-name gamma --search-budget 3 --seed 0

```

Trains the gamma mutant (e.g., values sampled randomly 3 times) for the same number of runs using the same configurations). The seed is to make the run repeatable (it should generate 0.9, 0.45, 0.7).
As before, the `--parallelize=true` commandline argument, parallelizes the five runs for each sampled mutant configuration. With `--parallelize=true` it takes ~16 minutes.

# 2.3 Replay the training configurations

```commandline

./test.sh --env-name CartPole-v1 --algo-name ppo --mode replay --mutant-name gamma --delete-mutants true

```

Replays the training configurations for both the original and the mutated agents 
if configurations were already replayed, it will not repeat them). It keeps only the mutant configuration
that is killable and closer to the original value of the mutation operator (i.e., 0.9 in the example). It takes ~15 minutes.

# 2.4 Build weaker test generator

```commandline

./test.sh --env-name CartPole-v1 --algo-name ppo --mode weak --mutant-name gamma --num-episodes 200 --seed 0

```

Samples 200 configurations at random and execute them (if configurations are already executed, it will not repeat them). It takes ~5 minutes.

```commandline

python -m test_generation.build_weaker_test_set --env-name CartPole-v1 --algo ppo --num-configurations-to-select 50

```

Selects 50 weaker configurations.

```commandline

./test.sh --env-name CartPole-v1 --algo-name ppo --mode weaker --mutant-name gamma --num-episodes 50

```

Executes the weaker test set. It takes ~50 seconds.

# 2.5 Build strong test generator

```commandline

./train_strong_test_generator.sh --env-name CartPole-v1 --algo-name ppo --seed 0 --num-epochs 500 --learning-rate 3e-4 \
    --batch-size 256 --early-stopping-patience 10 --num-layers 3 --testing-configurations true --mutant-name gamma

```

Trains the failure predictor (CPU only supported). It takes ~30 seconds.

```commandline

./test.sh --env-name CartPole-v1 --algo-name ppo --mode strong --mutant-name gamma --num-episodes 100 --seed 0 --num-layers 3 --testing-configurations true

```

Executes the strong test generator with trained predictor for 100 episodes. It takes ~5 minutes.

# 2.6 Analyze the results

```commandline

python evaluate.py --env CartPole-v1 --algo ppo --triv 1 --ms 1 --sens 1 --num-runs 5

```

The logs are written into `./mutation_analysis_results/ppo/CartPole-v1`. In the `sensitivity_weaker_strong.csv` file, we report the mutation scores of `weaker` and `strong` test generators. In this case we have that the mutation score of the `weaker` test generator is `0.4`, while the mutation score of the `strong` test generator is `0.6`; hence the sensitivity is `0.333` as shown in the last row.

# 2.7 Script

The complete script to replicate the CartPole results is [here](train_and_test.sh). The whole process on a Quadro GPU with 8GB of memory and with parallelization `true` takes around 120 minutes.

# 3. Mutation operators implemented in muPRL

## 3.1 List of mutation operators

| **Mutation operator**        | **PPO**            | **DQN**            | **SAC**            | **TQC**            |
| ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| ent_coef (SEC)               | :heavy_check_mark: | :x:                | :x:                | :x:                |
| exploration_final_eps (SMR)  | :x:                | :heavy_check_mark: | :x:                | :x:                |
| gamma (SDF)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| learning_starts (SLS)        | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| n_steps (SNR)                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| n_timesteps (NEI)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| target_update_interval (SNU) | :x:                | :heavy_check_mark: | :x:                | :x:                |
| tau (SPV)                    | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: |

## 3.2 Search space

| **Mutation operator**        | **Hyperparameter search space**               | **Mutation search space**                                         | 
| ---------------------------- | --------------------------------------------- | ----------------------------------------------------------------- | 
| ent_coef (SEC)               | float, (0.00000001, 0.1)                      | float, (0.01, 0.2), *We changed the lower bound to reduce the search space, rounding to two floating point digits.*                                               | 
| exploration_final_eps (SMR)  | float, (0.0, 0.2)                             | float, (0.0, 1.0), *1.0 is the theoretical maximum for this parameter.*                                                 | 
| gamma (SDF)                  | [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999] | [0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999], *0.45 is 50% less than the lowest value 0.9 in the original space. The theoretical minimum is 0.0, i.e., no discounting. The theoretical maximum is 1.0.* | 
| learning_starts (SLS)        | [0, 1000, 5000, 10000, 20000]                 | [0.5, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], *Transformed into percentages of training time steps; upper bound is 50% of the original training time steps.*                   | 
| n_steps (SNR)                | [8, 16, 32, 64, 128, 256, 512, 1024, 2048]    | [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072], *2 is 75% less than the lowest value 8 in the original space, and 3072 is 50% than the highest value 2048 in the original space. The theoretical minimum is 1.*            | 
| n_timesteps (NEI)            | N/A                                           | [10 -- 95, 5], *From 10% to 95% of the original training time steps, with a step size of 5%.*                                                     | 
| target_update_interval (SNU) | [1, 1000, 5000, 10000, 15000, 20000]          | [0.5, 1, 5, 10, 15, 20, 25, 30], *Transformed into percentages of training time steps; upper bound is 30% of the original training time steps.*                                   | 
| tau (SPV)                    | [0.001, 0.005, 0.01, 0.02, 0.05, 0.08]        | [0.0005, 0.0008, 0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.12], *0.12 is 50% more than 0.08 (i.e., highest value in original space); 0.0005 is 50% less than 0.001 (i.e., lowest value in original space).* | 

# 4. Trained agents

We share the agents for each environment as well as the associated killable and non-trivial mutants with the results mutation analysis. The data are on [Figshare](https://figshare.com/s/933cbcd8681c00e0ff21).

## 5. Citing the Project

To cite this repository in publications:

```bibtex
@article{DBLP:journals/corr/abs-2408-15150,
  author       = {Deepak{-}George Thomas and
                  Matteo Biagiola and
                  Nargiz Humbatova and
                  Mohammad Wardat and
                  Gunel Jahangirova and
                  Hridesh Rajan and
                  Paolo Tonella},
  title        = {muPRL: {A} Mutation Testing Pipeline for Deep Reinforcement Learning
                  based on Real Faults},
  journal      = {CoRR},
  volume       = {abs/2408.15150},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2408.15150},
  doi          = {10.48550/ARXIV.2408.15150},
  eprinttype    = {arXiv},
  eprint       = {2408.15150},
  timestamp    = {Mon, 30 Sep 2024 21:31:24 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2408-15150.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
The paper has been recently accepted at [ICSE 2025](https://conf.researchr.org/track/icse-2025/icse-2025-research-track#Accepted-papers-First-and-Second-Cycle). We will update the citation once the paper will be published in the proceedings.

