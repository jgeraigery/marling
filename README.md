<div align="center">

# MARLING ⚽

**Improving Sample Efficiency in Multi-Agent Reinforcement Learning for Simulated Football Games via Exploration**

[![arXiv](https://img.shields.io/badge/arXiv-2503.13077-b31b1b)](https://arxiv.org/abs/2503.13077)
[![License](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)](LICENSE.md)

</div>

This repository contains the code, improvements on training sample efficiency, and functionalities to train and evaluate multi-agent reinforcement learning methods based on the [TiZero Method](https://arxiv.org/abs/2302.07515) for football simulations. Three versions of TiZero are implemented:
- Standard TiZero (w/ minor architectural changes compared to the original paper)
- TiZero with the [Random Network Distillation (RND)](https://arxiv.org/abs/1810.12894) exploration bonus
- TiZero with the [Self-supervised Online Intrinsic Reward](https://proceedings.neurips.cc/paper_files/paper/2022/hash/266c0f191b04cbbbe529016d0edc847e-Abstract-Conference.html) and the RND exploration bonus

The project builds on the framework from [Boosting Studies of Multi-Agent Reinforcement Learning on Google Research Football Environment](https://arxiv.org/abs/2309.12951), built on top of [Google Research Football](https://arxiv.org/abs/1907.11180) and PyTorch. See [`README_lightmalib.md`](README_lightmalib.md) for the upstream docs.

## Citation
If you use this code or build on our work, please cite:
```bibtex
@article{baghi2025towards,
  title={Towards Better Sample Efficiency in Multi-Agent Reinforcement Learning via Exploration},
  author={Baghi, Amir and Sj{\"o}lund, Jens and Bergdahl, Joakim and Gissl{\'e}n, Linus and Sestini, Alessandro},
  journal={arXiv preprint arXiv:2503.13077},
  year={2025}
}
```

## Contents
- [Citation](#citation)
- [Install](#install)
- [Running Training Sessions](#running-training-sessions)
- [Running The Google Research Football Environment](#running-the-google-research-football-environment)
- [Scenarios and Further Customization](#scenarios-and-further-customization)
- [Generating FDG26 Paper's Plots, Table CSV Files, and Gameplay Evaluations](#generating-fdg26-papers-plots-table-csv-files-and-gameplay-evaluations)
- [New Training Environment Initialization and Running Scripts](#new-training-environment-initialization-and-running-scripts)
- [License](#license)
- [Contact](#contact)

## Install
You can use any tool to manage your python environment, for example, either virtualenv or Conda. The steps for Conda are as follows:

1. Clone the repository:
```bash
git clone https://github.com/electronicarts/marling.git
```

2. Follow the instructions in the official repo https://github.com/google-research/football and install the Google Research Football environment. Use "Option a. From PyPi package (recommended)". On Windows, if the PyPi install fails, fall back to their [engine compilation](https://github.com/google-research/football/blob/master/gfootball/doc/compile_engine.md) instructions (compilation can take a few minutes).
 
3. Create the Conda environment:
```bash
conda create -n marling_env -f environment.yml  
```

4. Activate the environment:
```bash
conda activate marling_env
```

If using Conda: (Based on instructions given for installing packages with pip in a Conda env [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)) When installing the `gfootball` environment using pip, using the command `python3 -m pip install .` given in the instructions for installation via source, make sure to add the flag `--upgrade-strategy only-if-needed` to the pip command, making this command in total:

```
python3 -m pip install --upgrade-strategy only-if-needed .
```

Use `--upgrade-strategy only-if-needed` for any pip command that you use while your Conda env is activated.

If you encounter any problems with installing the `gfootball` environment dependencies (e.g., gym==0.21.0), try 
```
pip install setuptools==65.5.0 
```

*Note 1 (Windows PYTHONPATH)*: You may need to add the cloned repo to `PYTHONPATH` so `light_malib` can be imported. In PowerShell: `$env:PYTHONPATH += ";C:\path\to\repo"`. In cmd: `set PYTHONPATH=%PYTHONPATH%;C:\path\to\repo`. Consider adding this to your environment's activation script.

*Note 2 (Windows manual gfootball compile)*: If you are manually compiling GRF on Windows, the engine's install script uses `vcpkg`. Set `VCPKG_ROOT` in that script to your local `vcpkg` install. `vcpkg` also reads a manifest whose package versions must match your Python version; a working Python 3.10 manifest is provided in this repo ([`vcpkg.json`](vcpkg.json)) and can be copied into GRF's manifest directory.

[Return to Contents](#contents)

## Running Training Sessions

The configurations for the training sessions are stored in the `expr_configs` directory in the ```yaml``` format. The three configuration files for the final three versions of TiZero are stored under the `expr_configs/tizero` directory. More specifically, the three versions are:
- Standard Version: 'tizero_full_training_10v10_vanilla_tizero.yaml'
- RND Version: 'tizero_full_training_10v10_tizero_rnd.yaml'
- Self-supervised Version: 'tizero_full_training_10v10_tizero_ssir.yaml' (or 'tizero_full_training_10v10_tizero_ssir_and_rnd.yaml' for SSIR combined with RND)

In each of these files, in addition to the regular attributes for the TiZero method, one can configure the use of the intrinsic and exploration bonus terms via these attributes:
- `use_intrinsic_reward`: Set to `True` to use the self-supervised intrinsic reward term
- `use_exploration_bonus`: Set to `True` to use the RND exploration bonus term

And then, for each of these additional rewards, one can configure the following attributes:
- `intrinsic_reward_update_freq`: The frequency of updating the intrinsic reward term
- `intrinsic_reward_lr`: The learning rate for the intrinsic reward term
- `clipping_epsilon`: Basically the weight of the intrinsic reward term
- `exploration_bonus_update_freq`: The frequency of updating the exploration bonus term
- `exploration_bonus_lr`: The learning rate for the exploration bonus term
- `exploration_bonus_coefficient`: The coefficient for the exploration bonus term
- `use_intrinsic_after`: The number of training epochs after which to start using the intrinsic and exploration bonus terms

You can also set a maximum number of global rollout steps (i.e., the number of rollouts after which training gets terminated) through the `max_global_rollout_steps` option.

To run a training session, simply run the following command, for example, for the standard version of TiZero:
```bash
python light_malib/main_tizero.py --config "./expr_configs/tizero/tizero_full_training_10v10_vanilla_tizero.yaml"
```
The configuration file should be changed to the desired version of TiZero.

During the training session, the console will display information regarding the current phase of the training (Curriculum, Challenge, Generalize) and the current scenario difficulty, as well as other more detailed information regarding the training process.

The TensorBoard event file and the checkpoints of the model will be regularly saved in the `logs` directory, in the subdirectory specified by `expr_name` in the configuration file.

Note that you can also utilize multiple GPUs for training by setting the `num_trainers` attribute for the `training_manager` in the configuration file to the desired number of GPUs.

Similar to the scenarios (described later), the fixed reward signals and their corresponding weights can be set in the session configuration file. The currently implemented signals are:
- `out_of_bound_penalty`: The penalty for going out of bounds
- `gather_penalty`: The penalty for gathering too close to each other
- `pass_reward`: The reward for passing the ball
- `hold_ball`: The reward for holding the ball
- `official_reward`: The reward for scoring a goal, (official reward as given by Google Research Football)
- `total_dist_to_ball_reward`: The reward for the total distance to the ball (often used for testing purposes)
- `different_action_reward`: The reward for taking different actions (often used for testing purposes)
Similar to the scenarios, make sure to update the partially hardcoded object in `tizero_runner.py` to ensure including the new reward signals.

[Return to Contents](#contents)

## Running The Google Research Football Environment

To run the environment between two different controlling mechanisms which can be user controls, built-in AI, or trained agents, this command can be used: 

```bash
python -m gfootball.play_game --players "{left_controller}:left_players=n;{right_controller}:right_players=m[,checkpoint=$YOUR_PATH]"
```

where `left_controller` and `n` are the controller mode (e.g. keyboard or a trained policy) and number of players controlled, respectively. Similarly, the controller and number of players for the right team can be chosen. Also, if a trained agent is used, a checkpoint file can be specified by providing `checkpoint=$YOUR_PATH` (in the command, the brackets indicate that it is optional to provide a checkpoint.) 

**NOTE 1**: If using a trained *agent*, make sure to use the `football_ai_light.py` controller script provided in `light_malib/scripts` by replacing `left_controller` or `right_controller` in the commands with `football_ai_light`. Furthermore, when using `football_ai_light`, the controller script should be placed in your virtual/conda environment's `site-packages` directory, i.e., `venv/Lib/site-packages/gfootball/env/players`, for the Google Research Football environment to recognize it.

**NOTE 2**: If using a trained *agent*, ensure that the string `agent` is not included in the path to the checkpoint files, as the `play_game.py` in the `gfootball` package includes an assertion that the path should not contain the string `agent`. Therefore, either remove the string `agent` from the path or modify the assertion in the `play_game.py` script in the original Google Research Football package.

### Enabling or Disabling Exploration
If using a trained policy which outputs a categorical distribution for the actions, one can test the deterministic behavior of the policy (i.e., taking the `argmax` of the action probability distribution) versus its stochastic behavior (i.e., sampling from the distribution). To enable or disable exploration (i.e., sampling), use the following command:

```bash
python -m gfootball.play_game --players "{left_controller}:left_players=n;{right_controller}:right_players=m[,explore=True|False,checkpoint=$YOUR_PATH]"
```

In this command, `explore=True` enables exploration, meaning actions are sampled from the policy's probability distribution, which can provide a more varied behavior. Conversely, `explore=False` disables exploration, causing the agent to always take the action with the highest probability (deterministic behavior). The placement of `explore=True|False` within the brackets indicates that it is an optional parameter, similar to the `checkpoint` option. The use of `True|False` clearly shows the two possible values for the `explore` switch.

### Customizing Environment Settings
One can customize the environment settings, control the rendering, and choose whether to render in real-time or at a slower speed using the switches in the following command example:

```bash
python -m gfootball.play_game --players "{left_controller}:left_players=n;{right_controller}:right_players=m[,checkpoint=$YOUR_PATH]" --level "full_game_10_vs_10_challenge" --render=true --real_time=false
```

In this command, the `--level` switch sets the environment. The `--render` switch enables visual rendering of the game, while `--real_time` runs the game at a slower speed instead of the speed at which the environment is processed, which is useful for detailed analysis or debugging.

### Running with Built-in AI for One Team
One can only provide the player specifications for one team/side as such:

```bash
python -m gfootball.play_game --players "{right_controller}:right_players=m[,checkpoint=$YOUR_PATH]"
```

Here, the other team will be controlled by the built-in AI by default. Therefore, if no player specifications are provided, a match, where both of the teams are controlled by the built-in AI, will be run.

### Example Run
An example run of the environment, in the final *Challenge* scenario, with a trained agent and built-in AI can be seen below:
```bash
python -m gfootball.play_game --players "football_ai_light:left_players=10,explore=True,checkpoint=experiments/rnd/checkpoint" --action_set=default --level "full_game_10_vs_10_challenge" --real_time=false --render=true
```

### Systematic Evaluation

To systematically evaluate the performance of a trained agent against the baseline built-in AI, the following command can be used for example:

```bash
python eval_against_baseline.py --dir .\experiments\rnd\checkpoint --n 10 --threads 10 --output results_rnd.csv
```

In this command, the `--dir` switch specifies the directory of the checkpoint files, the `--n` switch specifies the number of games to be played for each seed (10 seeds are evaluated by default by the script), the `--threads` switch specifies the number of threads to be used for parallel evaluation of the games (lower or equal to the number of games), and the `--output` switch specifies the output file for the results in the form of a `csv` file, where the average main metrics (e.g, win, loss, etc.) and additional ones (e.g., passing, shooting, etc.) are provided.

On the note of using different difficulties of the built-in baseline AI, the [Google Research Football paper](https://arxiv.org/abs/1907.11180) suggests using *player strength factors* of 0.05 for easy, 0.6 for medium, and 0.95 for hard difficulty of the built-in AI. This can be configured by changing `builder.config().right_team_difficulty` in the scenario script `full_game_10_vs_10_challenge.py` in the `scenarios` directory, both in your local copy and in the `gfootball` package in your Python environment (similar to customizing scenarios as described later). 

[Return to Contents](#contents)

## Scenarios and Further Customization

The scenarios for the training sessions are defined in `light_malib/envs/gr_football/scenarios` as Python scripts with values determining the attributes of the scenario and the positions of the players. The `10v10` scenarios (i.e., including 10 controlled players in each team and the goalkeepers) used in this project are defined as `full_game_10_vs_10_challenge.py` for the Challenge and Generalize scenario and `full_game_10_vs_10_diff_{n}.py` for the Curriculum scenario, where `n` is the difficulty level of the scenario.

More scenarios can be defined by following the structure of the existing scenarios and adding them to the `scenarios` directory. Note that if new scenarios are to be used, in addition to the training configurations, a partially hardcoded object in `tizero_runner.py` should be updated to include them. Also include your newly defined or modified scenarios in the `gfootball` package in your Python environment (e.g., `venv/Lib/site-packages/gfootball/envs/scenarios`) so that the Google Research Football environment can recognize them.

[Return to Contents](#contents)

## Generating FDG26 Paper's Plots, Table CSV Files, and Gameplay Evaluations

In the `experiments_scripts/` directory, we provide scripts to generate the CSV files with the metrics used to generate the training experiments table in the paper, as well as the script to generate the plot of two experiments against each other (used in the paper to plot the best TiZero against the best TiZero-RND).

The script generating the training table CSV files can be invoked as such:
```
python .\experiments_scripts\get_training_table_marling.py
```
and it will generate a CSV file in the same directory containing the aggregated metrics for each method over the different experiment seeds provided in the corresponding directories, e.g., `experiments/rnd/`.

Furthermore, the plots are generated as such:
```
python .\experiments_scripts\get_training_plot_marling.py
```

The directory of the two experiments to be plotted are hardcoded inside the script; make sure to update if needed to be changed.

Finally, the gameplay evaluations against the baseline AI, which runs a number of games per best-trained model for each seed and method, is run as such:
```
python .\eval_multiple_models_against_baseline.py --config evaluation_cfg.json --games 50 --threads 5 --output eval_results
```
where the models to be evaluated are placed in `evaluation_cfg.json` (see current `evaluation_cfg_backup.json` as an example), and the script runs 50 games per model using 5 threads in parallel and writes out the gameplay metrics in CSV format for each method in an individual file, and aggregated over the models of one method and brought as a row in a compound file in `eval_results/`. Note that increasing the number of threads more than the existing CPU threads may deteriorate performance.

Furthermore, if there is a new model added and one wants to update the existing results in `eval_results\` with the newly added model without having to rerun the evaluations for all existing models, the `--update-evaluations` can be used:
```
python .\eval_multiple_models_against_baseline.py --config evaluation_cfg.json --games 50 --threads 5 --output eval_results --update-evaluations
```

[Return to Contents](#contents)

## New Training Environment Initialization and Running Scripts (For Linux)

For quickly setting up the training environment and running training experiments on a Linux machine, as was done for the paper, bash scripts are provided.

To initialize the training environment without having to manually go through the instructions provided in [Install](#install), you can run the `init_env.sh` script.

Furthermore, to run a training session for a certain experiment, you can run, example:
```
./train_marling.sh ./expr_configs/tizero/tizero_full_training_10v10_tizero_rnd.yaml
```
which takes the experiment config file of the experiment you are interested in.

[Return to Contents](#contents)

## Authors

<div align="center">
<b>Search for Extraordinary Experiences Division (SEED) - Electronic Arts
<br>
<a href="https://seed.ea.com">seed.ea.com</a>
<br>
<a href="https://seed.ea.com"><img src="seed-logo.png" width="150px"></a>
<br>
SEED is a pioneering group within Electronic Arts, combining creativity with applied research.</b> <br>
We explore, build, and help define the future of interactive entertainment.
</p>
</div>

MARLING is a SEED (Electronic Arts) project created by [Amir Baghi](https://github.com/amirsbag).

## License

Project MARLING is open source licensed, see LICENSE for details.

Project MARLING utilizes open source software, see NOTICE for details.

[Return to Contents](#contents)

## Contact
If you have any questions, feel free to contact me at [email](mailto:abaghi@ea.com).
