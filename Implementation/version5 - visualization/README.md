# HyperPilotRL: RL Hyperparameter Framework

## Usage

### Run the HyperPilotRL.py file

##### Customize the following :-
 * Environment Name
 * RL Algorithm
 * Timesteps
 * Iterations
 * Execution Number
 * Number of episodes
 * Number of optuna trials

##### Choose the plots to display :-
* Optimization History
* Timeline
* Slice
* Pareto Font
* Params Importances
* Parallel Coordinates
* Intermediate Values
* Empirical Distribution Function
* Contour


##### Run the code
```
python HyperPilotRL.py
```
* The result of each study is printed in the terminal.
* The study is also stored in the local database.


<hr>

## Code Structure

### HyperPilotRL.py
> ##### **class HyperPilotRL**
> * ```__init__``` - Stores the selected options and redirects to the selected optimizer to initiate Hyperparameter Optimization.
> * ```optuna ```  - Creates an instance of **OptunaTuner**.<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Initiates the Optuna Optimizer. <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Prints the summary of the completed study.<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Generates HTML files of plots  related to the hyperparameter optimization process carried out using Optuna.
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 


### OptunaHypTuning.py
> ##### **class OptunaTuner**
> * ```__init__``` - Stores the selected options<br>
> * ```create_db``` - Create an SQLite Database at the specified location.<br>
> * ```select_pruner``` - Choose from the list of available **pruners** in Optuna.<br>
> * ```select_sampler``` - Choose from the list of available **samplers** in Optuna.<br>
> * ```study_loader``` - Load the study according to the class parameters.<br>
> * ```study_summaries``` - Iteratively print through the summaries of each study.<br>
> * ```return_study``` - Return a **Study** object according to the specified class parameters.
> * ```call_optuna``` - Initiates the Optuna Hyperparameter Optimization.<br>
> * ```visul``` - Generates HTML plots describing the study.<br>


### Trainer.py

> * check_json_exists <br>
> ##### **class RLAgentTrainer**
>> * ```__init___``` - Load the necessary variables and update the JSON file.
>> * ```make_directories``` - Create directories to store the logs and models.
>> * ```learn``` - Instantiate a PPO model and begin training it, saving the model at the end of each iteration.


### Loader.py
> ##### **class RLAgentLoader**
> * ```__init___``` - Load the necessary variables and update the JSON file.
> * ```model_selector``` - List of all models in the directory models/env_name/algorithm/exe_number
> * ```load``` - Evaluates the performance of the model, averaging the reward over *no_of_episodes* episodes.