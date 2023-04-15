# Basic Implementation of Hyperparameter Optimization using Optuna library for "PPO" Reinforcement Learning Algorithm

## Usage

### Run the OptunaHypTuning.py file
Customize the following :-
 * Environment Name
 * RL Algorithm
 * Timesteps
 * Iterations
 * Execution Number
 * Number of episodes
 * Number of optuna trials
<br>
Run the file.
    ```python OptunaHypTuner.py```
<br>
Result of each study is printed in the terminal.
<br>
The study is also stored in the local database.



## Code Structure

### OptunaHypTuning.py
> ```objective``` - The objective function being optimized by Optuna.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Training <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Simulating and obtaining average reward<br><br>
> ```__name__=="__main__"``` - Create a new study and optimize the **objective** function

### Trainer.py

> * check_json_exists <br>
> #### RLAgentTrainer
>> * ```__init___``` - Load the necessary variables and update the JSON file.
>> * ```make_directories``` - Create directories to store the logs and models.
>> * ```learn``` - Instantiate a PPO model and begin training it, saving the model at the end of each iteration.


### Loader.py
> * ```__init___``` - Load the necessary variables and update the JSON file.
> * ```model_selector``` - List of all models in the directory models/env_name/algorithm/exe_number
> * ```load``` - Evaluates the performance of the model, averaging the reward over *no_of_episodes* episodes.


### PPO_HypConfig.py - RENAME THIS
> #### Learning_Rate_Scheduler
> * ``` __init__``` - MUST CONTAIN UPPER AND LOWER BOUND OF THE HYPERPARAMETER UNDER STUDY - DO THIS FOR ALL
> * ```opt``` - Return Optuna suggested learning rate



> #### Learning_Rate_Scheduler
> * ``` __init__``` - MUST CONTAIN UPPER AND LOWER BOUND OF THE HYPERPARAMETER UNDER STUDY - DO THIS FOR ALL
> * ```opt``` - Return Optuna suggested learning rate


> #### Learning_Rate_Scheduler
> * ``` __init__``` - MUST CONTAIN UPPER AND LOWER BOUND OF THE HYPERPARAMETER UNDER STUDY - DO THIS FOR ALL
> * ```opt``` - Return Optuna suggested learning rate


> #### Learning_Rate_Scheduler
> * ``` __init__``` - MUST CONTAIN UPPER AND LOWER BOUND OF THE HYPERPARAMETER UNDER STUDY - DO THIS FOR ALL
> * ```opt``` - Return Optuna suggested learning rate


> #### Learning_Rate_Scheduler
> * ``` __init__``` - MUST CONTAIN UPPER AND LOWER BOUND OF THE HYPERPARAMETER UNDER STUDY - DO THIS FOR ALL
> * ```opt``` - Return Optuna suggested learning rate


> #### Learning_Rate_Scheduler
> * ``` __init__``` - MUST CONTAIN UPPER AND LOWER BOUND OF THE HYPERPARAMETER UNDER STUDY - DO THIS FOR ALL
> * ```opt``` - Return Optuna suggested learning rate


> #### Learning_Rate_Scheduler
> * ``` __init__``` - This class is for the hyperparameter **learning_rate**
MUST CONTAIN UPPER AND LOWER BOUND OF THE HYPERPARAMETER UNDER STUDY - DO THIS FOR ALL
> * ```opt``` - Return Optuna suggested learning rate


> #### PolicySelector
> * ``` __init__``` - This class is for the hyperparameter **policy**
> * ```opt``` - Return Optuna suggested policy --> CHECK COMPATIBILITY OF DIFFERENT POLCIY WITH ALGO AND ENV


> #### StepsPerUpdate
> * ``` __init__``` - This class is for the hyperparameter **n_steps**
> * ```opt``` - Return Optuna suggested n_steps


> #### BatchSize
> * ``` __init__``` - This class is for the hyperparameter **batch_size**
> * ```opt``` - Return Optuna suggested batch_size

> #### NoOfEpochs
> * ``` __init__``` - This class is for the hyperparameter **n_epochs**
> * ```opt``` - Return Optuna suggested n_epochs

> #### DiscountFactor
> * ``` __init__``` - This class is for the hyperparameter **gamma**
> * ```opt``` - Return Optuna suggested gamma

> #### BiasVarianceTradeoff
> * ``` __init__``` - This class is for the hyperparameter **gae_lambda**
> * ```opt``` - Return Optuna suggested gae_lambda

> #### ClipRange
> * ``` __init__``` - This class is for the hyperparameter **clip_range**
> * ```opt``` - Return Optuna suggested clip_range

> #### ClipRangeVF
> * ``` __init__``` - This class is for the hyperparameter **clip_range_vf**
> * ```opt``` - Return Optuna suggested clip_range_vf

> #### NormalizeAdvantage
> * ``` __init__``` - This class is for the hyperparameter **normalize_advantage**
> * ```opt``` - Return Optuna suggested normalize_advantage

> #### EntropyCoefficient
> * ``` __init__``` - This class is for the hyperparameter **ent_coef**
> * ```opt``` - Return Optuna suggested ent_coef

> #### ValueFunctionCoefficient
> * ``` __init__``` - This class is for the hyperparameter **vf_coef**
> * ```opt``` - Return Optuna suggested vf_coef 

> #### MaxGradNorm
> * ``` __init__``` - This class is for the hyperparameter **max_grad_norm**
> * ```opt``` - Return Optuna suggested max_grad_norm rate

> #### BoolGeneralizedStateDependentExploration
> * ``` __init__``` - This class is for the hyperparameter **use_sde**
> * ```opt``` - Return Optuna suggested use_sde

> #### SDESampleFrequency
> * ``` __init__``` - This class is for the hyperparameter **sde_sample_freq**
> * ```opt``` - Return Optuna suggested sde_sample_freq

> #### TargetKL
> * ``` __init__``` - This class is for the hyperparameter **target_kl**
> * ```opt``` - Return Optuna suggested target_kl

> #### PolicyKWargs
> * ``` __init__``` - This class is for the hyperparameter **policy_kwargs**
> * ```opt``` - Returns None

> #### HypRequestHandler
> * ``` __init__``` - This class is used to return the bundled set of Hyperparameter Configuration chosen by the Sampler
> * ```optuna_next_sample``` - Return the set of Hyperparameters suggested by Optuna


