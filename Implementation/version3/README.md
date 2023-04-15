# PyQt5 GUI for Selecting RL Algorithms and Environments

This version of the project streamlines the selection of RL algorithms and environments with a PyQt5-based graphical user interface (GUI) that makes the process more accessible and user-friendly.

<hr>

## Usage

### Run the RLAgent_GUI.py file - for Training and Simulation

#### Training the Agent

 - Choose the **RL Algorithm** to train the model.
 - Use the _Refresh_ option to load 10 random **environments** from all the list of available envriomnents in OpenAI Gym.
 - Choose the **Policy** to be used to train the RL Agent.
 - Click the **submit button** to launch the **training** of the Agent.
 - During the training process, the GUI will be closed.
* The completion of training is notified via Python Text-To-Speech Library (pttsx3).
* The PyQt5 GUI is automatically launched again.
 
#### Observing Agent's Performance
 - Click the **Load** button to fetch all the trained environments.
 - Select an **environment** and click **Submit Environment**.
 - Select the **RL Algorithm** and click **Submit Algorithm**.
 - Select the **Execution Number** and click **Submit execution number**.
 - Select the **Model** and click **Display**.
 - Observe **Agent's Performance** with Gym's built-in game rendering.


### Run the AllEnvsChecker.py file - to check Environment Compatibility
* Stores all the configurations with error in a pandas.DataFrame.
* The DataFrame is then stored in a JSON file for future reference.


<hr>

## File Structure

### RLAgentBuilder.py
* Building the model
* Training the model
* Simulating the model's performance

### RL_AgentGUI.py
* **PyQt5 GUI** - Labels, Buttons, EventListeners and Functions.
    
### AllEnvsChecker.py
* Analyzes all possible configuration to check **compatibility issues** between Environment, Algorithm and Policy.
        
* Stores the error messages in a JSON file.

### RL_AgentGUI_TB.py - IN DEVELOPMENT
    RL_AgentGUI with a button to Launch **TensorBoard** on default browser 
        to see the Learning Curve Statistics of the model being Trained.