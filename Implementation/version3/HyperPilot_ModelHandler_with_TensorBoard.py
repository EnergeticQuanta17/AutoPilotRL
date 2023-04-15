from random import sample
from PyQt5.QtCore import QTimer
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QLineEdit
from RLAgentBuilder import *

global loading_pressed

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create labels
        algorithm_label = QLabel('Algorithm:', self)
        algorithm_label.move(20, 20)
        environment_label = QLabel('Environment:', self)
        environment_label.move(20, 60)
        policy_label = QLabel('Policy:', self)
        policy_label.move(20, 130)

        # Create dropdowns
        self.algorithm_dropdown = QComboBox(self)
        self.algorithm_dropdown.addItem('A2C')
        self.algorithm_dropdown.addItem('DDPG')
        self.algorithm_dropdown.addItem('DQN')
        self.algorithm_dropdown.addItem('PPO')
        self.algorithm_dropdown.addItem('SAC')
        self.algorithm_dropdown.addItem('TD3')
        self.algorithm_dropdown.move(100, 20)

        self.environment_dropdown = QComboBox(self)
        self.environment_dropdown.setFixedWidth(260)
        self.environment_dropdown.move(100, 60)

        self.refresh_button = QPushButton('Refresh', self)
        self.refresh_button.move(400, 60)
        self.refresh_button.clicked.connect(self.refresh_environment)

        self.exe_button = QPushButton('Submit execution number', self)
        self.exe_button.move(180, 340)
        self.exe_button.clicked.connect(self.execution_submit)

        self.show_logs_button = QPushButton('Show logs on TensorBoard', self)
        self.show_logs_button.move(180, 340)
        self.show_logs_button.clicked.connect(self.show_logs)

        # Create a text box
        self.textbox = QLineEdit(self)
        self.textbox.move(100, 90)

        self.policy_dropdown = QComboBox(self)
        self.policy_dropdown.addItem('MlpPolicy')
        self.policy_dropdown.addItem('nlpPolicy')
        self.policy_dropdown.setFixedWidth(100)
        self.policy_dropdown.move(100, 130)

        self.models_label = QLabel('Record model per Timestep:', self)
        self.models_label.move(250, 135)
        
        self.timestep_textbox = QLineEdit(self)
        self.timestep_textbox.move(430, 130)
        self.timestep_textbox.setFixedWidth(80)

        self.models_label = QLabel('Iterations:', self)
        self.models_label.move(580, 135)

        self.iterations_textbox = QLineEdit(self)
        self.iterations_textbox.move(650, 130)
        self.iterations_textbox.setFixedWidth(80)

        self.load_button = QPushButton('Load', self)
        self.load_button.move(20, 260)
        self.load_button.clicked.connect(self.loader)

        self.load_env_dropdown = QComboBox(self)
        self.load_env_dropdown.setFixedWidth(260)
        self.load_env_dropdown.move(125, 265)

        self.load_button = QPushButton('Submit Environment', self)
        self.load_button.move(400, 260)
        self.load_button.clicked.connect(self.submit_env)

        self.load_algo_dropdown = QComboBox(self)
        self.load_algo_dropdown.move(570, 265)

        self.load_button = QPushButton('Submit Algorithm', self)
        self.load_button.move(670, 260)
        self.load_button.clicked.connect(self.submit_algo)

        self.load_button = QPushButton('Display', self)
        self.load_button.move(800, 300)
        self.load_button.clicked.connect(self.display)

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.move(200, 170)
        self.submit_button.clicked.connect(self.submit_selections)

        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset_gui)
        self.reset_button.move(100, 170)

        self.models_label = QLabel('Execution number:', self)
        self.models_label.move(150, 305)

        self.models_label = QLabel('Model:', self)
        self.models_label.move(370, 305)

        self.loader_dropdown = QComboBox(self)
        self.loader_dropdown.move(270, 300)

        self.model_dropdown = QComboBox(self)
        self.model_dropdown.move(410, 300)
        self.model_dropdown.setFixedWidth(300)

        # Set window properties
        self.setGeometry(300, 300, 1000, 400)
        self.setWindowTitle('Algorithm and Environment Selection')
        self.show()

    def display(self):
        model_dir = f"model/{self.m.env_name}/{self.m.algorithm}"
        self.m.load(2, True, all_files=os.listdir(f"{model_dir}/{self.loader_dropdown.currentText()}"), training_till=self.model_dropdown.currentText(), idk=f"{model_dir}/{self.loader_dropdown.currentText()}", env=gym.make(self.m.env_name))

    def submit_env(self):
        self.m.env_name = self.load_env_dropdown.currentText()
        for i in os.listdir("model/"+self.m.env_name):
            self.load_algo_dropdown.addItem(i)
        
    
    def submit_algo(self):
        self.m.algorithm = self.load_algo_dropdown.currentText()
        for i in os.listdir(f"model/{self.m.env_name}/{self.m.algorithm}"):
            self.loader_dropdown.addItem(i)

    def execution_submit(self):
        model_dir = f"model/{self.m.env_name}/{self.m.algorithm}"
        for i in os.listdir(f"{model_dir}/{self.loader_dropdown.currentText()}"):
            self.model_dropdown.addItem(i)

    def show_logs(self):
        pass
                

    def loader(self):
        print("before   ")
        self.m = RLAgent(False)
        self.m.third_init()
        print("after")
        loading_pressed = False
        for i in os.listdir("model"):
            self.load_env_dropdown.addItem(i)
        print("after")
        
    def refresh_environment(self):
        # update the environment list with 10 random strings
        self.update_env_list()

    def update_env_list(self):
        # list of environment names
        env_list = all_environments_latest_version
        # select 10 random environment names
        self.environments = sample(env_list, 10)
        # clear the current environment dropdown list
        self.environment_dropdown.clear()
        # add the random environment names to the dropdown list
        for env in self.environments:
            self.environment_dropdown.addItem(env)
        # print(f"Selected environments: {self.environments}")

    def submit_selections(self):
        if(self.textbox.text() not in all_environments_latest_version and self.environment_dropdown.currentText() not in all_environments_latest_version):
            return
        
        input_success = QLabel('Input Success !', self)
        input_success.move(200, 200)

        # Disable dropdowns and buttons
        self.algorithm_dropdown.setDisabled(True)
        self.environment_dropdown.setDisabled(True)
        self.refresh_button.setDisabled(True)
        self.submit_button.setDisabled(True)
        self.textbox.setDisabled(True)
        self.policy_dropdown.setDisabled(True)

        # Get selected values
        self.algorithm = self.algorithm_dropdown.currentText()
        self.environment = self.environment_dropdown.currentText()
        self.policy = self.policy_dropdown.currentText()
        if(self.environment==""):
            self.environment = self.textbox.text()

        print('Algorithm selected:', self.algorithm)
        print('Environment selected:', self.environment)
        print('Policy selected:', self.policy)

        self.m = RLAgent(False)
        self.m.third_init()
        self.m.second_init(self.environment, self.algorithm, self.policy)
        self.m.learn_and_save(int(self.timestep_textbox.text()), int(self.iterations_textbox.text()))
        #m.load()

    def reset_gui(self):
        m = RLAgent(False)
        m.third_init()
        # Reset values of GUI elements
        self.algorithm_dropdown.setCurrentIndex(0)
        self.environment_dropdown.setCurrentIndex(0)
        self.policy_dropdown.setCurrentIndex(0)
        self.textbox.setText('')

        # Reset variables
        self.algorithm = ''
        self.environment = ''
        self.policy =''

        self.algorithm_dropdown.setDisabled(False)
        self.environment_dropdown.setDisabled(False)
        self.refresh_button.setDisabled(False)
        self.submit_button.setDisabled(False)
        self.textbox.setDisabled(False)
        self.policy_dropdown.setDisabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())

