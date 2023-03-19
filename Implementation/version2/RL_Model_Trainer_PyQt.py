from random import sample
from PyQt5.QtCore import QTimer
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QLineEdit
from MegaDHyperPilotRL import *

m = MegaD26()

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

        # Create a text box
        self.textbox = QLineEdit(self)
        self.textbox.move(100, 90)

        self.policy_dropdown = QComboBox(self)
        self.policy_dropdown.addItem('MlpPolicy')
        self.policy_dropdown.addItem('nlpPolicy')
        self.policy_dropdown.setFixedWidth(100)
        self.policy_dropdown.move(100, 130)

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.move(100, 170)
        self.submit_button.clicked.connect(self.submit_selections)

        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset_gui)
        self.reset_button.move(100, 300)

        # Set window properties
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Algorithm and Environment Selection')
        self.show()

        

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

        m.second_init(self.environment, self.algorithm, self.policy)
        m.learn_and_save(100, 10)
        exit()
        m.load(1)

    def reset_gui(self):
        m = MegaD26()
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
    app.exec_()
