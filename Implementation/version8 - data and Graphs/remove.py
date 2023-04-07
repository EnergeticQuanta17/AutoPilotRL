import shutil
import os

if os.path.exists('logs'):
    shutil.rmtree('logs')

if os.path.exists('model'):
    shutil.rmtree('model')

dir_path = os.getcwd()
for file_name in os.listdir(dir_path):
    if file_name.endswith('.json'):
        os.remove(os.path.join(dir_path, file_name))