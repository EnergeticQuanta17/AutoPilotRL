import subprocess

# subprocess.run(["python", "remove.py"])

subprocess.run(["git", "add", "."])
commit_message = input("Enter commit message: ")
subprocess.run(["git", "commit", "-m", commit_message])
subprocess.run(["git", "push"])