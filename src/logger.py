import sys
import os

class Logger:
    def __init__(self, log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(log_file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
