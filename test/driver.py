import subprocess
import sys
import numpy as np

# runs executable with argument "-i input_file"
# expecting input_file to be a file name
# any additional arguments may be listed as well
# returns --- stdout & stderr as a list split into lines
def run_code(executable, input_file, *args):

    exec_list = [executable, "-i", input_file]
    for arg in args:
        exec_list.append(arg)

    return subprocess.check_output(exec_list,
        stderr=subprocess.STDOUT).decode(sys.stdout.encoding).strip().splitlines()

# given a list of string data (typically from run_code), look for a line
# beginning with "value" and return a floating point number after this "value"
# e.g. "keff = 1.0" return 1.0
# skips the "=" as well
def get_value(data, value):
    for line in data:
        this_line = line.split()
        if (len(this_line) > 0):
            if (this_line[0] == value):
                return float(this_line[2])
