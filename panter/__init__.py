import os

output_path = os.path.dirname(os.path.realpath(__file__)) + "/output"
assert os.path.exists(output_path), "ERROR: Directory panter/output is missing."