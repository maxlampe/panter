import os

output_path = os.path.dirname(os.path.realpath(__file__)) + "/output"
if not os.path.exists(output_path):
    try:
        os.mkdir(output_path)
    except OSError:
        print("Creation of the directory %s failed" % output_path)
    else:
        print("Successfully created the directory %s " % output_path)
