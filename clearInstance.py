import pandas as pd
import os

def clear(params_path):
    table = pd.read_csv(params_path, index_col=0)   # read the hyperParam.csv file using pandas
    for row_num in table.index:
        os.system(f"rm -rf ./{row_num}")
    

if __name__ == "__main__":
    # change the work dir path
    working_dir_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(working_dir_path)      # change to the dir containing instanceGenerator.py
    params_path = "./param.csv"    # input table
    clear(params_path)


