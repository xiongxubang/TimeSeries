import pandas as pd
import os

def sh_generator(params_path):
    filename = "run_all.sh"
    with open(filename, "w") as f:
        f.write("python ./instanceGenerator.py\n")
        table = pd.read_csv(params_path, index_col=0)   # read the hyperParam.csv file using pandas
        for row_num in table.index:
            f.write(f"cd ./{row_num}\n")

            f.write("./run.sh\n")
            
            f.write("cd ..\n")
        
        f.write("python ./resultGenerator.py\n")
    os.system(f"chmod +x {filename}")
    

if __name__ == "__main__":
    # change the work dir path
    working_dir_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(working_dir_path)      # change to the dir containing instanceGenerator.py
    
    params_path = "./param.csv"    # input table
    sh_generator(params_path)


