import os
import pandas as pd

def resultGenerator(csv_path="./param.csv", annotations_path="./annotations.txt"):
    # change the work dir path
    working_dir_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(working_dir_path)

    table = pd.read_csv(csv_path, index_col=0)              # read the whole csv table
    with open(annotations_path, "r") as f:                  # read the whole annotations.txt
        lines = f.read().splitlines()
        for line in lines:                                  # for the given output file
            elem = line.split("\t")                         # the elements of one line
            path = elem[0]
            # create the new column
            for i in range(1, len(elem)):                   # skip the first element (i.e., path)
                table[elem[i]] = None
            
            # open the given output file in each instance, and load the data into big table
            for row_num in table.index:
                os.chdir(f"./{row_num}")                    # enter the corresponding dir
                with open(path, "r") as f1:
                    outputs = f1.read().splitlines()
                    for i in range(1, len(elem)):           # the meaning i is equal to that of the previous one
                        value = float(outputs[i-1])
                        table.loc[row_num, elem[i]] = value
                os.chdir(working_dir_path)
    
    table.to_csv("result.csv")
    print(table)
            
        


if __name__ == "__main__":
    resultGenerator()