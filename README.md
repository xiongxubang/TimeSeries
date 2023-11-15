# TimeSeries
COMP-5331@HKUST project

## Introduction

1. Directory "src/DeepLearning", which contains the source codes of the deep learning models like RNN, LSTM, GRU and Transformer.
2. File "requirement.txt", which is the required packages.


## Get Started
1. "TimeSeries" is the working directory.
2. Install the required packages.
```
pip install -r requirements.txt
```
3. Unzip the dataset.

```
unzip ./src/DeepLearning/dataset.zip -d ./src/DeepLearning/
```

4. Modify the hyperparameter in "param.csv".
5. Train the models with different hyperparameters. We provide the experiment scripts that quickly create the instances based on the hyperparameter. The experiment results can be reproduced by:

```
python shGenerator.py
./run_all.sh
```

6. All the outputs of the instances will be summarized in the table "result.csv".
