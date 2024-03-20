import math

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd



def forward_pass(wiek: float, waga: float, wzrost:float) -> float:
    """simple forward pass function with predefined multipliers"""
    def f_act(n):
        return 1/(1+math.e^-n)
    w_11_21 = -0.46122
    w_11_22 = 0.78548
    w_12_21 = 0.97314
    w_12_22 = 2.10584
    w_13_21 = 0.39203
    w_13_22 = 0.57847
    b_21 = 0.80109
    b_22 = 0.43529
    b_o = -0.2368
    w_21_o = -0.81546
    w_22_o = 1.03775
    hidden1 =  wiek * w_11_21 + waga * w_12_21 + wzrost * w_13_21 + b_21
    hidden1_po_aktywacji = f_act(hidden1)
    hidden2 = wiek * w_11_22 + waga * w_12_22 + wzrost * w_13_22 + b_22
    hidden2_po_aktywacji = f_act(hidden2)
    output = hidden1_po_aktywacji * w_21_o + hidden2_po_aktywacji * w_22_o + b_o
    return output

def main() -> None:
    """main function"""
    # read data csv 
    data = pd.read_csv("assets/iris.csv")
    
    #normalize data columns petal.length, petal.width, sepal.length and sepal.width
    def normalize(x:pd.DataFrame) -> float:
        """normalize all values to between 0 and 1"""
        return (x - x.min()) / (x.max() - x.min())
    
    # labels = data["va"]
    labels = data["variety"]
    data = data[['petal.length', 'petal.width', 'sepal.length', 'sepal.width']].apply(normalize)
    # print(data.sample(5))


    datasets = train_test_split(data,
                            labels,
                            test_size=0.7)
    train_data, test_data, train_labels, test_labels = datasets

    #configuring our classifier
    clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(6,3),
                    random_state=1)

    clf.fit(train_data, train_labels)

    # testing the results of classification 
    predictions_train = clf.predict(train_data)
    predictions_test = clf.predict(test_data)
    train_score = accuracy_score(predictions_train, train_labels)
    print("score on train data: ", train_score)
    test_score = accuracy_score(predictions_test, test_labels)
    print("score on test data: ", test_score)



main()