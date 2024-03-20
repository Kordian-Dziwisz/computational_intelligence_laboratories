import math

from sklearn.model_selection import train_test_split
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
    pd.read_csv("assets/iris.csv")
    1