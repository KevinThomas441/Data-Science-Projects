import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X =  dataset.iloc[:, 1:2].values
Y =  dataset.iloc[:, 2].values

from sklearn.svm import SVR
regressor = SVR(kenel='rbf')
regressor.fit(X, Y)