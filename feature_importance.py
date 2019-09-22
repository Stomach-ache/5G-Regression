# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import glob, os
from numpy import genfromtxt
import time

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures


inst, label = [], []
os.chdir("/")

with open("./data_new.csv") as file:
    my_data = pd.read_csv(file).to_numpy()

train_X = np.array(my_data)[:5000]
train_y = np.array(my_data)[:5000]


# load JS visualization code to notebook
shap.initjs()

model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(train_X, label=train_y), 100)

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_X)

# features
a = np.array(['Cell Index', 'Cell X', 'Cell Y','Height','Azimuth','Electrical Downtilt','Mechanical Downtilt','Frequency Band','RS Power','Cell Altitude','Cell Building Height','Cell Clutter Index','X','Y','Altitude','Building Height','Clutter Index'])
shap.force_plot(explainer.expected_value, shap_values[0,:], a)


# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values, train_X)

# summarize the effects of all the features
a = a.reshape(1, 17)
print (a.shape, train_X.shape)
X = np.concatenate((a, train_X), axis=0)

shap.summary_plot(shap_values, X)

shap.summary_plot(shap_values, X, plot_type="bar")

