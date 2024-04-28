#!/usr/bin/env python

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from lassonet import LassoNetRegressor
from lassonet.inference_adaptive import AdaptiveLassoNetRegressor

boston = fetch_openml(name='boston', version=1)
X = boston['data']
y = boston['target']
_, true_features = X.shape
feature_names = X.columns
# add dummy feature
np.random.seed(1)
X = np.concatenate([X, np.random.randn(*X.shape)], axis=1)
feature_names = list(feature_names) + ["fake"] * true_features

# standardize
X = StandardScaler().fit_transform(X)
y = scale(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = LassoNetRegressor(
    hidden_dims=(10,),
    verbose=True,
    patience=(100, 5)
)
path = model.path(X_train, y_train, return_state_dicts=True)

n_selected = []
mse = []
lambda_ = []
skip_weights = []
for save in path:
    model.load(save.state_dict)
    y_pred = model.predict(X_test)
    n_selected.append(save.selected.sum().cpu().numpy())
    mse.append(mean_squared_error(y_test, y_pred))
    lambda_.append(save.lambda_)
    skip_weights.append(save.state_dict['skip.weight'])


fig = plt.figure(figsize=(12, 12))

plt.subplot(311)
plt.grid(True)
plt.plot(n_selected, mse, ".-")
plt.xlabel("number of selected features")
plt.ylabel("MSE")

plt.subplot(312)
plt.grid(True)
plt.plot(lambda_, mse, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("MSE")

plt.subplot(313)
plt.grid(True)
plt.plot(lambda_, n_selected, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("number of selected features")

plt.savefig("boston.png")

plt.clf()

n_features = X.shape[1]
importances = model.feature_importances_.numpy()
order = np.argsort(importances)[::-1]
importances = importances[order]
ordered_feature_names = [feature_names[i] for i in order]
color = np.array(["g"] * true_features + ["r"] * (n_features - true_features))[order]


plt.subplot(211)
plt.bar(
    np.arange(n_features),
    importances,
    color=color,
)
plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
colors = {"real features": "g", "fake features": "r"}
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.ylabel("Feature importance")

_, order = np.unique(importances, return_inverse=True)

plt.subplot(212)
plt.bar(
    np.arange(n_features),
    order + 1,
    color=color,
)
plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
plt.legend(handles, labels)
plt.ylabel("Feature order")

plt.savefig("boston-bar.png")
###############################################################################
model = AdaptiveLassoNetRegressor(
    hidden_dims=(10,),
    verbose=True,
    patience=(100, 5)
)
path = model.path(X_train, y_train, return_state_dicts=True)

n_selected = []
mse = []
lambda_ = []
skip_weights = []
for save in path:
    model.load(save.state_dict)
    y_pred = model.predict(X_test)
    n_selected.append(save.selected.sum().cpu().numpy())
    mse.append(mean_squared_error(y_test, y_pred))
    lambda_.append(save.lambda_)
    skip_weights.append(save.state_dict['skip.weight'])


fig = plt.figure(figsize=(12, 12))

plt.subplot(311)
plt.grid(True)
plt.plot(n_selected, mse, ".-")
plt.xlabel("number of selected features")
plt.ylabel("MSE")

plt.subplot(312)
plt.grid(True)
plt.plot(lambda_, mse, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("MSE")

plt.subplot(313)
plt.grid(True)
plt.plot(lambda_, n_selected, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("number of selected features")

plt.savefig("boston_scad.png")

plt.clf()

n_features = X.shape[1]
min_mse = np.argmin(mse)
for i, save in enumerate(path):
    if i == min_mse:
        model.load(save.state_dict)
        break
importances = model.feature_importances_.numpy()
order = np.argsort(importances)[::-1]
importances = importances[order]
ordered_feature_names = [feature_names[i] for i in order]
color = np.array(["g"] * true_features + ["r"] * (n_features - true_features))[order]


plt.subplot(211)
plt.bar(
    np.arange(n_features),
    importances,
    color=color,
)
plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
colors = {"real features": "g", "fake features": "r"}
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.ylabel("Feature importance")

_, order = np.unique(importances, return_inverse=True)

plt.subplot(212)
plt.bar(
    np.arange(n_features),
    order + 1,
    color=color,
)
plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
plt.legend(handles, labels)
plt.ylabel("Feature order")

plt.savefig("boston-bar-scad.png")

import pandas as pd
import torch
df = pd.DataFrame(torch.cat(skip_weights).numpy())
df.index = lambda_
plt.clf()
plt.subplot(211)

for i, column in enumerate(df.columns):
    if i < 13:  # First 13 columns with solid lines
        df[column].plot(linestyle='-', label=feature_names[i])
    else:  # Remaining columns with dashed lines
        df[column].plot(linestyle='--', label=feature_names[i])

plt.legend(loc='upper right')
plt.axvline(x=lambda_[min_mse], color='r', linestyle='-', linewidth=2, label='Min MSE')

plt.subplot(212)
df = df.iloc[:200]
for i, column in enumerate(df.columns):
    if i < 13:  # First 13 columns with solid lines
        df[column].plot(linestyle='-', label=feature_names[i])
    else:  # Remaining columns with dashed lines
        df[column].plot(linestyle='--', label=feature_names[i])

plt.legend(loc='upper right')
plt.axvline(x=lambda_[min_mse], color='r', linestyle='-', linewidth=2, label='Min MSE')
plt.savefig("boston-scad-path.png")
