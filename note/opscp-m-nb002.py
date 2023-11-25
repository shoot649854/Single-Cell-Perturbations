# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import gc
import glob
import random
import numpy as np 
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from itertools import groupby
import matplotlib.pyplot as plt

import joblib
import pickle

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

folder_path = "/kaggle/input/open-problems-single-cell-perturbations"
# -

de_train = pd.read_parquet(f'{folder_path}/de_train.parquet')
de_train

id_map = pd.read_csv(f'{folder_path}/id_map.csv', index_col='id')
id_map

genes = de_train.columns[5:]
genes


# +
def add_columns(de_train, id_map):
    sm_lincs_id = de_train.set_index('sm_name')["sm_lincs_id"].to_dict()
    sm_name_to_smiles = de_train.set_index('sm_name')['SMILES'].to_dict()

    id_map['sm_lincs_id'] = id_map['sm_name'].map(sm_lincs_id)
    id_map['SMILES'] = id_map['sm_name'].map(sm_name_to_smiles)
    
    return id_map

add_columns(de_train, id_map)


# -

def calculate_mae_and_mrrmse(y_true, y_pred, scaler=None):
    if scaler:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)

        # Calculate Mean Rowwise Root Mean Squared Error (MRRMSE)
        rowwise_rmse = np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
        mrrmse_score = np.mean(rowwise_rmse)
    else:
        # Calculate Mean Rowwise Root Mean Squared Error (MRRMSE)
        rowwise_rmse = np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
        mrrmse_score = np.mean(rowwise_rmse)
    
    # Print the results
    print(f"MRRMSE Score: {mrrmse_score}")


# +
kf = KFold(n_splits=6, shuffle=True, random_state=6174)

# Create k-fold columns
for fold, (train_index, test_index) in enumerate(kf.split(de_train)):
    de_train.loc[test_index, 'kfold'] = fold

de_train['kfold'] = de_train['kfold'].astype(int)

# Reindex DataFrame to put 'kfold' as the first column
cols = ['kfold'] + [col for col in de_train if col != 'kfold']
de_train = de_train[cols]

de_train
# -

test_data = de_train[de_train["kfold"]==5].sample(frac=1, random_state=6174)
train_data = de_train[de_train["kfold"]!=5].sample(frac=1, random_state=6174)
train_data

# +
# for i in range(1, 6):
#     train_index = de_train[de_train["kfold"]!=i].index
#     val_index = de_train[de_train["kfold"]==i].index

#     train_df = de_train.loc[train_index].copy()
#     val_df = de_train.loc[val_index].copy()

#     # genes
#     pred_df = val_df.copy()
#     pred_df[genes] = 0
#     for sm_name in val_df["sm_name"].unique():
#         pred_df.loc[val_df["sm_name"]==sm_name, genes] = train_df[train_df["sm_name"]==sm_name][genes].mean().to_list()

#     # Fillna by mean
#     pred_df.fillna(pred_df[genes].mean(), inplace=True)

#     # Get validation score
#     calculate_mae_and_mrrmse(val_df[genes].values, pred_df[genes].values)

# +
features = ['cell_type', 'sm_name']

n_components = 100
alpha = 0.5

models = []
svds = []

for i in range(5):
    train_index = train_data[train_data["kfold"]!=i].index
    val_index = train_data[train_data["kfold"]==i].index

    train_df = train_data.loc[train_index].sample(frac=1, random_state=6174)
    val_df = train_data.loc[val_index].sample(frac=1, random_state=6174)

    # genes
    pred_df = val_df.copy()
    pred_df[genes] = 0

    # Model
    model = make_pipeline(
        ColumnTransformer([('ohe', OneHotEncoder(handle_unknown='ignore'), ['sm_name'])]),
        AdaBoostRegressor()
    )
    svd = TruncatedSVD(n_components=n_components, random_state=6174)
    z_tr = svd.fit_transform(train_df[genes])
    model.fit(train_df[features], z_tr)
    models.append(model)
    svds.append(svd)
    
    y_pred = svd.inverse_transform(model.predict(val_df[features]))
    pred_df[genes] = y_pred

    # Get validation score
    calculate_mae_and_mrrmse(val_df[genes].values, pred_df[genes].values)

# +
y_preds = []
for i in range(5):
    y_preds.append(svds[i].inverse_transform(models[i].predict(val_df[features])))

y_preds = np.array(y_preds)
calculate_mae_and_mrrmse(val_df[genes].values, y_preds.mean(axis=0))
# -



# # Submission

# +
# Training with whole data.

features = ['cell_type', 'sm_name']

models = []
svds = []

for i in range(5):
    train_index = de_train[de_train["kfold"]!=i].index
    val_index = de_train[de_train["kfold"]==i].index

    train_df = de_train.loc[train_index].sample(frac=1, random_state=6174)
    val_df = de_train.loc[val_index].sample(frac=1, random_state=6174)

    # genes
    pred_df = val_df.copy()
    pred_df[genes] = 0

    # Model
    model = make_pipeline(
        ColumnTransformer([('ohe', OneHotEncoder(handle_unknown='ignore'), ['sm_name'])]),
        AdaBoostRegressor()
    )
    svd = TruncatedSVD(n_components=n_components, random_state=6174)
    z_tr = svd.fit_transform(train_df[genes])
    model.fit(train_df[features], z_tr)
    models.append(model)
    svds.append(svd)
    
    y_pred = svd.inverse_transform(model.predict(val_df[features]))
    pred_df[genes] = y_pred

    # Get validation score
    calculate_mae_and_mrrmse(val_df[genes].values, pred_df[genes].values)

# +
y_preds = []
for i in range(5):
    y_preds.append(svds[i].inverse_transform(models[i].predict(id_map[features])))

y_preds = np.array(y_preds)
id_map.loc[:, genes] = y_preds.mean(axis=0)
id_map = id_map.loc[:, genes]
id_map.to_csv('submission.csv')
id_map
# -


