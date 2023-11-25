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

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
import warnings
warnings.simplefilter('ignore')

import pandas as pd

pd.set_option('display.max_columns', 30)

# +
import numpy as np

SEED = 42
np.random.seed(SEED)
# -

de_train = pd.read_parquet('/kaggle/input/open-problems-single-cell-perturbations/de_train.parquet')
de_train

id_map = pd.read_csv ('/kaggle/input/open-problems-single-cell-perturbations/id_map.csv')
id_map

sub_0_566 = pd.read_csv('/kaggle/input/op-scp-submissions-3/submission_0_566.csv')
sub_0_566 = pd.concat([id_map, sub_0_566], axis=1).drop(columns='id')
sub_0_566

de_train = pd.concat([de_train, sub_0_566, sub_0_566, sub_0_566, sub_0_566], ignore_index=True)
de_train

# +
from tqdm import tqdm
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import category_encoders as ce
from sklearn.metrics import r2_score

START_GENE_ID = 5
START_GENE_ID_TEST = 2
N_DESCRIBE = 1

Y = de_train.iloc[:, START_GENE_ID:].values
Y_test = sub_0_566.iloc[:, START_GENE_ID_TEST:].values

Y_submit = id_map.drop(columns=['cell_type', 'sm_name'])
metrics = pd.DataFrame(columns=['gene', 'mrrmse', 'r2'])

for i, gene in (enumerate(tqdm(de_train.columns[START_GENE_ID:]))):
    
    model_lsvr = LinearSVR(max_iter=2000, epsilon=0.1)
    model_knr = KNeighborsRegressor(n_neighbors=13)
    
    cell_type_target_encoder = ce.TargetEncoder()
    sm_name_target_encoder = ce.TargetEncoder()
    
    Y_gene = Y[:, i]
    
    X_train = pd.concat([cell_type_target_encoder.fit_transform(de_train[['cell_type']], Y_gene),
                         sm_name_target_encoder.fit_transform(de_train[['sm_name']], Y_gene)], axis=1)
    
    model_lsvr.fit(X_train, Y_gene)
    model_knr.fit(X_train, Y_gene)
    
    
    X_valid = pd.concat([cell_type_target_encoder.transform(id_map[['cell_type']]),
                         sm_name_target_encoder.transform(id_map[['sm_name']])], axis=1)
    
    Y_pred_lsvr = model_lsvr.predict(X_valid)
    Y_pred_knr = model_knr.predict(X_valid)
    Y_pred = Y_pred_lsvr * 0.7 + Y_pred_knr * 0.3
    
    Y_submit[gene] = Y_pred
    
    Y_gene_test = Y_test[:, i]
    
    X_test = pd.concat([cell_type_target_encoder.fit_transform(sub_0_566[['cell_type']], Y_gene_test),
                        sm_name_target_encoder.fit_transform(sub_0_566[['sm_name']], Y_gene_test)], axis=1)
    
    model_lsvr.fit(X_test, Y_gene_test)
    model_knr.fit(X_test, Y_gene_test)
    
    Y_pred_lsvr_test = model_lsvr.predict(X_valid)
    Y_pred_knr_test = model_knr.predict(X_valid)
    Y_pred_test = Y_pred_lsvr_test * 0.7 + Y_pred_knr_test * 0.3
    
    mrrmse = np.sqrt(np.square(Y_gene_test - Y_pred_test).mean()).mean()
    r2 = r2_score(Y_gene_test, Y_pred_test)
    
    metrics.loc[i, 'gene'] = gene
    metrics.loc[i, 'mrrmse'] = mrrmse
    metrics.loc[i, 'r2'] = r2
    
    if i < N_DESCRIBE:
        print(f'I: {i}, GENE: {gene}')
        print(f'Y_GENE.SHAPE:{Y_gene.shape}')
        print(f'Y_GENE:\n{Y_gene}')
        print(30 * '-')

        print(f'X_TRAIN.SHAPE: {X_train.shape}')
        print(f'X_TRAIN:\n{X_train}')
        print(30 * '-')

        print(f'X_VALID.SHAPE: {X_valid.shape}')
        print(f'X_VALID:\n{X_valid}')
        print(30 * '-')

        print(f'Y_PRED.SHAPE: {Y_pred.shape}')
        print(f'Y_PRED:\n{Y_pred}')
        print(30 * '-')
        
        print(f'Y_SUBMIT:\n{Y_submit}')
        print(30 * '-')
        
        print(f'Y_GENE_TEST.SHAPE:{Y_gene_test.shape}')
        print(f'Y_GENE_TEST:\n{Y_gene_test}')
        print(30 * '-')

        print(f'X_TEST.SHAPE: {X_test.shape}')
        print(f'X_TEST:\n{X_test}')
        print(30 * '-')

        print(f'Y_PRED_TEST.SHAPE: {Y_pred_test.shape}')
        print(f'Y_PRED_TEST:\n{Y_pred_test}')
        print(30 * '-')
        
        print(f'Y_GENE_TEST.SHAPE:{Y_gene_test.shape}')
        print(f'Y_GENE_TEST:\n{Y_gene_test}')
        print(30 * '-')

        print(f'X_TEST.SHAPE: {X_test.shape}')
        print(f'X_TEST:\n{X_test}')
        print(30 * '-')

        print(f'Y_PRED_TEST.SHAPE: {Y_pred_test.shape}')
        print(f'Y_PRED_TEST:\n{Y_pred_test}')
        print(30 * '-')
        
        print(f'Y_GENE_TEST - Y_PRED_TEST:\n{Y_gene_test - Y_pred_test}')
        print(30 * '-')
        
        print(f'(Y_GENE_TEST - Y_PRED_TEST).SHAPE:\n{(Y_gene_test - Y_pred_test).shape}')
        print(30 * '-')
    
        print(f'METRICS:\n{metrics}')
        print(30 * '=')
# -

metrics

metrics.to_csv('metrics_V_6.csv')

print(Y_submit.shape)
Y_submit

submit = pd.DataFrame(Y_submit, columns=de_train.columns[5:])
submit.index.name = 'id'
submit

submit.to_csv('submission.csv')

pd.read_csv('/kaggle/working/submission.csv')




