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

# # What is about ?  
#
# The first brief look on the "Open Problems – Single-Cell Perturbations" data.
#
# And some baselines
#
#
# ###  Notes 
#
# - number of train samples - 614, test 255 - not much ! ML for such low number of samples - hmmm.... very unclearly how successful it can be ...   
#
# - 18 211 targets (genes)  
#
# - in some sense we have ONLY TWO features - cell type and compound. Both of them are categorical. Cell types - 6, compounds - 146 categories. Thus it is natural to use target encodings. Alternative way - one might think in direction of recsys approach, but with the modification that we have 18211 targets. 
#
#
#     V40 EDA : Genes top affected by Clotrimazole, Vorinostat :
#     Clotrimazole
#     Up: ['RPL10', 'RPL13', 'MAPRE3', 'TPT1', 'TRAV6', 'RPL3P4', 'AC107068.1', 'TMEM191C', 'SYNGR1', 'MTATP6P1', 'RPL7A', 'CHD3', 'GAL3ST4', 'SRPK1', 'HIST2H2BE', 'CCM2', 'RPLP1', 'PRKY', 'TOB2', 'SLC25A42']
#     Down: ['MX1', 'MX2', 'MRPS18B', 'RAB18', 'USP18', 'ISG20', 'ISG15', 'C19ORF66', 'SHISA5', 'IFI16', 'IFI44', 'AC090114.2', 'C6ORF62', 'TDRD7', 'LAP3', 'IFI44L', 'TRIM22', 'XAF1', 'AC116407.2', 'HMGN2']
#
#     Vorinostat
#     Up: ['HIST1H1D', 'H1FX', 'METRN', 'HIST1H1C', 'RPS18', 'HIST1H1B', 'STMN1', 'RPLP0', 'RPL7A', 'FXYD7', 'H2AFJ', 'RPS16', 'RPL8', 'HIST1H1E', 'CTNNAL1', 'RPS7', 'MARCKSL1', 'FXYD1', 'PCSK1N', 'RPL13A']
#     Down: ['CORO1A', 'DDX5', 'SRSF5', 'LIMD2', 'CD3E', 'EMP3', 'GPSM3', 'S100A6', 'RGL4', 'LTB', 'AES', 'HCST', 'ADI1', 'TRBC2', 'SRSF7', 'BIN2', 'UFC1', 'GZMM', 'CD37', 'RBM3']
#
#
#     V39 EDA: Genes top affected by 'Topotecan': 
#         UP: ['RPS27L', 'HNRNPC', 'VAMP2', 'TMEM14B', 'TTC39C', 'AP3M2', 'RSRP1', 'CXCR4', 'BAX', 'HINT1', 'FTH1', 'HMGB1', 'ITM2A', 'NDUFS5', 'RPL12', 'CREM', 'COX20', 'NSD1', 'RPL28', 'RPLP1']
#         Down: ['ARHGAP15', 'ZBTB20', 'STX8', 'IMMP2L', 'SYNE2', 'PTPRC', 'SMAP1', 'SND1', 'MRPS6', 'DOCK2', 'PRKCB', 'SKAP1', 'ACTB', 'ITPR2', 'TBC1D5', 'LPP', 'COMMD1', 'FAF1', 'SSU72', 'NDUFAF2']
#         
#         
#     V37 0.707 approach 5 - aggr by compound + Ridge
#     
#     V34 LB 0.688 iterations=3,  depth = 6  
#     V33 LB 0.709 Catboost iterations 10,  depth = 6    - try to reduce overfit - but even worse  
#     V32 LB 0.912 - Catboost depth = 2, iteration 100 - try to reduce overfit - but even worse  
#     V30 LB 0.905 ( terrible :)  - Catboost first draft -  depth = 6 , iteration 100
#
#     Versions 23-27 - first modeling approach - target encode cell type and compound and simple Ridge model
#     LB 0.616:  https://www.kaggle.com/code/alexandervc/op2-models-cv-tuning?scriptVersionId=143324657 (V4)
#     It was neccesary to drasticaly increase the regalarization - alpha = 100_000 to get better results. 
#     Without that - result were worse than naive approach used before:
#         V27 LB 0.659 LB Ridge nCT1 nCD25 Al10 TSVD35 - even more relaxed : alpha and encoded_compound size 
#         V26 LB 0.668 try to relax a model a bit: Ridge nCT1 nCD10 Al100 TSVD35 - results are better near mean/zero submission
#         V25 LB 0.677 - stronger constraining the model: Ridge nCT1 nCD5 Al100 TSVD35, but still we are worse than even predict by mean 
#         V24 LB 0.702 - try avoid overfit Ridge nCT3 nCD10 Al100 TSVD35 - better results but still bad 
#         V23 LB 0.747 Ridge model in target encoded features Ridge nCT10 nCD35 Al1 TSVD35 - modeling gives worse results than simple approach, might be bug or overfit
#     
#     V22 EDA - cell cycle genes brief analysis 
#     V19,20,21 EDA - clustering 10000, 15000, 18211(all) genes by sns.clustermap - 15min,47min,RAMcrash - see two clear clusters in genes - what are they ? 
#     V18 LB 0.623 TSVD-35 denoising  quantile 0.54, tsvd is better than pca/ica - similar to previous OP
#     V15 - bug -  LB 0.626 - NO it was not: TSVD-35 denoising  quantile 0.54 - so worse than pca,ICA, at least for these params
#     V14 LB 0.624 - ICA-35 denoising   quantile 0.54 - so ICA is not better than just pca at least for these the same params
#     Fork: LB 0.624 - pca-35 denoising, quantile 0.54 
#     V13 LB 0.626 pca25 denoising 
#     V12 LB 0.627 same with pca100 denoising
#             That means: first reduce data to pca100, and  take pca inverse transfrom (denoising)
#             and then apply same as in V9: groupby by compound and quantile(0.6)         
#             so idea is - pca100 reductions hopefully kills some noise
#             In the previous challenge it worked okay, but the number of samples was not 614, but near 100 000
#     V11 - bug - pay attention that column names in submit file should correspond to genes
#     V9 LB 0.638 groupby by compound and quantile(0.6)
#     V6 LB 0.666 quantile(0.7)
#     V5 LB 0.657 quantile(0.6) 
#             results are again better, that probably indicates some shift between train and public data 
#     V4 LB 0.659 - median instead of mean - results are a bit better, 
#             it might mean either a bit of presense of outliers, or  public is somewhat different from train - next experiments suggests second is true 
#     V1 LB 0.664 - submission of train means - the simplest baseline 

# # Preliminaries

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import time
t0start = time.time() 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -

# # Train data

# %%time
fn = '/kaggle/input/open-problems-single-cell-perturbations/de_train.parquet'
df_de_train = pd.read_parquet(fn)# , index_col = 0)
print(df_de_train.shape)
df_de_train

df_de_train.iloc[:,5:].head(1)

# # Genes top affected by some compounds (drugs)

print( list(df_de_train['sm_name'].unique()) )

# +
plt.figure(figsize = (20,4) )
v = df_de_train.iloc[:,5:].max(axis = 0 ).sort_values(ascending = False, key = abs )
plt.plot(v.values,'*-')
plt.title('Max-abs DE for genes',fontsize = 20 )
plt.grid()
plt.show()
# display(v.head(15))
t = v
n =0;d=15;
for n in [0,d,2*d,3*d,4*d,5*d,6*d,7*d,8*d,9*d]:
    t2 = t.iloc[n:n+d].to_frame().reset_index()
    t2.columns = ['Gene','DE']
    if n == 0:
        t3 = t2.copy()
    else:
        t3 = pd.concat( (t3,t2),axis = 1 )
        
display(t3.round(1))


# +
# %%time
print(' "Most often" (compound/cell-types averaged) affected genes ')
t =  (-(df_de_train.iloc[:,5:].T.abs())).rank().median(axis=1) # Use ranks and average (over pairs - compound/cell-types) them . Minus - to get top gene as rank=1 not as rank=len   
t = t.sort_values()
print(' "Most often" (compound/cell-types averaged) affected genes ')
n =0;d=15;
for n in [0,d,2*d,3*d,4*d,5*d,6*d,7*d,8*d,9*d]:
    t2 = t.iloc[n:n+d].to_frame().reset_index()
    t2.columns = ['Gene','Rank']
    if n == 0:
        t3 = t2.copy()
    else:
        t3 = pd.concat( (t3,t2),axis = 1 )
display(t3)

plt.figure(figsize = (20,4) )
plt.plot(t.values,'*')
plt.ylabel('Rank (averaged) ',fontsize = 15)
plt.grid()
plt.title( '  "Most often" (compound/cell-types averaged) affected genes ', fontsize = 20)
plt.show()

# +
# %%time
print(' Top affected genes at least once ')
t =  (-(df_de_train.iloc[:,5:].T.abs())).rank().min(axis=1) # Use ranks and average (over pairs - compound/cell-types) them . Minus - to get top gene as rank=1 not as rank=len   
t = t.sort_values()
n =0;d=15;
for n in [0,d,2*d,3*d,4*d,5*d,6*d,7*d,8*d,9*d]:
    t2 = t.iloc[n:n+d].to_frame().reset_index()
    t2.columns = ['Gene','Rank']
    if n == 0:
        t3 = t2.copy()
    else:
        t3 = pd.concat( (t3,t2),axis = 1 )
        
print('Count top1 at least once:',  (t==1).sum() )
display(t3)

plt.figure(figsize = (20,4) )
plt.plot(t.values,'*')
plt.ylabel('Rank (averaged) ',fontsize = 15)
plt.grid()
plt.title( '  Top affected genes at least once ', fontsize = 20)
plt.show()

# +
t =  (-(df_de_train.iloc[:,5:].T.abs())).rank()
m = (t == 1)
v = (m.sum(axis = 1))
v = v.sort_values(ascending = False)
for t in list(range(1,14))[::-1]:
    print(t, (v == t ).sum() )
print(' Genes Most often top1 ')
t = v
n =0;d=15;
for n in [0,d,2*d,3*d,4*d,5*d,6*d,7*d,8*d,9*d]:
    t2 = t.iloc[n:n+d].to_frame().reset_index()
    t2.columns = ['Gene','Rank']
    if n == 0:
        t3 = t2.copy()
    else:
        t3 = pd.concat( (t3,t2),axis = 1 )
display(t3)


t =  (-(df_de_train.iloc[:,5:].T.abs())).rank()
m = (t == 1)| (t == 2)
v = (m.sum(axis = 1))
v = v.sort_values(ascending = False)
for t in list(range(1,14))[::-1]:
    print(t, (v == t ).sum() )
print(' Genes Most often top1 or top2 ')
t = v
n =0;d=15;
for n in [0,d,2*d,3*d,4*d,5*d,6*d,7*d,8*d,9*d]:
    t2 = t.iloc[n:n+d].to_frame().reset_index()
    t2.columns = ['Gene','Rank']
    if n == 0:
        t3 = t2.copy()
    else:
        t3 = pd.concat( (t3,t2),axis = 1 )
display(t3)

t =  (-(df_de_train.iloc[:,5:].T.abs())).rank()
m = (t == 1)| (t == 2) | (t == 3)| (t == 4)  | (t == 5)| (t == 6) 
v = (m.sum(axis = 1))
v = v.sort_values(ascending = False)
for t in list(range(1,14))[::-1]:
    print(t, (v == t ).sum() )
print(' Genes Most often top1 - top5 ')
t = v
n =0;d=15;
for n in [0,d,2*d,3*d,4*d,5*d,6*d,7*d,8*d,9*d]:
    t2 = t.iloc[n:n+d].to_frame().reset_index()
    t2.columns = ['Gene','Rank']
    if n == 0:
        t3 = t2.copy()
    else:
        t3 = pd.concat( (t3,t2),axis = 1 )
display(t3)


# +
drug = 'Topotecan'
v_drug = (df_de_train['sm_name'] == drug).astype(float)
v_drug.sum()
df = df_de_train.copy()
df[drug] = v_drug
#for gene in df_de_train.columns[5:]:
d_agg = df.iloc[:,5:].groupby(drug).mean().T
d_agg = d_agg.sort_values(1)
display(d_agg.sort_values(1,ascending = False).head(10))
print(list(d_agg.sort_values(1,ascending = False).head(20).index))
display(d_agg.sort_values(1,ascending = True).head(10))
print(list(d_agg.sort_values(1,ascending = True).head(20).index))
# display(d_agg.head(5) )
# display(d_agg.tail(5) )
m = d_agg[1] > 4
display(d_agg[m])

plt.figure(figsize=(20,5))
plt.plot(d_agg[0].values)
plt.plot(d_agg[1].values)
plt.grid()
plt.title(drug,fontsize=20)
plt.xlabel('Genes sorted',fontsize = 20 )
plt.show()

plt.figure(figsize=(20,5))
d_agg[3] = d_agg[1].values - d_agg[0].values 
plt.plot(d_agg[3].values)
plt.plot()
plt.grid()
plt.title(drug,fontsize=20)
plt.xlabel('Genes sorted',fontsize = 20 )
plt.show()
display( d_agg.sort_values(3).head(20) )
display( d_agg.sort_values(3).tail(20) )


# +
drug = 'Vorinostat' # 'Topotecan' 'Belinostat', 'Clotrimazole',
v_drug = (df_de_train['sm_name'] == drug).astype(float)
v_drug.sum()
df = df_de_train.copy()
df[drug] = v_drug
#for gene in df_de_train.columns[5:]:
d_agg = df.iloc[:,5:].groupby(drug).mean().T
d_agg = d_agg.sort_values(1)
display(d_agg.sort_values(1,ascending = False).head(10))
print(list(d_agg.sort_values(1,ascending = False).head(20).index))
display(d_agg.sort_values(1,ascending = True).head(10))
print(list(d_agg.sort_values(1,ascending = True).head(20).index))
# display(d_agg.head(5) )
# display(d_agg.tail(5) )
m = d_agg[1] > 4
display(d_agg[m])

plt.figure(figsize=(20,5))
plt.plot(d_agg[0].values)
plt.plot(d_agg[1].values)
plt.grid()
plt.title(drug,fontsize=20)
plt.xlabel('Genes sorted',fontsize = 20 )
plt.show()

plt.figure(figsize=(20,5))
d_agg[3] = d_agg[1].values - d_agg[0].values 
plt.plot(d_agg[3].values)
plt.plot()
plt.grid()
plt.title(drug,fontsize=20)
plt.xlabel('Genes sorted',fontsize = 20 )
plt.show()
display( d_agg.sort_values(3).head(20) )
display( d_agg.sort_values(3).tail(20) )


# +
drug = 'Clotrimazole'#  'Vorinostat' # 'Topotecan' 'Belinostat', ,
v_drug = (df_de_train['sm_name'] == drug).astype(float)
v_drug.sum()
df = df_de_train.copy()
df[drug] = v_drug
#for gene in df_de_train.columns[5:]:
d_agg = df.iloc[:,5:].groupby(drug).mean().T
d_agg = d_agg.sort_values(1)
display(d_agg.sort_values(1,ascending = False).head(10))
print(list(d_agg.sort_values(1,ascending = False).head(20).index))
display(d_agg.sort_values(1,ascending = True).head(10))
print(list(d_agg.sort_values(1,ascending = True).head(20).index))
# display(d_agg.head(5) )
# display(d_agg.tail(5) )
m = d_agg[1] > 4
display(d_agg[m])

plt.figure(figsize=(20,5))
plt.plot(d_agg[0].values)
plt.plot(d_agg[1].values)
plt.grid()
plt.title(drug,fontsize=20)
plt.xlabel('Genes sorted',fontsize = 20 )
plt.show()

plt.figure(figsize=(20,5))
d_agg[3] = d_agg[1].values - d_agg[0].values 
plt.plot(d_agg[3].values)
plt.plot()
plt.grid()
plt.title(drug,fontsize=20)
plt.xlabel('Genes sorted',fontsize = 20 )
plt.show()
display( d_agg.sort_values(3).head(20) )
display( d_agg.sort_values(3).tail(20) )

# -

# # Dimensional reductions (pca, umap,...), visualizations, clustering train target data 

X = df_de_train.iloc[:,5:]
print(X.shape)

# +
# %%time
from sklearn.decomposition import PCA

v1_color = df_de_train[  'cell_type']
v2_color = df_de_train[  'sm_name'].copy()
v3_color = df_de_train[  'sm_name'].copy()
l = [t for t in df_de_train[  'sm_name'] if t.endswith('nib') ]
m = v2_color.isin( l)
v2_color[~m] = 'non -nib'
v3_color[m] = '*nib'

v4_color = df_de_train[  'control']#.copy()

list_top_drugs = ['MLN 2238', 'Resminostat', 'CEP-18770 (Delanzomib)', 'Oprozomib (ONX 0912)', 'Belinostat', 'Vorinostat', 'Ganetespib (STA-9090)', 'Scriptaid', 'Proscillaridin A;Proscillaridin-A', 'Alvocidib', 'IN1451']
m = df_de_train[  'sm_name'].isin( list_top_drugs)
v5_color = df_de_train[  'sm_name'].copy()
v5_color[~m] = np.nan



list_cfg = [ ['cell type',v1_color], ['control' , v4_color ] , ['top compounds',v5_color] ]
#     str_inf1 = ''
    #X = np.clip(df.iloc[N0:N1,33:137].fillna(0),0, 1)
str_inf = 'PCA' 
reducer = PCA(n_components=100 )
Xr = reducer.fit_transform(X)
for i,j in [[0,1],[0,2],[1,2],[3,4],[5,6],[7,8]]:
    plt.figure(figsize = (20,10)); ic=0
    for str_inf1, v_for_color in list_cfg: # , ['*nib compounds ',v2_color], ['non *nib compounds',v3_color ] ]:
        ic+=1; plt.subplot(1,len(list_cfg),ic)
        sns.scatterplot(x= Xr[:,i], y = Xr[:,j], hue =  v_for_color ,s = 100) # df['reads'])
        plt.xlabel(str_inf+str(i+1), fontsize = 20)
        plt.ylabel(str_inf+str(j+1), fontsize = 20)
        plt.title(str_inf1 + ' ', fontsize = 20 )

    plt.show()
    


# -

d = df_de_train.iloc[:,:5]
d['PCA1'] = Xr[:,0]
d['PCA2'] = Xr[:,1]
d['PCA3'] = Xr[:,2]
list_top_drugs = []
display( d.sort_values('PCA1', ascending = False ).head(8) )
list_top_drugs += d.sort_values('PCA1', ascending = False ).head(8)['sm_name'].to_list()
print(list_top_drugs)
display( d.sort_values('PCA2', ascending = False ).head(8) )
list_top_drugs += d.sort_values('PCA2', ascending = False ).head(8)['sm_name'].to_list()
display( d.sort_values('PCA3', ascending = False ).head(8) )
list_top_drugs += d.sort_values('PCA3', ascending = False ).head(8)['sm_name'].to_list()
print(list(set(list_top_drugs)))

# ## Clustering  Cell Types

# %%time
N = df_de_train.shape[1]# 5000
print(N)
X = df_de_train[ ['cell_type'] + list(df_de_train.columns[5:N]) ].groupby('cell_type').median()
print(X.shape)
cm = np.corrcoef(X)
print(cm[:3,:2])
cm = np.abs(cm)
l = list(X.index)# [df_de_train['sm_name'].iat[i] +' '+ df_de_train['cell_type'].iat[i]  for i in range(len(df_de_train))] # .columns[5:N]
cm = pd.DataFrame(cm, index =l , columns = l )
print(cm.shape)
sns.clustermap(cm,  annot=True, fmt=".2f", cmap="coolwarm" )
plt.show()


# ## Clustering compounds

# +
# %%time
N = df_de_train.shape[1]# 5000
print(N)
X = df_de_train[ ['sm_name'] + list(df_de_train.columns[5:N]) ].groupby('sm_name').median()
print(X.shape)
cm = np.corrcoef(X)
print(cm[:3,:2])
cm = np.abs(cm)
l = list(X.index)# [df_de_train['sm_name'].iat[i] +' '+ df_de_train['cell_type'].iat[i]  for i in range(len(df_de_train))] # .columns[5:N]
l = [t[:20] for t in l] # cut long names
cm = pd.DataFrame(cm, index =l , columns = l )
print(cm.shape)
clustergrid = sns.clustermap(cm,cmap="coolwarm" )# ,  annot=True, fmt=".2f", 
plt.show()
reordered_columns = clustergrid.dendrogram_col.reordered_ind
reordered_rows = clustergrid.dendrogram_row.reordered_ind
print(len(reordered_rows), len(reordered_columns) )
print( list(cm.index[reordered_rows]) )
# print( list(X.columns[reordered_columns]) )

sns.clustermap(cm,  annot=True, fmt=".2f", cmap="coolwarm" )
plt.show()

# -

# ## Clustering samples (i.e. pairs cell + compound)

# %%time
N = df_de_train.shape[1]# 5000
X = df_de_train.iloc[:,5:N]
print(X.shape)
cm = np.corrcoef(X)
print(cm[:3,:2])
cm = np.abs(cm)
l = [df_de_train['sm_name'].iat[i] +' '+ df_de_train['cell_type'].iat[i]  for i in range(len(df_de_train))] # .columns[5:N]
cm = pd.DataFrame(cm, index =l , columns = l )
print(cm.shape)
clustergrid = sns.clustermap(cm)
plt.show()
reordered_columns = clustergrid.dendrogram_col.reordered_ind
reordered_rows = clustergrid.dendrogram_row.reordered_ind
print(len(reordered_rows), len(reordered_columns) )
print( list(cm.index[reordered_rows]) )
# print( list(cm.columns[reordered_columns]) )


# ## Look in genes space 

# +
# %%time
from sklearn.decomposition import PCA

X = df_de_train.iloc[:,5:].T
print(X.shape)

v1_color = pd.Series(range(len(X)), name = 'index') # df_de_train[  'cell_type']



list_cfg = [ ['Genes',v1_color]]# , ['control' , v4_color ] , ['top compounds',v5_color] ]
#     str_inf1 = ''
    #X = np.clip(df.iloc[N0:N1,33:137].fillna(0),0, 1)
str_inf = 'PCA' 
reducer = PCA(n_components=10 )
Xr = reducer.fit_transform(X)
for i,j in [[0,1],[0,2],[1,2],[3,4],[5,6],[7,8]]:
    plt.figure(figsize = (20,10)); ic=0
    for str_inf1, v_for_color in list_cfg: # , ['*nib compounds ',v2_color], ['non *nib compounds',v3_color ] ]:
        ic+=1; plt.subplot(1,len(list_cfg),ic)
        sns.scatterplot(x= Xr[:,i], y = Xr[:,j], hue =  v_for_color ,s = 100) # df['reads'])
        plt.xlabel(str_inf+str(i+1), fontsize = 20)
        plt.ylabel(str_inf+str(j+1), fontsize = 20)
        plt.title(str_inf1 + ' ', fontsize = 20 )

    plt.show()
    



# +
# %%time
from sklearn.decomposition import PCA
import umap 

X = df_de_train.iloc[:,5:].T
print(X.shape)

v1_color = pd.Series(range(len(X)), name = 'index') # df_de_train[  'cell_type']



list_cfg = [ ['Genes',v1_color]]# , ['control' , v4_color ] , ['top compounds',v5_color] ]
# str_inf = 'PCA' 
# reducer = PCA(n_components=10 )
str_inf = 'UMAP' 
reducer = umap.UMAP()# (n_components=10 )

Xr = reducer.fit_transform(X)
for i,j in [[0,1] ]:# ,[0,2],[1,2],[3,4],[5,6],[7,8]]:
    plt.figure(figsize = (20,10)); ic=0
    for str_inf1, v_for_color in list_cfg: # , ['*nib compounds ',v2_color], ['non *nib compounds',v3_color ] ]:
        ic+=1; plt.subplot(1,len(list_cfg),ic)
        sns.scatterplot(x= Xr[:,i], y = Xr[:,j], hue =  v_for_color ,s = 100) # df['reads'])
        plt.xlabel(str_inf+str(i+1), fontsize = 20)
        plt.ylabel(str_inf+str(j+1), fontsize = 20)
        plt.title(str_inf1 + ' ', fontsize = 20 )

    plt.show()
    


# -

# ## Genes clustering 

# %%time
N = 1000#  18211 #  15_000# 10000 #
X = df_de_train.iloc[:,5:N].T
print(X.shape)
cm = np.corrcoef(X)
print(cm[:3,:2])
cm = np.abs(cm)
cm = pd.DataFrame(cm, index = df_de_train.columns[5:N], columns = df_de_train.columns[5:N] )
print(cm.shape)
sns.clustermap(cm)
plt.show()


# # Genes related to proliferation (cell cycle genes )
#
# Proliferation cycle or cell cycle - key process for many cells. Many drugs here are anti-cancer - so should affect the ability of cells to proliferate. So might be interesting to look on drugs affect on these particular genes.
#
# Tirosh et.al. proposed list of about 100 cell-cycle genes which are the most effectively seen by single cell data. It is good starting point.
#
# In general there are much more genes related to the cell cycle, with various degree of "relation". Some list are cell cycle genes may contain thousands genes, but in fact in such huge lists most of the genes are related to cell cycle very weakly or these genes not good captured by single cell sequencing, while Tirosh list contains genes very strongly related to cell cycle and well captured by single cell technology. Moreover, many genes are associated to various biological processes in the cell, but Tirosh genes are mostly associated with cell cycle, not with the other processes - another argument why they are convenient to work with. 
#
# One may look at Computational challenges of cell cycle analysis using single cell transcriptomics
# Alexander Chervov, Andrei Zinovyev https://arxiv.org/abs/2208.05229
#
# And hundreds Kaggle notebooks/datasets related to that work e.g. that discussion: https://www.kaggle.com/competitions/open-problems-multimodal/discussion/350314 , or that notebook: https://www.kaggle.com/code/alexandervc/mmscel-cell-cycle-03b-daybydaychange-allcelltypes/notebook
#
# PS
#
# Other cell cycle genes sets e.g. by Tom Freeman
#
# See some comparison e.g. here: https://www.kaggle.com/code/alexandervc/tabmuris-cell-cycle-1-data-one-by-one#Cell-cycle
# So list is bigger but kind of more "dirty", that means containing more non-cell cycle effects, and G1S - G2M split is less prominent. 
#

# +
G1S_genes_Tirosh = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8']
G2M_genes_Tirosh = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']
genes_Tirosh = G1S_genes_Tirosh + G2M_genes_Tirosh

# Subset of Tirosh genes to capture "fast" cell cycle pattern = see https://arxiv.org/abs/2208.05229
list_genes_fastCCsign = ['CDK1', 'UBE2C', 'TOP2A', 'TMPO', 'HJURP', 'RRM1', 'RAD51AP1', 'RRM2', 'CDC45', 'BLM', 'BRIP1', 'E2F8', 'HIST2H2AC']

G1S_genes_Freeman = ['ADAMTS1', 'ASF1B', 'ATAD2', 'BARD1', 'BLM', 'BRCA1', 'BRIP1', 'C17orf75', 'C9orf40', 'CACYBP', 'CASP8AP2', 'CCDC15', 'CCNE1', 'CCNE2', 'CCP110', 'CDC25A', 'CDC45', 'CDC6', 'CDC7', 'CDK2', 'CDT1', 'CENPJ', 'CENPQ', 'CENPU', 'CEP57', 'CHAF1A', 'CHAF1B', 'CHEK1', 'CLSPN', 'CREBZF', 'CRYL1', 'CSE1L', 'DCLRE1B', 'DCTPP1', 'DEK', 'DERA', 'DHFR', 'DNA2', 'DNAJC9', 'DNMT1', 'DONSON', 'DSCC1', 'DSN1', 'DTL', 'E2F8', 'EED', 'EFCAB11', 'ENDOD1', 'ETAA1', 'EXO1', 'EYA2', 'EZH2', 'FAM111A', 'FANCE', 'FANCG', 'FANCI', 'FANCL', 'FBXO5', 'FEN1', 'GGH', 'GINS1', 'GINS2', 'GINS3', 'GLMN', 'GMNN', 'GMPS', 'GPD2', 'HADH', 'HELLS', 'HSF2', 'ITGB3BP', 'KIAA0101', 'KNTC1', 'LIG1', 'MCM10', 'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7', 'MCMBP', 'METTL9', 'MMD', 'MNS1', 'MPP1', 'MRE11A', 'MSH2', 'MSH6', 'MYO19', 'NASP', 'NPAT', 'NSMCE4A', 'ORC1', 'OSGEPL1', 'PAK1', 'PAQR4', 'PARP2', 'PASK', 'PAXIP1', 'PBX3', 'PCNA', 'PKMYT1', 'PMS1', 'POLA1', 'POLA2', 'POLD3', 'POLE2', 'PRIM1', 'PRPS2', 'PSMC3IP', 'RAB23', 'RAD51', 'RAD51AP1', 'RAD54L', 'RBBP8', 'RBL1', 'RDX', 'RFC2', 'RFC3', 'RFC4', 'RMI1', 'RNASEH2A', 'RPA1', 'RRM1', 'RRM2', 'SLBP', 'SLC25A40', 'SMC2', 'SMC3', 'SSX2IP', 'SUPT16H', 'TEX30', 'TFDP1', 'THAP10', 'THEM6', 'TIMELESS', 'TIPIN', 'TMEM106C', 'TMEM38B', 'TRIM45', 'TRIP13', 'TSPYL4', 'TTI1', 'TUBGCP5', 'TYMS', 'UBR7', 'UNG', 'USP1', 'WDHD1', 'WDR76', 'WRB', 'YEATS4', 'ZBTB14', 'ZWINT']
G2M_genes_Freeman = ['ADGRE5', 'ARHGAP11A', 'ARHGDIB', 'ARL6IP1', 'ASPM', 'AURKA', 'AURKB', 'BIRC5', 'BORA', 'BRD8', 'BUB1', 'BUB1B', 'BUB3', 'CCNA2', 'CCNB1', 'CCNB2', 'CCNF', 'CDC20', 'CDC25B', 'CDC25C', 'CDC27', 'CDCA3', 'CDCA8', 'CDK1', 'CDKN1B', 'CDKN3', 'CENPE', 'CENPF', 'CENPI', 'CENPN', 'CEP55', 'CEP70', 'CEP85', 'CKAP2', 'CKAP5', 'CKS1B', 'CKS2', 'CTCF', 'DBF4', 'DBF4B', 'DCAF7', 'DEPDC1', 'DLGAP5', 'ECT2', 'ERCC6L', 'ESPL1', 'FAM64A', 'FOXM1', 'FZD2', 'FZD7', 'FZR1', 'GPSM2', 'GTF2E1', 'GTSE1', 'H2AFX', 'HJURP', 'HMGB2', 'HMGB3', 'HMMR', 'HN1', 'INCENP', 'JADE2', 'KIF11', 'KIF14', 'KIF15', 'KIF18A', 'KIF18B', 'KIF20A', 'KIF20B', 'KIF22', 'KIF23', 'KIF2C', 'KIF4A', 'KIF5B', 'KIFC1', 'KPNA2', 'LBR', 'LMNB2', 'MAD2L1', 'MELK', 'MET', 'METTL4', 'MIS18BP1', 'MKI67', 'MPHOSPH9', 'MTMR6', 'NCAPD2', 'NCAPG', 'NCAPG2', 'NCAPH', 'NDC1', 'NDC80', 'NDE1', 'NEIL3', 'NEK2', 'NRF1', 'NUSAP1', 'OIP5', 'PAFAH2', 'PARPBP', 'PBK', 'PLEKHG3', 'PLK1', 'PLK4', 'PRC1', 'PRR11', 'PSRC1', 'PTTG1', 'PTTG3P', 'RACGAP1', 'RAD21', 'RASSF1', 'REEP4', 'SAP30', 'SHCBP1', 'SKA1', 'SLCO1B3', 'SOGA1', 'SPA17', 'SPAG5', 'SPC25', 'SPDL1', 'STIL', 'STK17B', 'TACC3', 'TAF5', 'TBC1D2', 'TBC1D31', 'TMPO', 'TOP2A', 'TPX2', 'TROAP', 'TTF2', 'TTK', 'TUBB4B', 'TUBD1', 'UBE2C', 'UBE2S', 'VANGL1', 'WEE1', 'WHSC1', 'XPO1', 'ZMYM1']





# +
# %%time
N = 1000#  18211 #  15_000# 10000 #

for l,str_inf in [ [G1S_genes_Tirosh, 'G1S Tirosh'], [G2M_genes_Tirosh, 'G2M Tirosh'],  [G1S_genes_Tirosh + G2M_genes_Tirosh, 'All Tirosh'],
                  [list_genes_fastCCsign, 'FastCC Signature'],
                  [ G1S_genes_Freeman, 'G1S Freeman' ],  [ G2M_genes_Freeman, 'G2M Freeman' ], [ G1S_genes_Freeman + G2M_genes_Freeman, 'All Freeman' ],   ]: 
    ll = set(l) & set(df_de_train.columns) 
    ll = list(ll)
    print(len(ll), str_inf )
    X = df_de_train[ll].T # .iloc[:,5:N].T
    print(X.shape)
    cm = np.corrcoef(X)
    print(cm[:3,:2])
    cm = np.abs(cm)
    cm = pd.DataFrame(cm, index = ll, columns = ll )
    print(cm.shape)
    clustergrid = sns.clustermap(cm)
    plt.title(str_inf, fontsize = 20 )
    plt.show()
    reordered_columns = clustergrid.dendrogram_col.reordered_ind
    reordered_rows = clustergrid.dendrogram_row.reordered_ind
    print(len(reordered_rows), len(reordered_columns) )
    print( list(cm.index[reordered_rows]) )
#     print( list(cm.columns[reordered_columns]) )    

# -

df_de_train

# +
# %%time
# for l,str_inf in [ [G1S_genes_Tirosh, 'G1S Tirosh'], [G2M_genes_Tirosh, 'G2M Tirosh'],  [G1S_genes_Tirosh + G2M_genes_Tirosh, 'All Tirosh'],
#                   [list_genes_fastCCsign, 'FastCC Signature'],
#                   [ G1S_genes_Freeman, 'G1S Freeman' ],  [ G2M_genes_Freeman, 'G2M Freeman' ], [ G1S_genes_Freeman + G2M_genes_Freeman, 'All Freeman' ],   ]: 
for l,str_inf in [  [G1S_genes_Tirosh + G2M_genes_Tirosh, 'All Tirosh']   ]: 
    ll = set(l) & set(df_de_train.columns) 
    ll = list(ll)
    print(len(ll), str_inf )
    X = df_de_train[ ['cell_type'] + list(df_de_train.columns[5:]) ].groupby('cell_type').median()
    X = X[ll]
    print(X.shape)
    clustergrid = sns.clustermap(X)# ,  annot=True, fmt=".2f", cmap="coolwarm" )
    plt.title(str_inf, fontsize = 20 )
    plt.show()
    reordered_columns = clustergrid.dendrogram_col.reordered_ind
    reordered_rows = clustergrid.dendrogram_row.reordered_ind
    print(len(reordered_rows), len(reordered_columns) )
    print( list(X.index[reordered_rows]) )
    print( list(X.columns[reordered_columns]) )
    
col = 'sm_name'    
for l,str_inf in [  [G1S_genes_Tirosh + G2M_genes_Tirosh, 'All Tirosh']   ]: 
    ll = set(l) & set(df_de_train.columns) 
    ll = list(ll)
    print(len(ll), str_inf )
    X = df_de_train[ [col] + list(df_de_train.columns[5:]) ].groupby(col).median()
    X = X[ll]
    print(X.shape)
    X.index = [t[:20] for t in X.index] # cut too long names
    clustergrid = sns.clustermap(X)# ,  annot=True, fmt=".2f", cmap="coolwarm" )
    plt.title(str_inf, fontsize = 20 )
    plt.show()    
    reordered_columns = clustergrid.dendrogram_col.reordered_ind
    reordered_rows = clustergrid.dendrogram_row.reordered_ind
    print(len(reordered_rows), len(reordered_columns) )
    print( list(X.index[reordered_rows]) )
    print( list(X.columns[reordered_columns]) )
# -

# # Look on compounds ( count = 146  )
#
# 15 compounds - 6 times data - only in train 
#

# +
d = df_de_train[['sm_name','sm_lincs_id','SMILES']].drop_duplicates()
print(d.shape)
d.to_csv('compounds.csv')
display( d.head(10) )

print(list(df_de_train['sm_name'].unique() ) )
# -

l = [len(s) for s in df_de_train['SMILES']]
np.sort(list(set(l)) )

display( df_de_train['sm_name'].value_counts().head(20) )
display( df_de_train['sm_name'].value_counts().tail(10) )
df_de_train['sm_name'].value_counts().value_counts()

# # Aggregations by compounds, cell_types 
#
# It is used for prediction in early versions of the notebook

# %%time
train_aggregate_mean_or_median_or_whatever = df_de_train.iloc[:,5:].quantile(0.7)# median()
train_aggregate_mean_or_median_or_whatever


# +
# %%time
d = train_aggregate_mean_or_median_or_whatever
plt.figure(figsize = (20,4) )
plt.plot(d.values)
plt.show()
plt.figure(figsize = (10,4) )
plt.hist(d.values, bins = 100)
plt.show()

display( d.describe() )
# -

# # Load,look on submission data

# %%time
fn = '/kaggle/input/open-problems-single-cell-perturbations/id_map.csv'
df_id_map = pd.read_csv(fn)
print(df_id_map.shape)
display(df_id_map)
fn = '/kaggle/input/open-problems-single-cell-perturbations/sample_submission.csv'
df = pd.read_csv(fn, index_col = 0)
print(df.shape)
df

# # Baselines. Predictions and submissions. Several approaches.

# # Target mean/median/quantile . Approch 0. 
#
# That is the first baseline for any ML task one needs to do. 
#
# Constant prediction. (Different constant for different targets, but it is constant - that means does not depend on sample)
#
#
#     V6 LB 0.666 quantile(0.7)
#     V5 LB 0.657 quantile(0.6) 
#             results are again better, that probably indicates some shift between train and public data 
#     V4 LB 0.659 - median instead of mean - results are a bit better, 
#             it might mean either a bit of presense of outliers, or  public is somewhat different from train - next experiments suggests second is true 
#     V1 LB 0.664 - submission of train means - the simplest baseline 
#

# +
# %%time
train_aggregate_mean_or_median_or_whatever = df_de_train.iloc[:,5:].quantile(0.6)# median()
train_aggregate_mean_or_median_or_whatever

# consant for each target submission:
for i,col in enumerate( df.columns ):
    df[col] = train_aggregate_mean_or_median_or_whatever[col]
    if (i%1000) == 0: print(i,col)
        
print(df.shape )        
display(df )

df.to_csv('submission.csv')
# -

# # Aggregation by compounds. Approach 1
#
# That is very similar to the previous approach, but now prediction is not quite constant, but depends on compound, but not on anything else - i.e. not from cell type).  
#
# The values for each compound are obtained by aggregation of targets over the train. 
#
# One can consider aggregated  mean, median, quantile ... Seems quantile 0.54 is better for LB. 
#
#     V9 LB 0.638 groupby by compound and quantile(0.6)
#

# +
# %%time
train_aggr = df_de_train[ ['sm_name'] + list(df_de_train.columns[5:])  ].groupby('sm_name' ).quantile(0.6)# median()
train_aggr

df = df_id_map.merge(train_aggr, how = 'left', on = 'sm_name')
df = df.set_index('id').iloc[:,2:]
display( df )

df.to_csv('submission.csv')
# -

# #  Aggregation by compounds with denoising. Approach 2. 
#
# It is similar to the approach above but we use "denoising for poors" - i.e. reduce dimension and then inverse transform it back, the common wisdom (which was e,g, helpful on the previous challenge) that keeping only N main components helps to reduce noise in the data, thus prediction migh become better. We see that it is indeed true for the that challenge.
#
# That means: first reduce dimensions of the data by say pca100, and  take pca inverse transfrom - we get kind of "denoised data".
#
# And then apply same as in the previous approach i.e. groupby by compound and aggregation - mean/median/quantile()         
# So idea is - pca100 reductions hopefully kills some noise. 
# In the previous challenge it worked okay, but the number of samples was not 614, but near 100 000, still any biological data is very noisy and highly correlated - thus it is natural to expect it will help and it does indeed. 
#
#     V18 LB 0.623 TSVD-35 denoising  quantile 0.54, tsvd is better than pca/ica - similar to previous OP
#     V15 - bug -  LB 0.626 - NO it was not: TSVD-35 denoising  quantile 0.54 - so worse than pca,ICA, at least for these params
#     V14 LB 0.624 - ICA-35 denoising   quantile 0.54 - so ICA is not better than just pca at least for these the same params
#     Fork: LB 0.624 - pca-35 denoising, quantile 0.54 
#     V13 LB 0.626 pca25 denoising 
#     V12 LB 0.627 same with pca100 denoising
#

# ## Key params

# predict_method = 'train_aggregation_by_compounds'
# predict_method = 'train_aggregation_by_compounds_with_denoising_pca'
# predict_method = 'train_aggregation_by_compounds_with_denoising_ICA'
predict_method = 'train_aggregation_by_compounds_with_denoising_TSVD'
quantile = 0.54
n_components = 35


# ## Denoising (by pca/ica/tsvd) and aggregation 

# +
# %%time 
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD

Y = df_de_train.iloc[:,5:]
print(X.shape)
if '_pca' in predict_method:
    str_inf_target_dimred = 'PCA' 
    reducer = PCA(n_components=n_components )
elif '_ICA' in predict_method:
    str_inf_target_dimred = 'ICA' 
#     reducer = PCA(n_components=n_components )
    reducer = FastICA(n_components=n_components, random_state=0, whiten='unit-variance')
elif '_TSVD' in predict_method:
    str_inf_target_dimred = 'TSVD' 
#     reducer = PCA(n_components=n_components )
#     reducer = FastICA(n_components=n_components, random_state=0, whiten='unit-variance')
    reducer = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
else:
    str_inf_target_dimred = ''
    
print(str_inf_target_dimred , reducer)
Yr = reducer.fit_transform(Y)
Yr_inv_trans = reducer.inverse_transform(Yr)
df_red_inv_trans = pd.DataFrame(Yr_inv_trans, columns = df_de_train.columns[5:])
df_red_inv_trans['sm_name'] = df_de_train['sm_name']

train_aggr_denoised = df_red_inv_trans.groupby('sm_name' ).quantile(quantile)# median()
train_aggr_denoised

# -

# ## Prepapre predictions for submit

# +
# %%time

if predict_method.startswith('train_aggregation_by_compounds_with_denoising_'):
    df = df_id_map.merge(train_aggr_denoised, how = 'left', on = 'sm_name')
    df = df.set_index('id').iloc[:,2:]
elif predict_method == 'train_aggregation_by_compounds': # Older - approach 1 
    ## Prepare predictions by direct aggregation of targets by compounds
    train_aggr_direct = df_de_train[ ['sm_name'] + list(df_de_train.columns[5:])  ].groupby('sm_name' ).quantile(quantile)# median()
    train_aggr_direct
    df = df_id_map.merge(train_aggr, how = 'left', on = 'sm_name')
    df = df.set_index('id').iloc[:,2:]
else: # Older - approach 0 
    # consant for each target submission:
    for i,col in enumerate( df.columns ):
        df[col] = train_aggregate_mean_or_median_or_whatever[col]
        if (i%1000) == 0: print(i,col)
    
df    
# -

# %%time
df.to_csv('submission.csv')

# # Target endcoding and Ridge. Approach 3. 
#
# Here we will try to train the first ML model. 
#
# We will do target encodnging and train Ridge model upon the target encoded cell type and compound - two features which are actually categorical, 
#
#     Versions 23-27 - first modeling approach - target encode cell type and compound and simple Ridge model - results currently worse than naive approach used before - without models. 
#         V27 LB 0.659 LB Ridge nCT1 nCD25 Al10 TSVD35 - even more relaxed : alpha and encoded_compound size 
#         V26 LB 0.668 try to relax a model a bit: Ridge nCT1 nCD10 Al100 TSVD35 - results are better near mean/zero submission
#         V25 LB 0.677 - stronger constraining the model: Ridge nCT1 nCD5 Al100 TSVD35, but still we are worse than even predict by mean 
#         V24 LB 0.702 - try avoid overfit Ridge nCT3 nCD10 Al100 TSVD35 - better results but still bad 
#         V23 LB 0.747 Ridge model in target encoded features Ridge nCT10 nCD35 Al1 TSVD35 - modeling gives worse results than simple approach, might be bug or overfit
#

# ## Key params

# +
n_components_for_cell_type_encoding = 1
n_components_for_compound_encoding = 25
alpha_regularization_for_linear_models = 10

# predict_method

model_type = 'Ridge'


# +
str_model_id = model_type
str_model_id += ' nCT'+ str(n_components_for_cell_type_encoding)
str_model_id += ' nCD'+ str(n_components_for_compound_encoding)
str_model_id += ' Al'+ str(alpha_regularization_for_linear_models)
str_model_id += ' ' +str_inf_target_dimred+str( n_components )

print( str_model_id )

# -

# ## Target encoded features 

# +
# %%time
# Yr = reducer.fit_transform(X)
# n_components_for_cell_type_encoding = 10
df_tmp = pd.DataFrame(Yr[:, :n_components_for_cell_type_encoding  ], index = df_de_train.index  )
df_tmp['column for aggregation'] = df_de_train['cell_type']
df_cell_type_encoded = df_tmp.groupby('column for aggregation').quantile( quantile )
print('df_cell_type_encoded.shape', df_cell_type_encoded.shape )
display( df_cell_type_encoded )


# n_components_for_compound_encoding = 10
df_tmp = pd.DataFrame(Yr[:, :n_components_for_compound_encoding  ], index = df_de_train.index  )
df_tmp['column for aggregation'] = df_de_train['sm_name']
df_compound_encoded = df_tmp.groupby('column for aggregation').quantile( quantile )
print('df_compound_encoded.shape', df_compound_encoded.shape )
display( df_compound_encoded )


# -

# ## Prepare X_train, X_submit - target encoded cell type and compound features

# +
# %%time
X_train = np.zeros( (len( df_de_train ) , n_components_for_cell_type_encoding + n_components_for_compound_encoding ))

for i in range(len( X_train )):
    cell_type = df_de_train['cell_type'].iat[i] 
    X_train[i,:n_components_for_cell_type_encoding] = df_cell_type_encoded.loc[cell_type,:].values  
    compound = df_de_train['sm_name'].iat[i] 
    X_train[i,n_components_for_cell_type_encoding:] = df_compound_encoded.loc[ compound, : ].values
print( X_train.shape)     
print( X_train)     
    

X_submit = np.zeros( (len( df_id_map ) , n_components_for_cell_type_encoding + n_components_for_compound_encoding ))
for i in range(len( X_submit )):
    cell_type = df_id_map['cell_type'].iat[i] 
    X_submit[i,:n_components_for_cell_type_encoding] = df_cell_type_encoded.loc[cell_type,:].values  
    compound = df_id_map['sm_name'].iat[i] 
    X_submit[i,n_components_for_cell_type_encoding:] = df_compound_encoded.loc[ compound, : ].values
    
    
print( X_submit.shape)     
print( X_submit)     
    
# -

# ## Modeling

# +
# %%time
from sklearn.linear_model import Ridge

model = Ridge(alpha=alpha_regularization_for_linear_models)
print(model)
model.fit(X_train, Yr)

Y_submit = reducer.inverse_transform(   model.predict(X_submit) )
print(Y_submit.shape)
Y_submit
# -

# ## Save submission CSV

# %%time
df_submit = pd.DataFrame(Y_submit, columns = df_de_train.columns[5:])
df_submit.index.name = 'id'
print( df_submit.shape )
display(df_submit)
df_submit.to_csv('submission.csv')

# # Catboost -  approach 4
#
#     V34 LB 0.688 iterations=3,  depth = 6 
#     V33 LB 0.709 Catboost iterations 10,  depth = 6    - try to reduce overfit - but even worse  
#     V32 LB 0.912 - Catboost depth = 2, iteration 100 - try to reduce overfit - but even worse  
#     V30 LB 0.905 ( terrible :)  - Catboost first draft -  depth = 6 , iteration 100
#

import catboost
from catboost import CatBoostRegressor, Pool
categorical_features = ['cell_type','sm_name']
model = CatBoostRegressor(iterations=3,  # Number of boosting iterations
                          depth=6,        # Depth of the trees
                          learning_rate=0.1,  # Learning rate
                          loss_function='RMSE',  # Specify your loss function (e.g., RMSE for regression)
                          cat_features=categorical_features,  # Categorical features
                          verbose=0)  # Set verbose to 0 to suppress output


Yr.shape

# +
# %%time
Y_reduced_submit = np.zeros( (len(df_id_map) , Yr.shape[1] )   )
df_train = df_de_train[['cell_type','sm_name']]

for k in range(Yr.shape[1]):
    train_data = Pool(data=df_train, 
                      label=Yr[:,k],
                      cat_features=categorical_features)
    test_data = Pool(data=df_id_map[['cell_type','sm_name']], 
                      cat_features=categorical_features)
    model.fit(train_data)    
    predicted_value = model.predict(test_data)
    Y_reduced_submit[:,k] = predicted_value
    
    
# -

# %%time
Y_submit = reducer.inverse_transform(  Y_reduced_submit )
print(Y_submit.shape)
Y_submit

# %%time
df_submit = pd.DataFrame(Y_submit, columns = df_de_train.columns[5:])
df_submit.index.name = 'id'
print( df_submit.shape )
display(df_submit)
df_submit.to_csv('submission.csv')

# # Target Encoded simplified. Approach 5
#
# Target encoding and Ridge, but here is Ridge is done only on 1 feature and thus it should be quite similar to approach 3 with just denoise+groupby, but a kind of Ridge instead of "mean/median/quantile". 
#
# If it were sucessful we will add more features. 
#
# But prelimary result is not good: 
#
#     V37 0.707 approach 5 - aggr by compound + Ridge.
#

# +
# %%time 

# predict_method = 'train_aggregation_by_compounds'
# predict_method = 'train_aggregation_by_compounds_with_denoising_pca'
# predict_method = 'train_aggregation_by_compounds_with_denoising_ICA'
predict_method = 'train_aggregation_by_compounds_with_denoising_TSVD'
quantile = 0.54
n_components = 35

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD

Y = df_de_train.iloc[:,5:]
print(X.shape)
if '_pca' in predict_method:
    str_inf_target_dimred = 'PCA' 
    reducer = PCA(n_components=n_components )
elif '_ICA' in predict_method:
    str_inf_target_dimred = 'ICA' 
#     reducer = PCA(n_components=n_components )
    reducer = FastICA(n_components=n_components, random_state=0, whiten='unit-variance')
elif '_TSVD' in predict_method:
    str_inf_target_dimred = 'TSVD' 
#     reducer = PCA(n_components=n_components )
#     reducer = FastICA(n_components=n_components, random_state=0, whiten='unit-variance')
    reducer = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
else:
    str_inf_target_dimred = ''
    
print(str_inf_target_dimred , reducer)
Yr = reducer.fit_transform(Y)
# Yr_inv_trans = reducer.inverse_transform(Yr)
df_red = pd.DataFrame(Yr)
df_red['sm_name'] = df_de_train['sm_name']

train_aggr_red = df_red.groupby('sm_name' ).quantile(quantile)# median()
train_aggr_red = train_aggr_red.reset_index()
train_aggr_red

# +
# %%time
df_X =  df_de_train[ ['sm_name']].copy()
df_X =  df_X.merge( train_aggr_red, on = 'sm_name' )
df_X = df_X.iloc[:,1:]
display( df_X )
 

df_X_submit = df_id_map[['sm_name']].copy()
df_X_submit = df_X_submit.merge( train_aggr_red, on = 'sm_name' ).iloc[:,1:]
display(df_X_submit)

# +
# %%time 
from sklearn.linear_model import Ridge
alpha_regularization_for_linear_models = 1
model = Ridge(alpha=alpha_regularization_for_linear_models)

Yr_submit = np.zeros(   (len(df_id_map) , Yr.shape[1] )   )
for i in range(df_X.shape[1]):
    model.fit(df_X[[ df_X.columns[i] ]], Yr[:,i])
    Yr_submit[:,i] = model.predict(df_X_submit[[ df_X_submit.columns[i] ]])
    

# +
# %%time
Y_submit = reducer.inverse_transform(  Yr_submit )
print(Y_submit.shape)
Y_submit

df_submit = pd.DataFrame(Y_submit, columns = df_de_train.columns[5:])
df_submit.index.name = 'id'
print( df_submit.shape )
display(df_submit)
df_submit.to_csv('submission.csv')
# -





print('%.1f seconds passed total '%(time.time()-t0start) )
print('%.1f minutes passed total '%( (time.time()-t0start)/60)  )
print('%.2f hours passed total '%( (time.time()-t0start)/3600)  )




