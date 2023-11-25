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

# + active=""
# # **<span style="color:#F7B2B0;">Introduction</span>**
#
# Human biology can be complex, in part due to the function and interplay of the body's approximately 37 trillion cells, which are organized into tissues, organs, and systems. However, recent advances in single-cell technologies have provided unparalleled insight into the function of cells and tissues at the level of DNA, RNA, and proteins. Yet leveraging single-cell methods to develop medicines requires mapping causal links between chemical perturbations and the downstream impact on cell state. These experiments are costly and labor intensive, and not all cells and tissues are amenable to high-throughput transcriptomic screening. If data science could help accurately predict chemical perturbations in new cell types, it could accelerate and expand the development of new medicines.
# Several methods have been developed for drug perturbation prediction, most of which are variations on the autoencoder architecture (Dr.VAE, scGEN, and ChemCPA). However, these methods lack proper benchmarking datasets with diverse cell types to determine how well they generalize. The largest available training dataset is the NIH-funded Connectivity Map (CMap), which comprises over 1.3M small molecule perturbation measurements. However, the CMap includes observations of only 978 genes, less than 5% of all genes. Furthermore, the CMap data is comprised almost entirely of measurements in cancer cell lines, which may not accurately represent human biology.
#
# # **<span style="color:#F7B2B0;">Goal</span>**
#
# The goal of this competition is to accurately predict chemical perturbations in new cell types could accelerate the discovery and enable the creation of new medicines to treat or cure disease.
#
# # **<span style="color:#F7B2B0;">Install the Libraries</span>**
#
# -

# !pip install -q wandb tabpfn

# # **<span style="color:#F7B2B0;">Import the Packages</span>**

# + _uuid="7b575297-32e1-4d21-ab73-817900926e53" _cell_guid="a3899b3e-7985-483d-b155-33283f5b5084" jupyter={"outputs_hidden": false}
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from kaggle_secrets import UserSecretsClient
import wandb
from datetime import datetime
# -

# <img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67">
#
# > I will be integrating W&B for visualizations and logging artifacts!
# > 
# > [Open Problems = Single Cell Pertubations Project on W&B Dashboard](https://wandb.ai/usharengaraju/open_problems)
# > 
# > - To get the API key, create an account in the [website](https://wandb.ai/site) .
# > - Use secrets to use API Keys more securely 

# Setup user secrets for login
user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("api_key") 
wandb.login(key=wandb_api)

# # **<span style="color:#F7B2B0;">Data Pipeline</span>**

fn = '/kaggle/input/open-problems-single-cell-perturbations/de_train.parquet'
df_de_train = pd.read_parquet(fn)# , index_col = 0)
print(df_de_train.shape)
df_de_train.head()

# +
run = wandb.init(project = 'open_problems',
                 config = {},
                 save_code = True,
                 
)
table = wandb.Table(dataframe=df_de_train)

wandb.log({"Table":table})
run.finish()
# -

X = df_de_train.iloc[:,:5]
y=  df_de_train.iloc[:,5:]
X.shape, y.shape


from sklearn.decomposition import TruncatedSVD
reducer = TruncatedSVD(n_components=35, n_iter=7, random_state=42)
Yr = reducer.fit_transform(y)

Yr.shape

# +
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# Encode Categorical Data
for col in X.columns:
  X[col]=encoder.fit_transform(X[col])

# + _uuid="1b778375-9872-4f05-a5a2-e2bd3dcf46b2" _cell_guid="8691ffdb-b3c1-4499-8e24-f48547ec1193" jupyter={"outputs_hidden": false}
X_train, X_val, y_train, y_val = train_test_split(X.values, Yr, test_size=0.2)

# +
from wandb.keras import WandbCallback, WandbMetricsLogger
run = wandb.init(project = 'open_problems',
                 save_code = True,
                 name='tabtransformer'
                 
)
# -

# # **<span style="color:#F7B2B0;">Tab transformer</span>**
#
# [Source](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/tabtransformer.ipynb)
#
# The TabTransformer architecture works as follows:
#
# 📌 All the categorical features are encoded as embeddings, using the same embedding_dims. This means that each value in each categorical feature will have its own embedding vector.
#
# 📌 A column embedding, one embedding vector for each categorical feature, is added (point-wise) to the categorical feature embedding.
#
# 📌 The embedded categorical features are fed into a stack of Transformer blocks. Each Transformer block consists of a multi-head self-attention layer followed by a feed-forward layer.
#
# 📌 The outputs of the final Transformer layer, which are the contextual embeddings of the categorical features, are concatenated with the input numerical features, and fed into a final MLP block.
#
# 📌 A softmax classifer is applied at the end of the model.
#
# The [paper](https://arxiv.org/pdf/2012.06678.pdf) discusses both addition and concatenation of the column embedding in the Appendix: Experiment and Model Details section. The architecture of TabTransformer is shown below, as presented in the paper.
#
# ![](https://i.imgur.com/kSB0jYw.png)

# +
from tensorflow import keras
import tensorflow as tf 
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # parametreleri
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        # batch-layer
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TabTransformer(keras.Model):

    def __init__(self, 
            categories,
            num_continuous,
            dim,
            dim_out,
            depth,
            heads,
            attn_dropout,
            ff_dropout,
            mlp_hidden,
            normalize_continuous = True):
        
        super(TabTransformer, self).__init__()

        # --> continuous inputs
        self.normalize_continuous = normalize_continuous
        if normalize_continuous:
            self.continuous_normalization = layers.LayerNormalization()

        # --> categorical inputs

        # embedding
        self.embedding_layers = []
        for number_of_classes in categories:
            self.embedding_layers.append(layers.Embedding(input_dim = number_of_classes, output_dim = dim))

        # concatenation
        self.embedded_concatenation = layers.Concatenate(axis=1)

        # adding transformers
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(TransformerBlock(dim, heads, dim))
        self.flatten_transformer_output = layers.Flatten()

        # --> MLP
        self.pre_mlp_concatenation = layers.Concatenate()

        # mlp layers
        self.mlp_layers = []
        for size, activation in mlp_hidden:
            self.mlp_layers.append(layers.Dense(size, activation=activation))

        self.output_layer = layers.Dense(dim_out)

    def call(self, inputs):
        categorical_inputs = inputs
#         print(inputs[:,0])
        # --> categorical
        embedding_outputs = []
        for i in range(5):
            embedding_outputs.append(tf.expand_dims(self.embedding_layers[i](categorical_inputs[:,i]),axis=1))
#         print(embedding_outputs[0].shape)
        categorical_inputs = self.embedded_concatenation(embedding_outputs)
#         categorical_inputs = tf.expand_dims(categorical_inputs,axis=1)
        for transformer in self.transformers:
            categorical_inputs = transformer(categorical_inputs)
        contextual_embedding = self.flatten_transformer_output(categorical_inputs)


        for mlp_layer in self.mlp_layers:
            mlp_input = mlp_layer(contextual_embedding)

        return self.output_layer(mlp_input)
# -

nu = []
for col in X.columns:
    nu.append(len(X[col].unique()))

from tensorflow.keras.optimizers import Adam


tabTransformer = TabTransformer(
    categories = nu, # number of unique elements in each categorical feature
    num_continuous = 5,      # number of numerical features
    dim = 16,                # embedding/transformer dimension
    dim_out = 35,             # dimension of the model output
    depth = 6,               # number of transformer layers in the stack
    heads = 8,               # number of attention heads
    attn_dropout = 0.1,      # attention layer dropout in transformers
    ff_dropout = 0.1,        # feed-forward layer dropout in transformers
    mlp_hidden = [(32, 'relu'), (16, 'relu')] # mlp layer dimensions and activations
)
tabTransformer.compile(Adam(0.001),'mae',metrics=['mae'])
tabTransformer.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=32,epochs=30,callbacks=[WandbMetricsLogger()])

run.finish()

# # **<span style="color:#F7B2B0;">SAINT</span>**
#
# SAINT, performs attention over both rows and columns, and it includes an enhanced embedding method. A new contrastive self-supervised pre-training method is used when labels are scarce. SAINT consistently improves performance over previous deep learning methods, and it even outperforms gradient boosting methods, including XGBoost, CatBoost, and LightGBM, on average over a variety of benchmark tasks.
#
# ![](https://i.imgur.com/WVEY6uy.png)

# +
run = wandb.init(project = 'open_problems',
                 save_code = True,
                 name='SAINT'
                 
)

# +
import tensorflow as tf
trainds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
trainds = trainds.batch(32, drop_remainder = True)

valds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
valds = valds.batch(32, drop_remainder = True)

# +
import tensorflow as tf
from wandb.keras import WandbCallback, WandbMetricsLogger
from tensorflow.keras.layers import *
from tensorflow.keras.activations import gelu, softmax
from tensorflow.keras.models import Sequential
class MHA(Layer):
    '''
    Multi-Head Attention Layer
    '''
    
    def __init__(self, num_head, dropout = 0):
        super(MHA, self).__init__()
        
        # Constants
        self.num_head = num_head
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        query_shape = input_shape
        d_model = query_shape[-1]
        units = d_model // self.num_head
        
        # Loop for Generate each Attention
        self.layer_q = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build(query_shape)
            self.layer_q.append(layer)
            
        self.layer_k = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build(query_shape)
            self.layer_k.append(layer)
            
        self.layer_v = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build(query_shape)
            self.layer_v.append(layer)
            
        self.out = Dense(d_model, activation = None, use_bias = False)
        self.out.build(query_shape)
        self.dropout = Dropout(self.dropout_rate)
        self.dropout.build(query_shape)
        
    def call(self, x):
        d_model = x.shape[-1]
        scale = d_model ** -0.5
        
        attention_values = []
        for i in range(self.num_head):
            attention_score = softmax(tf.matmul(self.layer_q[i](x), self.layer_k[i](x), transpose_b=True) * scale)
            attention_final = tf.matmul(attention_score, self.layer_v[i](x))
            attention_values.append(attention_final)
            
        attention_concat = tf.concat(attention_values, axis = -1)
        out = self.out(self.dropout(attention_concat))
        
        return out

class IMHA(Layer):
    '''
    Intersample Multi Head Attention
    Attend on row(samples) not column(features)
    '''
    
    def __init__(self, num_head, dropout = 0):
        super(IMHA, self).__init__()
        
        # Constants
        self.num_head = num_head
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        b, n, d = input_shape
        query_shape = input_shape
        units = (d * n) // self.num_head
        # Loop for Generate each Attention
        self.layer_q = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build([1, b, int(n * d)])
            self.layer_q.append(layer)
            
        self.layer_k = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build([1, b, int(n * d)])
            self.layer_k.append(layer)
            
        self.layer_v = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build([1, b, int(n * d)])
            self.layer_v.append(layer)
            
        self.out = Dense(d, activation = None, use_bias = False)
        self.out.build(query_shape)
        self.dropout = Dropout(self.dropout_rate)
        self.dropout.build(query_shape)
        
    def call(self, x):
        b, n, d = x.shape
        scale = d ** -0.5
        x = tf.reshape(x, (1, b, int(n * d)))
        attention_values = []
        
        for i in range(self.num_head):
            attention_score = softmax(tf.matmul(self.layer_q[i](x), self.layer_k[i](x), transpose_b=True) * scale)
            attention_final = tf.matmul(attention_score, self.layer_v[i](x))
            attention_final = tf.reshape(attention_final, (b, n, int(d / self.num_head)))
            attention_values.append(attention_final)
            
        attention_concat = tf.concat(attention_values, axis = -1)
        out = self.out(self.dropout(attention_concat))
        
        return out

class FeedForwardNetwork(Layer):
    def __init__(self, dim, dropout = 0.0):
        super(FeedForwardNetwork, self).__init__()
        self.dense = Dense(dim, activation = 'gelu')
        self.dropout = Dropout(dropout)
        
    def call(self, x):
        return self.dropout(self.dense(x))

class CustomEmbedding(Layer):
    def __init__(self, num_categorical, dim):
        super(CustomEmbedding, self).__init__()
        self.num_categorical = num_categorical
        self.dim = dim
        
    def build(self, input_shape):
        b, n = input_shape
        self.embedding_categorical = Embedding(self.dim * 2, self.dim)
        self.embedding_categorical.build([b, self.num_categorical])
        
        self.embedding_numerical = Dense(self.dim, activation = 'relu')
        self.embedding_numerical.build([b, int(n - self.num_categorical), 1])
        
    def call(self, x):
        b, n = x.shape
        categorical_x = x[:, :self.num_categorical]
        numerical_x = x[:, self.num_categorical:]
        numerical_x = tf.reshape(numerical_x, (b, int(n - self.num_categorical), 1))
        
        embedded_cat = self.embedding_categorical(categorical_x)
        embedded_num = self.embedding_numerical(numerical_x)
    
        embedded_x = tf.concat([embedded_cat, embedded_num], axis = 1)
        
        return embedded_x


class SAINT(Layer):
    def __init__(self, repeat, num_categorical, EMB_DIM, MHA_HEADS, IMHA_HEADS):
        super(SAINT, self).__init__()
        self.repeat = repeat
        self.layer_mha = []
        self.layer_imha = []
        self.layer_ffn = []
        self.layer_layernorm = []
        self.embedding = CustomEmbedding(num_categorical, EMB_DIM)
        
        for _ in range(repeat):
            mha = MHA(MHA_HEADS)
            imha = IMHA(IMHA_HEADS)
            ffn_1 = FeedForwardNetwork(EMB_DIM)
            ffn_2 = FeedForwardNetwork(EMB_DIM)
            layernorm_1 = LayerNormalization()
            layernorm_2 = LayerNormalization()
            layernorm_3 = LayerNormalization()
            layernorm_4 = LayerNormalization()
            
            self.layer_mha.append(mha)
            self.layer_imha.append(imha)
            self.layer_ffn.append(ffn_1)
            self.layer_ffn.append(ffn_2)
            self.layer_layernorm.append(layernorm_1)
            self.layer_layernorm.append(layernorm_2)
            self.layer_layernorm.append(layernorm_3)
            self.layer_layernorm.append(layernorm_4)
            
    def call(self, x):
        x = self.embedding(x)
        # Depth of SAINT Layer
        for i in range(self.repeat):
            # Multi-Head part
            x = self.layer_layernorm[i](self.layer_mha[i](x)) + x
            x = self.layer_layernorm[i+1](self.layer_ffn[i](x)) + x
            
            # Intersample Multi-Head part
            x = self.layer_layernorm[i+2](self.layer_imha[i](x)) + x
            x = self.layer_layernorm[i+3](self.layer_ffn[i+1](x)) + x
       
        # only using cls token for final output
        out = x[:, 0] # CLS Token
        
        return out
nu = []
for col in X.columns:
    nu.append(len(X[col].unique()))
    
model = Sequential([
            Input(shape = (5),batch_size=32),
            SAINT(1, 4, 64, 8, 8),
            Dense(35, activation = 'linear')
        ])

model.compile(tf.keras.optimizers.Adam(0.001), 'mae',metrics=['mae'])
model.summary()
model.fit(trainds,epochs=20,batch_size=32,validation_data=valds,callbacks=[WandbMetricsLogger()])
# -

run.finish()
