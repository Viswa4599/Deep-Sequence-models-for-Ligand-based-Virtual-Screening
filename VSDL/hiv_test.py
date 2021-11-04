import pickle
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from utils import split
import rdkit
import pandas as pd
from sklearn.utils import class_weight
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.metrics import Precision, Recall, AUC, Accuracy
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score,average_precision_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM,Embedding, Bidirectional
from keras.layers.core import Dropout
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras import regularizers
from keras.optimizers import RMSprop, Adam
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import tensorflow as tf
from keras import backend as k



rates = 2**np.arange(7)/80
print(rates)

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

vocab = WordVocab.load_vocab('data/new_aid_vocab.pkl')

trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 2)
trfm.load_state_dict(torch.load('.save/trfm_NEWAID_1_210000.pkl'))
trfm.eval()
print('Total parameters:', sum(p.numel() for p in trfm.parameters()))


df = pd.read_csv('../AID_CSV/1332.csv')

extra_features = df[['PUBCHEM_CACTVS_COMPLEXITY','PUBCHEM_CACTVS_HBOND_ACCEPTOR',
                    'PUBCHEM_CACTVS_HBOND_DONOR','PUBCHEM_CACTVS_ROTATABLE_BOND','PUBCHEM_XLOGP3_AA'
                    ,'PUBCHEM_EXACT_MASS','PUBCHEM_CACTVS_TPSA','PUBCHEM_HEAVY_ATOM_COUNT','PUBCHEM_ATOM_DEF_STEREO_COUNT',
                    'PUBCHEM_CACTVS_TAUTO_COUNT']]

min_max_scaler = preprocessing.MinMaxScaler()
extra_scaled = min_max_scaler.fit_transform(extra_features)
extra_features = pd.DataFrame(extra_scaled).fillna(0)

print(df['Activity'].value_counts())
def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm)>218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109]+sm[-109:]
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1]*len(ids)
    padding = [pad_index]*(seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg

def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a,b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)

def ablation_hiv(X, X_test, y, y_test, rate, n_repeats):
    auc = np.empty(n_repeats)
    prec = np.empty(n_repeats)
    rec = np.empty(n_repeats)
    acc = np.empty(n_repeats)
    f1 = np.empty(n_repeats)
    pr_auc = np.empty(n_repeats)
    for i in range(n_repeats):
        
        clf = MLPClassifier(max_iter=1000,verbose = True,n_iter_no_change = 25)
        input_shape = X.shape[1:]
        output_shape = 1
        model = Sequential()
        model.add(Dense(100,activation = 'tanh',input_shape  = input_shape))
        model.add(Dense(output_shape,activation="sigmoid"))

        if rate==1:
            X_train, y_train = X,y
        else:
            X_train, _, y_train, __ = train_test_split(X, y, test_size=1-rate, stratify=y)

        print('MLP training for {:1f}'.format(rate))
        class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
        print(class_weights)
        weights = {0:class_weights[0],1:class_weights[1]}
        int_weights = {0:2.5,1:1}
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                 patience=5, min_lr=0.00000001,epsilon = 1e-04,verbose =1)
        model.compile(loss="binary_crossentropy", optimizer='adam',metrics = [AUC(), Precision(), Recall()])
        #model.fit(X_train,y_train, epochs=250,callbacks = [reduce_lr])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
       #y_pred = model.predict_classes(X_test)
       #y_pred = [item[0] for item in y_pred]
        #print(y_score)
        print(pd.Series(y_test).value_counts())
        print(pd.Series(y_pred).value_counts())
        auc[i] = roc_auc_score(y_test, y_pred)
        prec[i] = precision_score(y_test,y_pred)
        rec[i] = recall_score(y_test,y_pred)
        f1[i] = f1_score(y_test,y_pred)
        acc[i] = accuracy_score(y_test,y_pred)
        pr_auc[i] = average_precision_score(y_test,y_pred)
        print(pr_auc[i])


    ret = {}
    ret['auc mean'] = np.mean(auc)
    ret['auc std'] = np.std(auc)
    ret['prec mean'] = np.mean(prec)
    ret['prec std'] = np.std(prec)
    ret['rec mean'] = np.mean(rec)
    ret['rec std'] = np.std(rec)
    ret['f1 mean'] = np.mean(f1)
    ret['f1 std'] = np.std(f1)
    ret['acc mean'] = np.mean(acc)
    ret['acc std'] = np.std(acc)
    ret['prauc mean'] = np.mean(pr_auc)
    ret['prauc std'] = np.std(pr_auc)

    

    return ret


df_train,df_test,extra_train,extra_test = train_test_split(df,extra_features,test_size = 0.3,random_state = 42)

#df_train = df[np.array(list(map(len, df['smiles'])))<=218]
#df_test = df[np.array(list(map(len, df['smiles'])))>218]
x_split = [split(sm) for sm in df_train['SMILES'].values]
xid, _ = get_array(x_split)
X = trfm.encode(torch.t(xid))
extra_train = extra_train.to_numpy()
X = np.concatenate((X,extra_train),axis = 1)
X = min_max_scaler.fit_transform(X)
print("Transformer encoding done")
print(X.shape)

x_split = [split(sm) for sm in df_test['SMILES'].values]
xid, _ = get_array(x_split)
X_test = trfm.encode(torch.t(xid))
extra_test = extra_test.to_numpy()
X_test = np.concatenate((X_test,extra_test),axis = 1)
print(X_test.shape)
X_test = min_max_scaler.fit_transform(X_test)
y, y_test = df_train['Activity'].values, df_test['Activity'].values

rf = RandomForestClassifier(class_weight = 'balanced')
rf.fit(extra_train,y)
rfpred = rf.predict(extra_test)

print("Prec : ", precision_score(y_test,rfpred))
print("Rec : ", recall_score(y_test,rfpred))
print("F1 : ", f1_score(y_test,rfpred))
print("Acc : ", accuracy_score(y_test,rfpred))

'''
scores = []
for rate in rates:
    score_dic = ablation_hiv(X, X_test, y, y_test, rate, 20)
    print(rate, score_dic)
    scores.append(score_dic['auc mean'])
print(np.mean(scores))
'''

#raise KeyboardInterrupt

score_dic = ablation_hiv(X, X_test, y, y_test, 1, 1)
print(score_dic)



