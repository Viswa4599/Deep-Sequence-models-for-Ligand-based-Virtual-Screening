import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import glob
from utils import split
import rdkit
import pandas as pd
from sklearn.utils import class_weight
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
import re

state_dict = torch.load('.save/trfm_chembl_4_200000.pkl')
vocab = WordVocab.load_vocab('data/chembl_vocab.pkl')
trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 2)
trfm.load_state_dict(state_dict)
trfm.eval()

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4

print('Total parameters:', sum(p.numel() for p in trfm.parameters()))

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


dataset = pd.read_csv('../AID_CSV/1332.csv',low_memory=False)
X = dataset['SMILES']
y = dataset['Activity']

X_split = [split(sm) for sm in X.to_numpy()]
xid,_ = get_array(X_split)
X = trfm.encode(torch.t(xid))
print(X.shape)

X = pd.DataFrame(X,header = list(range(1,1025)))
X.append(y,axis = 1)
X.to_csv('')