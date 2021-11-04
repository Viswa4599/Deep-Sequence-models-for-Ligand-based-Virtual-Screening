import os
import deepchem as dc
from deepchem.models import GraphConvModel
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,roc_auc_score

def precision_average(y_test, y_pred):
  return precision_score(y_test, y_pred)

def ablation_hiv_dc(dataset, test_data, rate, n_repeats):
    auc = np.empty(n_repeats)
    recalls = np.empty(n_repeats)
    f1s = np.empty(n_repeats)
    precisions = np.empty(n_repeats)
    for i in range(n_repeats):
        clf = GraphConvModel(n_tasks=1, batch_size=64, mode='classification')
        splitter = dc.splits.RandomStratifiedSplitter()
        train_data, _, __ = splitter.train_valid_test_split(dataset, frac_train=rate, frac_valid=1-rate, frac_test=0)
        clf.fit(train_data)
        metrics = [dc.metrics.Metric(dc.metrics.roc_auc_score),dc.metrics.Metric(dc.metrics.f1_score),dc.metrics.Metric(dc.metrics.recall_score)]
        scores = clf.evaluate(test_data, metrics)
        auc[i] = scores['roc_auc_score']
        recalls[i] = scores['recall_score']
        f1s[i] = scores['f1_score']
        precisions[i] = f1s[i]*recalls[i]/(2*recalls[i]-f1s[i])
        print(auc[i])
        print(recalls[i])
        print(f1s[i])
        print(precisions[i])
    ret = {}
    ret['auc mean'] = np.mean(auc)
    ret['auc std'] = np.std(auc)
    ret['rec mean'] = np.mean(recalls)
    ret['rec std'] = np.std(recalls)
    ret['f1 mean'] = np.mean(f1s)
    ret['f1 std'] = np.std(f1s)
    ret['precisions mean'] = np.mean(precisions)
    ret['precisions std'] = np.std(precisions)
    return ret

df = pd.read_csv('../AID_CSV/778.csv')
print(df.shape)
df.head()

featurizer = dc.feat.ConvMolFeaturizer()
loader = dc.data.CSVLoader(
      tasks=['Activity'],
      smiles_field='SMILES',
      featurizer=featurizer)
dataset = loader.featurize('../AID_CSV/778.csv')

splitter = dc.splits.RandomStratifiedSplitter()

train_data, test_data,_ = splitter.train_valid_test_split(dataset, frac_train=0.8,frac_valid = 0.2)
print(len(train_data))
print(len(test_data))
#train_data = dataset.select(np.where(np.array(list(map(len, df['smiles'])))<=218)[0])
#test_data = dataset.select(np.where(np.array(list(map(len, df['smiles'])))>218)[0])

#cores = []
#for rate in rates:
#    score_dic = ablation_hiv_dc(train_data, test_data, rate, 20)
#    print(rate, score_dic)
#    scores.append(score_dic['auc mean'])
#print(np.mean(scores))

score_dic = ablation_hiv_dc(train_data, test_data, 1, 1)
print(score_dic)


