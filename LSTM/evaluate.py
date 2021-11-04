from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.layers import Layer
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM,Embedding,Input
from keras.layers.core import Dropout
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
from keras.optimizers import RMSprop, Adam
from sklearn.utils import class_weight
import keras.backend as K
from keras import Model
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,roc_auc_score
from smiles_vectorizer import SmilesEnumerator, SmilesIterator

model = keras.models.load_model('attention_lstm')


sme = SmilesEnumerator()
dataset = pd.read_csv('../AID_CSV/1332.csv')
X = dataset['SMILES']
y = dataset['Activity']
X_train,  X_test, y_train, y_test = tts(X,y,test_size = 0.3,random_state=42)
sme.fit(X)
sme.leftpad = True
print(model.summary())
precisions = []
recalls = []
f1 = []
accuracies = []
roc  = []


for i in range(0,100):
    

    try:
        trans_X_test = sme.transform(X_test)
        predictions = model.predict_classes(trans_X_test)
        predictions =(model.predict(trans_X_test) > 0.5).astype("int32")
        #print(predictions)
        #print(model.predict(trans_X_test))
        #break
        precisions.append(precision_score(y_test,predictions))
        recalls.append(recall_score(y_test,predictions))
        f1.append(f1_score(y_test,predictions))
        roc.append(roc_auc_score(y_test,predictions))
        accuracies.append(accuracy_score(y_test,predictions))
    except:
        continue    

print(np.median(precisions))
print(np.median(recalls))
print(np.median(f1))
print(np.median(roc))
print(np.median(accuracies))
'''
trans_X_test = sme.transform(X_test)
predictions = model.predict_classes(trans_X_test)
predictions = predictions[0]
precisions.append(precision_score(y_test,predictions))
recalls.append(recall_score(y_test,predictions))
f1.append(f1_score(y_test,predictions))
roc.append(roc_auc_score(y_test,predictions))
accuracies.append(accuracy_score(y_test,predictions))

print(precisions[0])
print(recalls[0])
print(f1[0])
print(roc[0])
print(accuracies[0])
'''