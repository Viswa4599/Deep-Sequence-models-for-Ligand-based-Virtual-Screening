from sklearn.model_selection import train_test_split as tts
import glob
from tqdm import tqdm
import datetime
import pickle as pkl
import pandas as pd  
import numpy as np
import argparse
from keras.layers import Layer
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM,Embedding,Input, GRU, Bidirectional,Concatenate
from keras.layers.core import Dropout
from keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras.metrics import Precision, Recall, AUC, Accuracy
from keras.optimizers import RMSprop, Adam
from sklearn.utils import class_weight
import keras.backend as k
from keras import Model
import keras
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,roc_auc_score
from smiles_vectorizer import SmilesEnumerator, SmilesIterator
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

#Diffrent attention class. Returns a context vector. Can append other SDF features also to this context vector
class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
          
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
          
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

all_files = glob.glob("../AID_CSV/*.csv")

parser = argparse.ArgumentParser(description = "LSTM program for Virtual Screening")
parser.add_argument("-l", "--lists", nargs='+', default = all_files, help = "List of AID files to test on")
parser.add_argument("-i", "--ignore", nargs='+', default = None, help = "List of AID files to ignore testing on")
parser.add_argument("-v", "--verbose", action = "store_true", help = "List the files that will be tested on")

args = parser.parse_args()
files_list = args.lists
ignore_list = args.ignore

if ignore_list is not None:
    ignore_list = ["../AID_CSV/" + x for x in ignore_list]
else:
    ignore_list = []

if args.lists is not all_files:
    files_list = ["../AID_CSV/" + x for x in files_list]

if ignore_list != []:
    files_list.remove(*ignore_list)

if args.verbose:
    print(f"Files List : \t{files_list}\nIgnore List : \t{ignore_list}")


for file_path in files_list:

    start = datetime.datetime.now()
    print(f"Working on {file_path}")
    #ry:
    sme = SmilesEnumerator()
    dataset = pd.read_csv(file_path)
    X = dataset['SMILES']
    y = dataset['Activity']

    X_train,  X_test, y_train, y_test = tts(X,y,test_size = 0.2, random_state=42, stratify = y)   
    sme.fit(X, extra_chars = ["9","%","0"])
    sme.leftpad = True
    print(sme.charset)
    print(sme.pad)

    print('making generator')
    generator = SmilesIterator(X_train, y_train, sme, batch_size=256, dtype=K.floatx(), shuffle = True)
    X,y = generator.next()
    print(X.shape)
    print (y.shape)
    input_shape = X.shape[1:]
    embed_dim = 200
    output_shape = 1
    print("Input Shape: ", input_shape)
    
    RNN_CELL_SIZE = 128
    sequence_input = Input(shape = input_shape)
    lstm = LSTM(RNN_CELL_SIZE, return_sequences = True)(sequence_input)

    # Getting our LSTM outputs
    (lstm, forward_h, forward_c) = LSTM(RNN_CELL_SIZE, return_sequences=True, return_state=True)(lstm)

    context_vector, attention_weights = Attention(50)(lstm, forward_h)
    dense1 = Dense(64, activation="relu")(context_vector)
    dropout = Dropout(0.4)(dense1)
    dense1 = Dense(32, activation="relu")(context_vector)
    dropout = Dropout(0.4)(dense1)
    dense1 = Dense(16, activation="relu")(dropout)
    dropout = Dropout(0.4)(dense1)
    output = Dense(1, activation="sigmoid")(dropout)

    fnmodel = Model(inputs=sequence_input, outputs=output)
    print(fnmodel.summary())
    
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                    patience=10, min_lr=0.00001,epsilon = 1e-04,verbose =1)
    es = EarlyStopping(monitor='val_loss', mode = 'max', patience = 15,restore_best_weights = True)
    fnmodel.compile(loss="binary_crossentropy", optimizer='adam',metrics = [AUC(), Precision(), Recall()])
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
    
    weights = {0:class_weights[0],1:class_weights[1]}
    filepath = "attention_lstm_weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_precision', verbose=1, save_best_only=True, mode='max',save_weights_only = True)
    fnmodel.fit_generator(generator, steps_per_epoch=100, epochs=100,callbacks = [reduce_lr])
    
    #model.save('./model_saves/attention_lstm' + file_path.split("/")[-1])
    #fnmodel.load_weights("attention_lstm_weights.best.hdf5")
    #fnmodel.compile(loss="binary_crossentropy", optimizer='adam',metrics = [AUC(), Precision(), Recall()])
    print("Created model and loaded weights from file")

    precisions = []
    recalls = []
    f1 = []
    accuracies = []
    roc  = []
    ensemble_flag = 1
    preds_list = []

    if ensemble_flag == 1:
        filename = 'test_proba.pkl'
        fileObject = open(filename, 'wb')
        trans_X_train = sme.transform(X_train)
        predictions = fnmodel.predict(trans_X_train)
        print(predictions)
        pkl.dump(predictions,fileObject)

    else:
        for i in tqdm(range(0,100), desc = "Testing " + file_path.split("/")[-1], unit = "test", leave = False):
            try:
                trans_X_test = sme.transform(X_test)
                predictions =(fnmodel.predict(trans_X_test) > 0.5).astype("int32")
                preds_list.append(predictions)
                precisions.append(precision_score(y_test,predictions))
                recalls.append(recall_score(y_test,predictions))
                f1.append(f1_score(y_test,predictions))
                roc.append(roc_auc_score(y_test,predictions))
                accuracies.append(accuracy_score(y_test,predictions))
            except Exception as E:
                print(f"E on {file_path}")

        #Voting Predictions
        final_predictions = []
        for i in range(0,len(y_test)):
            count0 = 0
            count1 = 1
            for preds in preds_list:
                if preds[i] ==0:
                    count0+=1
                else:
                    count1+=1
            if count0 > count1:
                final_predictions.append(0)
            else:
                final_predictions.append(1)

        voted_precisions = precision_score(y_test,predictions)
        voted_recalls = recall_score(y_test,predictions)
        voted_f1 = f1_score(y_test,predictions)
        voted_roc = roc_auc_score(y_test,predictions)
        voted_accuracies = accuracy_score(y_test,predictions)
        end = datetime.datetime.now()
        res_string = f"(Medians) File Name : {file_path}\t{end-start}\nMPrecision : {np.median(precisions)}\tMRecall : {np.median(recalls)}\nMF1 : {np.median(f1)}\tMROC : {np.median(roc)}\tMAccuracy : {np.median(accuracies)}\n\n" 
        max_res_string = f"(Max) File Name : {file_path}\t{end-start}\nMPrecision : {np.max(precisions)}\tMRecall : {np.max(recalls)}\nMF1 : {np.max(f1)}\tMROC : {np.max(roc)}\tMAccuracy : {np.max(accuracies)}\n\n" 
        voted_res_string =  f"(Voted) File Name : {file_path}\t{end-start}\nPrecision : {voted_precisions}\tRecall : {voted_recalls}\nF1 : {voted_f1}\tMROC : {voted_roc}\tAccuracy : {voted_accuracies}\n\n" 
        print(res_string)
        print(max_res_string)
        print(voted_res_string)

        with open("report_0.2.txt", "a") as fp:
            fp.write(res_string)
            fp.write(max_res_string)
            fp.write(voted_res_string)
    '''
    except KeyboardInterrupt:
        print("\nProgram execution terminated")
    except Exception as e:
        print(e)
        print(f"Error while working on {file_path}. Please check log files")

        with open("error_logs.txt", "a") as fe:
            fe.write(f"File Name : {file_path}\t{datetime.datetime.now()}\n{e}\n\n")
    '''