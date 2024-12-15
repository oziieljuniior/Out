import pandas as pd
import numpy as np
import os
import json
import random
import matplotlib.pyplot as mt
import matplotlib.pyplot as plt
from scipy.stats import logistic
from scipy.special import erfinv
import math
from scipy.special import beta as beta_func
from scipy.special import betaincinv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.metrics import Precision, Recall

import statistics as stt
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, classification_report, mean_absolute_error, mean_squared_error, ConfusionMatrixDisplay
from scipy.stats import pearsonr
from tensorflow import keras
from tensorflow.keras import layers

# Libs
import warnings

import seaborn as sns
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import time

#data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
#data2 = pd.read_csv("/home/darkcover/Documentos/Out/dados/odds_200k_1.csv")

array1, array2, array3 = [], [], []

# Configs
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

data = pd.read_csv('/home/darkcover/Documentos/Out/dados/Saidas/FUNCOES/DOUBLE - 17_09_s1.csv')
a1 = 0
i = 0
j = 0
acerto = 0
inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

while i <= 210000:
    print(f'Número da Entrada - {i}')
    if i <= inteiro:
        odd = float(data['Entrada'][i].replace(",",'.'))
        #odd = float(data['Entrada'][i])
        if odd == 0:
            odd = 1
    else:
        odd = float(input("Entrada -> ").replace(",",'.'))
    
    print(f'Entrada: {odd}')

    if odd == 0:
        break
    array2.append(odd)
    
    if i <= inteiro:
    
        if i >= 141:
            array3.append(array2[-30:])
            if float(data['Entrada'][i + 1].replace(",",'.')) >= 2:
            #if float(data['Entrada'][i + 1]) >= 2:
                array1.append([1])
            else:
                array1.append([0])
            
        if i >= 272:
            for name in predicted_classes:
                if name == 1:
                    if odd >= 2:
                        count = 1
                        if count == name:
                            acerto = acerto + 1
                            j = j + 1
                    else:
                        j = j + 1
            if j == 0:
                acuracia = 0
            else:
                acuracia = (acerto / j) * 100
            print(24*'-')
            print(f'Acuracia modelo: {acuracia}')
            print(24*'-')
            


        #print(len(array2))
        if i % 30 == 0 and i >= 270:
            j, acerto=0,0
            # Extraindo as 60 primeiras entradas de cada sublista e salvando no array 'X'
            X = np.array(array3)  # Pegue todas as colunas, exceto a última
            # Extraindo a última entrada de cada sublista e salvando no array 'y'
            y = np.array(array1)  # Pegue a última coluna de cada sublista (última entrada)
            # Verificando as formas dos arrays
            print("Shape de X (entradas):", X.shape)  # Deve ser algo como (n_amostras, 60)
            print("Shape de y (saídas):", y.shape)    # Deve ser algo como (n_amostras,)

            #print(X)
            #print(y)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(X)
            # Model / data parameters
            num_classes = 2
            input_shape = (30, 1, 1) #verifique

            # Load the data and split it between train and test sets
            # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

            # Scale images to the [0, 1] range
            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")
            # Make sure images have shape (28, 28, 1)
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
            print("x_train shape:", x_train.shape)
            print(x_train.shape[0], "train samples")
            print(x_test.shape[0], "test samples")


            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            model = keras.Sequential(
                [
                    keras.Input(shape=input_shape),
                    #layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    #layers.MaxPooling2D(pool_size=(2, 2)),
                    #layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    #layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dropout(0.6),
                    layers.Dense(num_classes, activation="relu"),
                    layers.Dropout(0.55),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

            print(model.summary())
            batch_size = 264
            epochs = 30
            class_weights = {0: 1., 1: 3.}  # Ajuste de acordo com a distribuição das classes
            model.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=['accuracy', Precision(), Recall()])
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, class_weight=class_weights)
            score = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])
        
        if i >= 271:
            x_new = np.array(array3[-1])
            x_new = x_new.astype("float32")
            x_new = np.expand_dims(x_new, -1)
            x_new = np.reshape(x_new, (-1, 30, 1, 1))
            #print(x_new)
            predictions = model.predict(x_new)

            predicted_classes = np.argmax(predictions, axis=1)
            print(f'Proxima entrada: {predicted_classes}')
            print(24*'*-')
            #time.sleep(0.5)

    else:
        #Organizar sem os dados futuros
        print(24*'*-')
        if i >= 141:
            if a1 == 0:
                a1 = a1 + 1
                
            if a1 >= 1:
                array4 = array2[-31:]
                array_ajuste1 = array4[:30]

                array3.append(array_ajuste1)
                #organizar
            
                if array4[-1] >= 2:
                    array1.append([1])
                else:
                    array1.append([0])
            
        if i >= 272:
            for name in predicted_classes:
                if name == 1:
                    if odd >= 2:
                        count = 1
                        if count == name:
                            acerto = acerto + 1
                            j = j + 1
                    else:
                        j = j + 1
            if j == 0:
                acuracia = 0
            else:
                acuracia = (acerto / j) * 100
            print(24*'-')
            print(f'Acuracia modelo: {acuracia}')
            print(24*'-')
            


        #print(len(array2))
        if i % 30 == 0 and i >= 270:
            j, acerto=0,0
            # Extraindo as 60 primeiras entradas de cada sublista e salvando no array 'X'
            X = np.array(array3)  # Pegue todas as colunas, exceto a última
            # Extraindo a última entrada de cada sublista e salvando no array 'y'
            y = np.array(array1)  # Pegue a última coluna de cada sublista (última entrada)
            # Verificando as formas dos arrays
            print("Shape de X (entradas):", X.shape)  # Deve ser algo como (n_amostras, 60)
            print("Shape de y (saídas):", y.shape)    # Deve ser algo como (n_amostras,)

            #print(X)
            #print(y)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(X)
            # Model / data parameters
            num_classes = 2
            input_shape = (30, 1, 1) #verifique

            # Load the data and split it between train and test sets
            # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

            # Scale images to the [0, 1] range
            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")
            # Make sure images have shape (28, 28, 1)
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
            print("x_train shape:", x_train.shape)
            print(x_train.shape[0], "train samples")
            print(x_test.shape[0], "test samples")


            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
            model = keras.Sequential(
                [
                    keras.Input(shape=input_shape),
                    #layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    #layers.MaxPooling2D(pool_size=(2, 2)),
                    #layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    #layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dropout(0.60),
                    layers.Dense(num_classes, activation="relu"),
                    layers.Dropout(0.55),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

            print(model.summary())
            batch_size = 264
            epochs = 30
            class_weights = {0: 1., 1: 3.}  # Ajuste de acordo com a distribuição das classes
            model.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=['accuracy', Precision(), Recall()])
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, class_weight=class_weights)
            score = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])
        
        if i >= 271:
            x_new = np.array(array2[-30:])
            x_new = x_new.astype("float32")
            x_new = np.expand_dims(x_new, -1)
            x_new = np.reshape(x_new, (-1, 30, 1, 1))
            #print(x_new)
            predictions = model.predict(x_new)

            predicted_classes = np.argmax(predictions, axis=1)
            print(f'Proxima entrada: {predicted_classes}')
            print(24*'*-')
            #time.sleep(0.25)
    i += 1 

print(array2)