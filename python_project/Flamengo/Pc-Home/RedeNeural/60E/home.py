import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from tensorflow import keras
from tensorflow.keras import layers

# Libs
import warnings

import time


# Libs
import warnings

# Configs
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

import time 


## Carregar data
data = pd.read_csv('/home/darkcover/Documentos/Data/Out/Entrada.csv')

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j = 0,0,0

media_parray, acerto01 = [], []

# Inicializar classes
lautgh1 = np.zeros(60, dtype = int)
lautgh2 = np.zeros(60, dtype = int)

acerto, core = 0,0

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

while i <= 210000:
    print(24*'---')
    #print(len(media_parray))
    if len(media_parray) < 59:
        m = 0
    else:
        m = media_parray[len(media_parray) - 60]

    print(f'Número da Entrada - {i} | Acuracia_{core + 1}: {round(m,4)}')
    if i <= inteiro:
        odd = float(data['Entrada'][i].replace(",",'.'))
        #odd = float(data['Entrada'][i])
        if odd == 0:
            odd = 1
        print(f'Entrada: {odd}')
    else:
        odd = float(input("Entrada -> ").replace(",",'.'))

    if odd == 0:
        break
    
    array2s.append(odd)
    if odd >= 2:
        corte2 = 1
    else:
        corte2 = 0

    array2n.append(corte2)

    if i <= inteiro:
        print('**'*20)
        if i >= 60:
            track = i - 60
            corte1 = array2s[track]
            corte3 = array2n[track]
            
            array3s.append(corte1)
            array3n.append(corte3)
        
            print(f'Order1: {corte1} | Order2: {corte3} | LArrayI: {[len(array3n), len(array3s)]}')
        
            if (i % 60) == 0 and i > 60:
                print('*-'*16)
                print(f'LArrayII: {[len(array3n[i - 120: i - 60]), len(array3s[i - 120: i - 60])]}')
                if i == 120:
                    matrix1s = np.array(array3s[i - 120: i - 60]).reshape(-1,1)
                    matrix1n = np.array(array3n[i - 120: i - 60]).reshape(-1,1)
            
                else:
                    array3ss = np.array(array3s[i - 120: i - 60]).reshape(-1,1)
                    array3ns = np.array(array3n[i - 120: i - 60]).reshape(-1,1)

                    matrix1s = np.hstack([matrix1s,array3ss])
                    matrix1n = np.hstack([matrix1n,array3ns]) 
                    
                #print(matrix1n)
                print(f'Order3: {i} | LArrayIII: {[len(array3n), len(array3s)]} | MatrixS: {[matrix1n.shape, matrix1s.shape]}')
                print('*-'*16)
        print('**'*20)
        
        if i >= 301:
            print(24*"-'-")
            
            name = y_pred[0]
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
            
            if np.sum(y_pred[0]) == 1:
                if odd >= 2:
                    lautgh1[core] = lautgh1[core] + 1
                    lautgh2[core] = lautgh2[core] + 1
                    medida_pontual = lautgh2[core] / lautgh1[core]
                else:
                    lautgh1[core] = lautgh1[core] + 1
                    lautgh2[core] = lautgh2[core]
                    medida_pontual = lautgh2[core] / lautgh1[core]
            else:
                if len(media_parray)<59:
                    medida_pontual = 0
                else:
                    medida_pontual = media_parray[len(media_parray) - 60]

            media_parray.append(medida_pontual)
            print(f'Acuracia modelo Geral: {round(acuracia,4)} | Acuracia_{core + 1}: {round(medida_pontual,4)}')
            print(24*"-'-")

        if i >= 300 and i % 60 == 0:
            print('/-/'*16)
            print(f'Treinamento necessario ...')
            print(f'MatrixS: {[matrix1n.shape, matrix1s.shape]} | Indice: {matrix1n.shape[1]}')
            m,n = matrix1n.shape

            array3 = matrix1s[:,:-1]
            array1 = matrix1n[:,-1]
            #j, acerto=0,0
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
            input_shape = ((n-1), 1, 1) #verifique

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
            class_weights = {0: 1., 1: 1.}  # Ajuste de acordo com a distribuição das classes
            model.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=['accuracy', Precision(), Recall()])
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, class_weight=class_weights)
            score = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])
            print(f'Treinamento Realizado com Sucesso ...')
            print('/-/'*16)

        if i >= 300:
            core = i % 60
            if core == 59:
                x_new = np.array(matrix1s[0,1:])
            
                x_new = x_new.astype("float32")
                x_new = np.expand_dims(x_new, -1)
                x_new = np.reshape(x_new, (-1, (n-1), 1, 1))
                #print(x_new)
                predictions = model.predict(x_new)

                y_pred = np.argmax(predictions, axis=1)
                print(f'Proxima entrada: {y_pred[0]}')
                print(24*'*-')
                #time.sleep(0.5)
            else:
                x_new = np.array(matrix1s[core,1:])
            
                x_new = x_new.astype("float32")
                x_new = np.expand_dims(x_new, -1)
                x_new = np.reshape(x_new, (-1, (n-1), 1, 1))
                #print(x_new)
                predictions = model.predict(x_new)

                y_pred = np.argmax(predictions, axis=1)
                print(f'Proxima entrada: {y_pred[0]}')
                print(24*'*-')
                #time.sleep(0.5)            
            
            
    else:
        print('**'*20)
        if i >= 60:
            track = i - 60
            corte1 = array2s[track]
            corte3 = array2n[track]
            
            array3s.append(corte1)
            array3n.append(corte3)
        
            print(f'Order1: {corte1} | Order2: {corte3} | LArrayI: {[len(array3n), len(array3s)]}')
        
            if (i % 60) == 0 and i > 60:
                print('*-'*16)
                print(f'LArrayII: {[len(array3n[i - 120: i - 60]), len(array3s[i - 120: i - 60])]}')
                if i == 120:
                    matrix1s = np.array(array3s[i - 120: i - 60]).reshape(-1,1)
                    matrix1n = np.array(array3n[i - 120: i - 60]).reshape(-1,1)
            
                else:
                    array3ss = np.array(array3s[i - 120: i - 60]).reshape(-1,1)
                    array3ns = np.array(array3n[i - 120: i - 60]).reshape(-1,1)

                    matrix1s = np.hstack([matrix1s,array3ss])
                    matrix1n = np.hstack([matrix1n,array3ns]) 
                    
                #print(matrix1n)
                print(f'Order3: {i} | LArrayIII: {[len(array3n), len(array3s)]} | MatrixS: {[matrix1n.shape, matrix1s.shape]}')
                print('*-'*20)
        
        if i >= 301:
            print(24*"-'-")
            name = y_pred[0]
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

            if np.sum(y_pred[0]) == 1:
                if odd >= 2:
                    lautgh1[core] = lautgh1[core] + 1
                    lautgh2[core] = lautgh2[core] + 1
                    medida_pontual = lautgh2[core] / lautgh1[core]
                else:
                    lautgh1[core] = lautgh1[core] + 1
                    lautgh2[core] = lautgh2[core]
                    medida_pontual = lautgh2[core] / lautgh1[core]
            else:
                if len(media_parray)<59:
                    medida_pontual = 0
                else:
                    medida_pontual = media_parray[len(media_parray) - 60]
            

            media_parray.append(medida_pontual)
            print(f'Acuracia modelo Geral: {round(acuracia,4)} | Acuracia_{core + 1}: {round(medida_pontual,4)}')
            print(24*"-'-")

        if i >= 300 and i % 60 == 0:
            print('/-/'*16)
            print(f'Treinamento necessario ...')
            print(f'MatrixS: {[matrix1n.shape, matrix1s.shape]} | Indice: {matrix1n.shape[1]}')
            m,n = matrix1n.shape

            array3 = matrix1s[:,:-1]
            array1 = matrix1n[:,-1]
            #j, acerto=0,0
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
            input_shape = ((n-1), 1, 1) #verifique

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
            class_weights = {0: 1., 1: 1.}  # Ajuste de acordo com a distribuição das classes
            model.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=['accuracy', Precision(), Recall()])
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, class_weight=class_weights)
            score = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])
            print(f'Treinamento Realizado com Sucesso ...')
            print('/-/'*16)

        if i >= 300:
            core = i % 60
            if core == 59:
                x_new = np.array(matrix1s[0,1:])
            
                x_new = x_new.astype("float32")
                x_new = np.expand_dims(x_new, -1)
                x_new = np.reshape(x_new, (-1, (n-1), 1, 1))
                #print(x_new)
                predictions = model.predict(x_new)

                y_pred = np.argmax(predictions, axis=1)
                print(f'Proxima entrada: {y_pred[0]}')
                print(24*'*-')
                #time.sleep(0.5)
            else:
                x_new = np.array(matrix1s[core,1:])
            
                x_new = x_new.astype("float32")
                x_new = np.expand_dims(x_new, -1)
                x_new = np.reshape(x_new, (-1, (n-1), 1, 1))
                #print(x_new)
                predictions = model.predict(x_new)

                y_pred = np.argmax(predictions, axis=1)
                print(f'Proxima entrada: {y_pred[0]}')
                print(24*'*-')
                #time.sleep(0.5)            
            
            
    
    i += 1            
            
