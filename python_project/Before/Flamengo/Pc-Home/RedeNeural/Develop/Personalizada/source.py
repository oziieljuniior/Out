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

## Funções
def matriz(i0, array1, array2):
    lista = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
    final1, final2, info = [],[],[]
    
    for name in lista:
        order = i0 % name
        if order == 0:
            info.append(name)
        
    print(f'Novas colunas para: {info} ...')
    info1 = []
    for name in info:
        m0, m1 = len(array1), len(array2)
        order1 = m1 // name
        
        print(name, order1, m0, m1)
        
        if order1 >= 5:
            matriz1 = np.array(array1).reshape(-1, name).T
            matriz2 = np.array(array2).reshape(-1, name).T
            info1.append(matriz1.shape[0])
            print(f'Order3: {i} | MatrixS: {[matriz1.shape, matriz2.shape]}')
            final1.append(matriz1), final2.append(matriz2)
    return final1, final2, info1

def reden(array1, array3, m, n):
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
    return model
        
def ponderar_lista(lista):
    """
    Realiza uma ponderação dos elementos da lista, atribuindo o maior peso à primeira coluna,
    o segundo maior peso à última coluna, e pesos crescentes às colunas intermediárias.

    Args:
        lista (list): Lista de inteiros contendo apenas 0 e 1.

    Returns:
        int: Resultado ponderado, 0 ou 1.
    """
    n = len(lista)
    if n == 0:
        raise ValueError("A lista não pode estar vazia.")

    # Configurar pesos
    if n == 1:
        pesos = [1]  # Caso especial de lista com um único elemento
    else:
        maior_peso = n  # Maior peso para a primeira coluna
        segundo_maior_peso = n - 1  # Segundo maior peso para a última coluna
        
        # Pesos intermediários crescentes
        pesos_intermediarios = list(range(2, segundo_maior_peso + 1))
        
        # Compor a lista completa de pesos
        pesos = [maior_peso] + pesos_intermediarios + [segundo_maior_peso]

    # Calcular soma ponderada e total de pesos
    soma_ponderada = sum(elemento * peso for elemento, peso in zip(lista, pesos))
    total_pesos = sum(pesos)

    # Determinar o valor ponderado
    resultado = soma_ponderada / total_pesos

    # Retornar 1 se média ponderada >= 0.5, senão 0
    return 1 if resultado >= 0.6 else 0



## Carregar data
data = pd.read_csv('/home/darkcover/Documentos/Data/Out/Entrada.csv')

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j = 0,0,0

media_parray, acerto01 = [], []

# Inicializar classes
lautgh1 = np.zeros(60, dtype = int)
lautgh2 = np.zeros(60, dtype = int)

acerto, core = 0,0

modelos = [None]*10
recurso1, recurso2 = [None]*10, [None]*10
inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

while i <= 210000:
    print(24*'---')
    #print(len(media_parray))
    if len(media_parray) < 59:
        m = 0
        core1 = 0
    else:
        m = media_parray[len(media_parray) - 60]

    print(f'Número da Entrada - {i} | Acuracia_{core1 + 1}: {round(m,4)}')
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

        
    if i >= 601:
        print(24*"-'-")
        
        name = resultado
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
        
        core1 = i % 60
        if resultado == 1:
            if odd >= 2:
                lautgh1[core1] = lautgh1[core1] + 1
                lautgh2[core1] = lautgh2[core1] + 1
                medida_pontual = lautgh2[core1] / lautgh1[core1]
            else:
                lautgh1[core1] = lautgh1[core1] + 1
                lautgh2[core1] = lautgh2[core1]
                medida_pontual = lautgh2[core1] / lautgh1[core1]
        else:
            if len(media_parray)<59:
                medida_pontual = 0
            else:
                medida_pontual = media_parray[len(media_parray) - 60]

        media_parray.append(medida_pontual)
        print(f'Acuracia modelo Geral: {round(acuracia,4)} | Acuracia_{core1}: {round(medida_pontual,4)}')
        print(24*"-'-")

    if i >= 600 and (i % 60) == 0:
        print('**'*20)
        print(f'Carregando dados ...')
        retorno1, retorno2, info = matriz(i,array2n[1:],array2s[1:])
        print(f'Posições que devem ser carregadas: {info}')
        
        print('**'*20)
        i0 = 0
        for name in range(0, len(info)):
            print('/-/'*16)
            print(f'Treinamento necessario para: {info[name]}')
            source1 = (info[name] // 60) - 1
            matrix1n, matrix1s = np.array(retorno1[i0]), np.array(retorno2[i0])
            print(f'MatrixS: {[matrix1n.shape, matrix1s.shape]} | Indice: {matrix1n.shape[1]}')
            m,n = matrix1n.shape
            array3 = matrix1s[:,:-1]
            array1 = matrix1n[:,-1]
            modelss = reden(array1,array3,m,n)

            modelos[source1] = modelss
            recurso1[source1], recurso2[source1] = matrix1n, matrix1s
            
            print(f'Treinamento {info[name]} realizado com sucesso ...')
            print('/-/'*16)
            i0 += 1
    
    if i >= 600:
        y_pred1 = []
        for sk in range(0,len(modelos)):
            if modelos[sk] is not None:
                posicao = 60*sk + 60
                print(sk, posicao)
                
                core = i % posicao
                e = recurso2[sk]
                nn = e.shape[1]
                mm = modelos[sk]
                print(nn, mm)
                if core == (posicao - 1):
                    x_new = np.array(e[0,1:])
                
                    x_new = x_new.astype("float32")
                    x_new = np.expand_dims(x_new, -1)
                    x_new = np.reshape(x_new, (-1, (nn-1), 1, 1))
                    #print(x_new)
                    predictions = mm.predict(x_new)

                    y_pred = np.argmax(predictions, axis=1)
                    #print(f'Proxima entrada: {y_pred[0]}')
                    #print(24*'*-')
                    #time.sleep(0.5)
                else:
                    x_new = np.array(e[core,1:])
                
                    x_new = x_new.astype("float32")
                    x_new = np.expand_dims(x_new, -1)
                    x_new = np.reshape(x_new, (-1, (nn-1), 1, 1))
                    #print(x_new)
                    predictions = mm.predict(x_new)

                    y_pred = np.argmax(predictions, axis=1)
                    #print(f'Proxima entrada: {y_pred[0]}')
                    #print(24*'*-')
                    #time.sleep(0.5)
                y_pred1.append(y_pred[0])
        resultado = ponderar_lista(y_pred1)
        print(24*'*-')
        print(f'Proxima Entrada: {resultado}')
        print(24*'*-')

                                    
    i += 1            
    print(24*"-'-")
        
            
