import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Nadam

# Libs
import time
import warnings

# Configs
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

## Funções
def matriz(num_colunas, array1):
    """
    Gera uma matriz sequencial a partir de um array, com o número de colunas especificado.

    Args:
        array (list ou np.ndarray): Array de entrada.
        num_colunas (int): Número de colunas desejado na matriz.

    Returns:
        np.ndarray: Matriz sequencial.
    """
    if num_colunas > len(array1):
        raise ValueError("O número de colunas não pode ser maior que o tamanho do array.")

    # Número de linhas na matriz
    num_linhas = len(array1) - num_colunas + 1

    # Criando a matriz sequencial
    matriz = np.array([array1[i:i + num_colunas] for i in range(num_linhas)])
    return matriz

def lista_predicao(t, modelos, array1):
    """
    Gera uma lista com possíveis predições.
    Args:
        t (int): Quantidade de modelos contidos na lista original.
        modelos (np.array): Array que contém modelos treinados.
        array1 (np.array): Lista contendo os últimos valores.
    Returns:
        np.array: Array que contém a predição de cada modelo da lista original.
    """
    y_pred1 = []
    for sk in range(0,t):
        if modelos[sk] is not None:
            posicao = 60*sk + 60
            print(sk, posicao)
            matriz1s = matriz(posicao,array1)

            x_new = np.array(matriz1s[-1,1:])
            x_new = x_new.astype("float32")
            x_new = np.expand_dims(x_new, -1)
            x_new = np.reshape(x_new, (-1, ((matriz1s.shape[1])-1), 1, 1))

            predictions = modelos[sk].predict(x_new)

            y_pred = np.argmax(predictions, axis=1)
            y_pred1.append(y_pred[0])
    return y_pred1

def reden(array1, array3, m, n):
    """
    Função para treinar uma rede neural usando as entradas e saídas fornecidas.

    Args:
        array1 (numpy.array): Saídas (rótulos) binárias (0 ou 1).
        array3 (numpy.array): Entradas preditoras.
        m (int): Número de amostras.
        n (int): Número de características por amostra.

    Returns:
        keras.Model: Modelo treinado.
    """
    # Dividindo os dados em treino e teste
    X = np.array(array3)
    y = np.array(array1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Ajustando dimensões para entrada no modelo
    x_train = np.expand_dims(x_train, -1).astype("float32")
    x_test = np.expand_dims(x_test, -1).astype("float32")
    input_shape = (n - 1, 1)  # Formato esperado de entrada

    # Convertendo saídas para categóricas
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Definição do modelo
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(264, activation="relu", use_bias=True),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu", use_bias=True),
        layers.Dense(64, activation="relu", use_bias=True),



        #layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    #model.layers[-1].bias.assign([1]*64),
    model.compile(
        loss="binary_crossentropy",
        #optimizer="adam",
        optimizer=Nadam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999),
        metrics=['accuracy', Precision(name="precision"), Recall(name="recall")]
    )

    # Treinamento
    batch_size = 2**10
    epochs = 50
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2
    )

    # Avaliação
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")
    print(f"Precision: {score[2]:.4f}")
    print(f"Recall: {score[3]:.4f}")

    return model

def ponderar_lista(lista, base=2):
    """
    Realiza uma ponderação dos elementos da lista com pesos exponenciais crescentes.

    Args:
        lista (list): Lista de inteiros contendo apenas 0 e 1.
        base (float): Base da função exponencial. Deve ser maior que 1.

    Returns:
        int: Resultado ponderado, 0 ou 1.
    """
    n = len(lista)
    if n == 0:
        raise ValueError("A lista não pode estar vazia.")

    # Calcular pesos exponenciais
    pesos = [base ** i for i in range(n)]

    # Calcular soma ponderada e total de pesos
    soma_ponderada = sum(elemento * peso for elemento, peso in zip(lista, pesos))
    total_pesos = sum(pesos)

    # Retornar 1 se média ponderada >= 0.5, senão 0
    return 1 if soma_ponderada / total_pesos >= 0.5 else 0


## Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
data = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j, j1, j2, j3 = 0,0,0,0,0,0

media_parray, acerto01 = [], []

# Inicializar classes
lautgh1 = np.zeros(60, dtype = int)
lautgh2 = np.zeros(60, dtype = int)

acerto, acerto1, acerto2, acerto3, core = 0,0,0,0,0
precisao, acuracia = 0,0

modelos = [None]*5000
recurso1, recurso2 = [None]*5000, [None]*5000
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
        odd = float(data['Odd'][i])
        #odd = float(data['Entrada'][i].replace(",",'.'))
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


    if i >= 301:
        print(24*"-'-")
        name = resultado

        if name == 1:
            if odd >= 2:
                count = 1
                if count == name:
                    acerto = acerto + 1
                    acerto1 = acerto1 + 1
                    j = j + 1
                    j1 = j1 + 1
            else:
                j = j + 1
                j1 = j1 + 1
        else:
            if odd < 2:
                count = 0
                if count == name:
                    acerto1 = acerto1 + 1
                    j1 = j1 + 1
            else:
                j1 = j1 + 1

        if j == 0 or j1 == 0:
            if j == 0:
                acuracia = 0
            if j1 == 0:
                precisao = 0
        else:
            acuracia = (acerto / j) * 100
            precisao = (acerto1 / j1) * 100



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
        if core1 == 0:
            core11 = 60
        else:
            core11 = core1
        print(f'Acuracia modelo Geral: {round(acuracia,4)} | Acuracia_{core11}: {round(medida_pontual,4)} \nPrecisao modelo Geral: {round(precisao,4)}')
        print(24*"-'-")

    if i >= 300 and (i % 60) == 0:
        print('***'*20)
        print(f'Carregando dados ...')
        info = []
        cronor = (i + 600) // 5
        lista = [name for name in range(60, cronor, 60)]
        for click in lista:
            k0 = i % click
            if k0 == 0:
                info.append(click)
        print(f'{12*"*-"} \nPosições que devem ser carregadas: {info} \n{12*"*-"}')
        for click in info:
            print(f'Treinamento para {click}')
            matrix1s, matrix1n = matriz(click, array2s), matriz(click, array2n)
            posicao0 = int((click / 60) - 1)
            array1, array3 = matrix1n[:,-1], matrix1s[:,:-1]
            m, n = matrix1n.shape
            print(f'Matrix_{click}: {[matrix1n.shape, matrix1s.shape]} | Indice: {matrix1n.shape[1]} | Posicao: {posicao0}')
            models = reden(array1,array3, m, n)
            modelos[posicao0] = models
            print(f'Treinamento {click} realizado com sucesso ...  \n')
        print('***'*20)


    if i >= 300:
        y_pred1 = lista_predicao(len(modelos), modelos, array2s)
        resultado = ponderar_lista(y_pred1)
        print(24*'*-')
        print(f'Proxima Entrada: {resultado}')
        print(24*'*-')


    i += 1


