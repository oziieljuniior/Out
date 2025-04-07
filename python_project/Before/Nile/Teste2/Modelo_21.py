import pandas as pd
import numpy as np

from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import to_categorical

import time


def coletarodd(i, inteiro, data, array2s, array2n):
    """
    Função que coleta e organiza as entradas iniciais do banco de dados.
    Args:
        i (int): Valor inteiro não-negativo. Entrada que controla o loop principal. É um valor cumulativo.
        inteiro (int): Valor inteiro não-negativo. Entrada que determina até aonde os dados devem ser carregados automaticamente, através de um banco de dados.
        data (pd.DataFrame): Variável carregada inicialmente para treinamento/desenvolvimento. Do tipo data frame.   #FIXWARNING2
        array2s (np.array): Array cumulativo que carrega as entradas reais com duas casas decimais.
        array2n (np.array): Array cumulativo que carrega as entredas inteiras(0 ou 1).
    Returns:
        np.array: Array cumulativo que carrega as entradas reais com duas casas decimais.
        np.array: Array cumulativo que carrega as entredas inteiras(0 ou 1).
        float: Valor real com duas casas decimais. Ele é determinado pela entrada dos dados, ou usuário.
    """

#FIXWARNING1: O formato da data de entrada pode ser mudado? Atualmente está em .csv

    if i <= inteiro:
        odd = float(data['Odd'][i])
        #odd = float(data['Entrada'][i].replace(",",'.'))
        if odd == 0:
            odd = 1
        print(f'Entrada: {odd}')
    else:
        odd = float(input("Entrada -> ").replace(",",'.'))

    if odd == 0:
        return array2s, array2n, odd
    if odd >= 4:
        odd = 4

    array2s.append(odd)
    if odd >= 2:
        corte2 = 1
    else:
        corte2 = 0

    array2n.append(corte2)

    return array2s, array2n, odd

def matriz_posicional(num_linhas, array):
    """
    Transforma um array unidimensional em uma matriz organizada por colunas.
    
    Args:
        array (list ou np.array): Lista de números a serem organizados.
        num_linhas (int): Número de linhas desejadas na matriz.

    Returns:
        np.array: Matriz ordenada.
    """
    
    # Reshape para matriz (por linha) e depois transpõe para organizar por colunas
    matriz = np.array(array).reshape(-1, num_linhas).T
    
    return matriz # Retorna como lista para melhor legibilidade

def matriz_historica(qt_colunas, array):
    """
    Gera uma matriz sequencial a partir de um array, com o número de colunas especificado.

    Args:
        array (list ou np.ndarray): Array de entrada.
        qt_colunas (int): Número de colunas desejado na matriz.

    Returns:
        np.ndarray: Matriz sequencial.
    """
    if qt_colunas > len(array):
        raise ValueError("O número de colunas não pode ser maior que o tamanho do array.")

    # Número de linhas na matriz
    num_linhas = len(array) - qt_colunas + 1

    # Criando a matriz sequencial
    matriz = np.array([array[i:i + qt_colunas] for i in range(num_linhas)])
    return matriz

def media_historica(array):
    """
    Função que organiza o histórico das médias das últimas 60 entradas.
    Args:
        array (list ou np.ndarray): Array de entrada.
    Returns:
        np.ndarray: Array das médias sequenciais.
    """
    array = np.asarray(array)  # Garante que a entrada seja um np.ndarray
    if len(array) < 60:
        raise ValueError("O array deve ter pelo menos 60 elementos para calcular a média histórica.")
    
    # Calcula a média móvel com janela de 60
    medias = np.convolve(array, np.ones(60) / 60, mode='valid')
    
    return medias

global_saved_arrays = []  # Lista global para armazenar os arrays

def save_merged_array(X):
    """
    Salva a matriz X em um array global.
    """
    global global_saved_arrays
    global_saved_arrays = X
    print(f'Matriz salva com sucesso! Total de matrizes armazenadas: {global_saved_arrays.shape}' )

def reden(array1, array2, array3):
    """
    Função para treinar uma rede neural com múltiplas entradas.

    Args:
        array1 (numpy.array): Saídas (rótulos) binárias (0 ou 1).
        array2 (numpy.array): Primeira matriz de entrada.
        array3 (numpy.array): Segunda matriz de entrada.

    Returns:
        keras.Model, float: Modelo treinado e a precisão da predição.
    """

    # Convertendo os rótulos para one-hot encoding
    y = to_categorical(array1, num_classes=2).reshape(-1,1)

    # Separando dados de treino e teste
    x_train_1, x_test_1, x_train_2, x_test_2, y_train, y_test = train_test_split(
        array2, array3, y, test_size=0.2, random_state=42
    )

    # Definindo as entradas
    input1 = Input(shape=(array2.shape[1],))  # Para Mf (60, 4)
    input2 = Input(shape=(array3.shape[1],))  # Para Mh (60, 183)

    # Camadas para a primeira entrada (Mf)
    x1 = Dense(32, activation="relu")(input1)
    x1 = Dense(16, activation="relu")(x1)

    # Camadas para a segunda entrada (Mh)
    x2 = Dense(64, activation="relu")(input2)
    x2 = Dense(32, activation="relu")(x2)

    # Concatenando as saídas
    merged = Concatenate()([x1, x2])

    # Camadas densas após a concatenação
    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)

    # Camada de saída
    output = Dense(1, activation="softmax")(x)

    # Criando o modelo
    model = Model(inputs=[input1, input2], outputs=output)

    # Compilando o modelo
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")]
    )

    batch_size = min(1024, x_train_1.shape[0])  # Garante que o batch_size não seja maior que o conjunto de treino
    epochs = 50

    # Treinando o modelo
    model.fit(
        [x_train_1, x_train_2], y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2
    )

    # Avaliação
    score = model.evaluate([x_test_1, x_test_2], y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")
    print(f"Precision: {score[2]:.4f}")
    print(f"Recall: {score[3]:.4f}")

    return model, score[1]  # Retorna o modelo e a acurácia da predição

data = pd.read_csv("/home/darkcover/Documentos/Out/dados/odds_200k_1.csv")
array_binario = []
array_float = []
i1 = 0
contagem = 0

array_contagem = []
array_acuracia = []
array_media = []

for i in range(5000):
    array_float, array_binario, odd = coletarodd(i, 5000, data, array_float, array_binario)
    
    if i >= 301:
        if y_pred[0] == 1:
            if odd >= 2:
                trick1 = 1
            else:
                trick1 = 0
            
            if y_pred[0] == trick1:
                contagem = contagem + 1
            else:
                contagem = contagem - 1
        print(f'contagem: {contagem}')
        array_contagem.append(contagem)

    if i % 60 == 0 and i >= 300:
        # Criando matrizes com os últimos 60 valores
        matriz_float = matriz_posicional(60, array_float[61:])
        matriz_binaria = matriz_posicional(60, array_binario[61:])

        print(f'Shapes -> Mf{matriz_float.shape} | Mb{matriz_binaria.shape} \n{"-"*12}')

        # Gerando a matriz de médias, garantindo que tenha 60 linhas
        array_media = media_historica(array_binario)
        matriz_media = matriz_historica(60, array_media).T

        print(f'Shape -> Mh{matriz_media.shape}')

        model = reden(matriz_binaria, matriz_float, matriz_media)

    if i >= 300:
        print(global_saved_arrays.shape)
        time.sleep(60)
        X_real = np.array(global_saved_arrays[i1 + 1,:]).reshape(1,-1)
        y_pred = model.predict(X_real)
        print(f'y_pred: {y_pred[0]}')

        i1 += 1
        if i1 == 59:
            i1 = 0
        
# Criando DataFrames
data_final = pd.DataFrame({"Contagem": array_contagem})
data_acuracia = pd.DataFrame({"Acuracia": array_acuracia})

# Escrevendo ambos no mesmo arquivo, mas em sheets diferentes
with pd.ExcelWriter('order.xlsx', engine='xlsxwriter') as writer:
    data_final.to_excel(writer, sheet_name='Contagem', index=False)
    data_acuracia.to_excel(writer, sheet_name='Acuracia', index=False)
