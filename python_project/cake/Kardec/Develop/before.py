import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
#activation = tf.keras.activations.elu

import skfuzzy as fuzz

# Libs
import time
import warnings


# Configs
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

## Funções
def calculate_means(array4):
    """
    Calcula a média dos elementos de array4 em janelas deslizantes de 59 elementos.

    Args:
        array4 (list): Lista de inteiros (0 ou 1).

    Returns:
        list: Lista com a média dos elementos em janelas de 59 elementos.
    """
    array6 = []
    array7 = []
    for i in range(len(array4) - 1):
        array6.append(array4[i])
        if i >= 59:
            order = float(np.mean(array6))
            array7.append(order)
    
    return array7

def calculate_orders(array4):
        """
        Calcula a soma dos elementos de array4 em janelas deslizantes de 59 elementos.

        Args:
            array4 (list): Lista de inteiros (0 ou 1).

        Returns:
            list: Lista com a soma dos elementos em janelas de 59 elementos.
        """
        # Calcular a soma dos elementos em janelas de 59 elementos
        array5 = []
        for i in range(len(array4) - 1):
            if i >= 59:
                order = sum(array4[i - 59: i])
                array5.append(order)
        return array5

def placar60(df1, i, media_parray, resultado, odd):
    """
    Função que organizar o placar das 60 entradas.
    Args:
        df1 (pd.DataFrame): DataFrame responsavel por armazenar as somas de cada entrada.
        i (int): Valor inteiro não-negativo crescente. Responsável por controlar a posição dos dados.
        media_parray (array): Array responsavel por armazenar a acuracia de cada entrada.
        resultado (int): Valor interio(0 ou 1) gerado pela função ponderar lista com a entrada anterior.
        odd (float): Valor real com duas casas decimais. Ele é determinado pela entrada dos dados, ou usuário.
    Returns:
        array: Array contendo as variaveis df1 e mediap_array atualizadas, também retorna a acuracia atualizada.
    """
    core1 = i % 60
    if resultado == 1:
        if odd >= 3:
            df1.iloc[core1,:] += 1
            medida_pontual = df1.iloc[core1, 0] / df1.iloc[core1, 1]
        else:
            df1.iloc[core1,1] += 1
            medida_pontual = df1.iloc[core1, 0] / df1.iloc[core1, 1]
    else:
        if len(media_parray)<59:
            medida_pontual = 0
        else:
            medida_pontual = media_parray[len(media_parray) - 60]

    media_parray.append(medida_pontual)

    return media_parray

def fuzzy_classification(odd):
    """
    Implementação da lógica fuzzy para classificar as odds no intervalo de 1 a 6.
    """
    odd_range = np.arange(1, 6.1, 0.1)
    
    # Conjuntos fuzzy ajustados para cobrir todo o intervalo de 1 a 6
    baixo = fuzz.trimf(odd_range, [1, 1, 2])
    medio = fuzz.trimf(odd_range, [1.5, 3, 4.5])
    alto = fuzz.trimf(odd_range, [3.5, 5, 6])
    muito_alto = fuzz.trimf(odd_range, [4.5, 6, 6])
    
    # Graus de pertinência
    pert_baixo = fuzz.interp_membership(odd_range, baixo, odd)
    pert_medio = fuzz.interp_membership(odd_range, medio, odd)
    pert_alto = fuzz.interp_membership(odd_range, alto, odd)
    pert_muito_alto = fuzz.interp_membership(odd_range, muito_alto, odd)
    
    # Classificação baseada nos graus de pertinência
    max_pert = max(pert_baixo, pert_medio, pert_alto, pert_muito_alto)
    
    if max_pert == 0:
        return 0  # Nenhuma confiança
    
    if max_pert == pert_muito_alto:
        return 1  # Alta confiança na subida
    elif max_pert == pert_alto:
        return 0.75  # Confiança moderada-alta
    elif max_pert == pert_medio:
        return 0.5  # Confiança média
    else:
        return 0.25  # Baixa confiança

def coletarodd(i, inteiro, data, array2s, array2n, alavanca=True):
    """
    Função que coleta e organiza as entradas iniciais do banco de dados.
    Args:
        i (int): Valor inteiro não-negativo. Entrada que controla o loop principal. É um valor cumulativo.
        inteiro (int): Valor inteiro não-negativo. Entrada que determina até aonde os dados devem ser carregados automaticamente, através de um banco de dados.
        data (pd.DataFrame): Variável carregada inicialmente para treinamento/desenvolvimento. Do tipo data frame.   #FIXWARNING2
        array2s (np.array): Array cumulativo que carrega as entradas reais com duas casas decimais.
        array2n (np.array): Array cumulativo que carrega as entredas inteiras(0 ou 1).
        alanvanca (bool): Variável booleana que determina se a entrada é automática ou manual.   #FIXWARNING1
    Returns:
        np.array: Array cumulativo que carrega as entradas reais com duas casas decimais.
        np.array: Array cumulativo que carrega as entredas inteiras(0 ou 1).
        float: Valor real com duas casas decimais. Ele é determinado pela entrada dos dados, ou usuário.
    """

#FIXWARNING1: O formato da data de entrada pode ser mudado? Atualmente está em .csv

    if i <= inteiro:
        if alavanca == True:
            odd = float(data['Entrada'][i].replace(",",'.'))
        else:
            odd = data['Entrada'][i] 

        if odd == 0:
            odd = 1
        print(f'Entrada: {odd}')
    else:
        odd = float(input("Entrada -> ").replace(",",'.'))

    if odd == 0:
        return array2s, array2n, odd
    if odd >= 6:
        odd = 6
    
    corte1 = fuzzy_classification(odd)  # Aplicando lógica fuzzy
    array2s.append(corte1)
    if odd >= 3:
        corte2 = 1
    else:
        corte2 = 0    
    array2n.append(corte2)

    return array2s, array2n, odd

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

def placargeral(resultado, odd, array_geral):
    """
    Função que realiza o gerenciamento da precisão e acurácia gerais.
    Args:
        resultado (int): Valor interio(0 ou 1) gerado pela função ponderar lista com a entrada anterior.
        odd (float): Valor real com duas casas decimais. Ele é determinado pela entrada dos dados, ou usuário.
        array_geral (np.array): O array contém algumas informações essenciais para o prosseguimento das entradas. -->["acuracia" (float): Valor real não-negativo. Responsável por acompanhar acerto 0,"precisao" (float): Valor real não-negativo. Responsavel por acompanhar acertos 0 e 1, "acerto"(int): Valor inteiro não-negativo. Essa variável é cumulativa com valor inicial igual a zero, ela é determinada a partir da entrada 301. Além disso, ela é responsável pela contagem de acertos da acurácia.   #FIXWARNING1, "acerto1" (int): Valor inteiro não-negativo. Essa variável é cumulativa com valor inicial igual a zero, ela é determinada a partir da entrada 301. . Além disso, ela é responsável pela contagem de acertos da precisão.   #FIXWARNING1, "j" (int): Valor inteiro não-negativo. Essa variável é cumulativa com valor inicial igual a zero, ela é determinada a partir da entrada 301. . Além disso, ela é responsável pela contagem geral da acurácia.   #FIXWARNING1, "j1" (int): j (int): Valor inteiro não-negativo. Essa variável é cumulativa com valor inicial igual a zero, ela é determinada a partir da entrada 301. . Além disso, ela é responsável pela contagem geral da precisão.   #FIXWARNING1 #FIXWARNING3
    Returns:
        np.array: Array contendo acurácia e precisão. Alem de conter acerto, acerto1, j e j1.
    """

#FIXWARNING1: Essa entrada é fixa ?
#FIXWARNING3: Ajustar essa documentação.
#array_geral = [acuracia, precisao, acerto, acerto1, j, j1]

    name = resultado

    if name == 1:
        if odd >= 3:
            count = 1
            if count == name:
                array_geral[2] += 1
                array_geral[3] += 1
                array_geral[4] += 1
                array_geral[5] += 1
        else:
            array_geral[4] += 1
            array_geral[5] += 1
    else:
        if odd < 3:
            count = 0
            if count == name:
                array_geral[3] += 1
                array_geral[5] += 1
        else:
            array_geral[5] += 1

    if array_geral[4] != 0 and array_geral[5] != 0:
        array_geral[0] = (array_geral[2] / array_geral[4]) * 100
        array_geral[1] = (array_geral[3] / array_geral[5]) * 100
    else:
        array_geral[0], array_geral[1] = 0,0

    return array_geral

def lista_predicao(i, t, modelos, array1, array2):
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
            matriz1s, matriz1n, posicao0 = tranforsmar_final_matriz(posicao,array1, array2)

            x_new = np.array(matriz1s[-1,3:])
            x_new = x_new.astype("float32")
            x_new = np.expand_dims(x_new, -1)
            x_new = np.reshape(x_new, (-1, (matriz1s.shape[1] - 3), 1, 1))

            predictions = modelos[sk].predict(x_new)

            y_pred = np.argmax(predictions, axis=1)
            y_pred1.append(y_pred[0])
    print(y_pred1)
    return y_pred1

def reden(array1, array2):
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
    X = np.array(array1)
    y = np.array(array2)
    n = X.shape[1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Ajustando dimensões para entrada no modelo
    x_train = np.expand_dims(x_train, -1).astype("float32")
    x_test = np.expand_dims(x_test, -1).astype("float32")
    input_shape = (n, 1)  # Formato esperado de entrada

    # Convertendo saídas para categóricas
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Definição do modelo
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation="relu", use_bias=True),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu", use_bias=True),
        layers.Dense(32, activation=tf.keras.activations.swish, use_bias=True),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.15, gdamma=2.0),
        #loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0), #testa loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        metrics=['accuracy', Precision(name="precision"), Recall(name="recall")]
    )

    # Treinamento
    batch_size = 2**10
    epochs = 50
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
    )

    # Avaliação
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")
    print(f"Precision: {score[2]:.4f}")
    print(f"Recall: {score[3]:.4f}")

    return model

def ponderar_lista(lista, base=1.10):
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
    pesos = [base ** (n - i) for i in range(n)]

    # Calcular soma ponderada e total de pesos
    soma_ponderada = sum(elemento * peso for elemento, peso in zip(lista, pesos))
    total_pesos = sum(pesos)
    
    qrange = 1 / n

    # Retornar 1 se média ponderada >= 0.5, senão 0
    return 1 if soma_ponderada / total_pesos >= qrange else 0

def tranforsmar_final_matriz(click, array1s, array1n):
    """
        Reponsavel por carregar matriz final. Idealmente elaborado
        para comportar outras variáveis de entrada.
        Args:
            click (int): Valor inteiro não-negativo. Entrada 
                que controla o loop principal. É um valor cumulativo.
            array1s (np.array): Array com entradas vetorizadas float.
            array1n (np.array): Array com entradas vetorizadas int.
        Returns:
            np.array: Matriz final.
    """
    n1 = len(array1n) - 61
    print(n1)
    arrayacertos60 = calculate_orders(array1n)
    matrizacertos60 = matriz(click, arrayacertos60[1:])
    arraymediamovel = calculate_means(array1n)
    matrizmediamovel = matriz(click, arraymediamovel[1:])
    print(len(array1s[1:]), len(array1n[1:]))
    matrix1s, matrix1n = matriz(click, array1s[1:]), matriz(click, array1n[1:])
    matrix1s, matrix1n = matrix1s[60:,:], matrix1n[60:,:]

    print(matrix1n.shape, matrix1s.shape, matrizacertos60.shape, matrizmediamovel.shape)
    posicao0 = int((click // 60) - 1)

    # Criar DataFrames separando cada coluna
    df_x1 = pd.DataFrame(matrizacertos60, columns=[f'X1_{i}' for i in range(matrizacertos60.shape[1])])
    df_x2 = pd.DataFrame(matrizmediamovel, columns=[f'X2_{i}' for i in range(matrizmediamovel.shape[1])])
    df_x3 = pd.DataFrame(matrix1s, columns=[f'X3_{i}' for i in range(matrix1s.shape[1])])

    # Juntar todas as colunas
    X_df = pd.concat([df_x1, df_x2, df_x3], axis=1)

    # Transformar para valores NumPy
    matrix1s = X_df.values

    print(matrix1s.shape, matrix1n.shape)  # Saída: (60, 24)

    return matrix1s, matrix1n, posicao0

## Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
data = pd.read_csv('/home/darkcover/Documentos/Out/dados/Saidas/FUNCOES/DOUBLE - 17_09_s1.csv')

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j2, j3 = 0,0,0,0

media_parray, acerto01 = [], []

acerto2, acerto3, core1 = 0,0,0

modelos = [None]*5000
data_matrizes = [None]*5000
recurso1, recurso2 = [None]*5000, [None]*5000

array_geral = np.zeros(6, dtype=float)
df1 = pd.DataFrame({'lautgh1': np.zeros(60, dtype = int), 'lautgh2': np.zeros(60, dtype = int)})

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

data_matriz_float = []
data_matriz_int = []
data_array_float = []
data_array_int = []
array_geral_float = []

while i <= 210000:
    print(24*'---')
    #print(len(media_parray))
    if len(media_parray) < 59:
        m = 0
        core1 = 0
    else:
        m = media_parray[len(media_parray) - 60]

    print(f'Número da Entrada - {i} | Acuracia_{core1 + 1}: {round(m,4)}')
    
    array2s, array2n, odd = coletarodd(i, inteiro, data, array2s, array2n)
    array_geral_float.append(float)
    if odd == 0:
        break

    if i >= 361:
        print(24*"-'-")
        
        array_geral = placargeral(resultado, odd, array_geral)
        media_parray = placar60(df1, i, media_parray, resultado, odd)
        
        if i % 60 == 0:
            core11 = 60
        else:
            core11 = core1
        print(f'Acuracia modelo Geral: {round(array_geral[0],4)} | Acuracia_{core11}: {round(media_parray[-1],4)} \nPrecisao modelo Geral: {round(array_geral[1],4)}')
        print(24*"-'-")

    if i >= 360 and (i % 60) == 0:
        print('***'*20)
        print(f'Carregando dados ...')
        info = []
        cronor = (i + 300) // 5
        lista = [name for name in range(60, cronor, 60)]
        tier = True

        for click in lista:
            k0 = i % click
            k1 = (i - 60) % click
            if k0 == 0 and k1 == 0:
                info.append(click)
        print(f'{12*"*-"} \nPosições que devem ser carregadas: {info} \n{12*"*-"}')

        for click in info:
            print(f'Treinamento para {click}')
            matriz_final_float, matriz_final_int, posicao0 = tranforsmar_final_matriz(click, array2s, array2n)
            print(f'Matrix_{click}: {[matriz_final_float.shape, matriz_final_int.shape]} | Posicao: {posicao0}')
            data_matriz_float.append(matriz_final_float), data_matriz_int.append(matriz_final_int)
            n = matriz_final_float.shape[1]
            array1, array2 = matriz_final_float[:,:(n - 3)], matriz_final_int[:,-1]
            models = reden(array1, array2)
            modelos[posicao0] = models
            #data_matrizes[posicao0] = matriz_final_float
            print(f'Treinamento {click} realizado com sucesso ...  \n')
        print('***'*20)

    if i >= 360:
        core2 = i % 60
        y_pred1 = lista_predicao(i, len(modelos), modelos, array2s, array2n)
        resultado = ponderar_lista(y_pred1)
        print(24*'*-')
        print(f'Proxima Entrada: {resultado}')
        print(24*'*-')


    i += 1

print(f'Ordenandos os dados ...')

data_array_float = np.array(array2s)
data_array_int = np.array(array2n)

data_final = pd.DataFrame({'Entrada': data_array_float, 'Resultado': data_array_int})
data_final.to_csv('/home/darkcover/Documentos/Out/dados/DoIt/Order1.csv', index=False)



