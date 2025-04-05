import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Nadam

import skfuzzy as fuzz

# Libs
import time
import warnings

# Configs
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

## Funções

array_global = []

def modificar_ag(odd):
    global array_global
    array_global.append(odd)
    

def placar60(df1, core1, media_parray, resultado, odd):
    """
    Função que organizar o placar das 60 entradas.
    Args:
        df1 (pd.DataFrame): DataFrame responsavel por armazenar as somas de cada entrada.
        core1 (int): Valor inteiro não-negativo crescente ciclico. Responsável por controlar a posição dos dados.
        media_parray (array): Array responsavel por armazenar a acuracia de cada entrada.
        resultado (int): Valor interio(0 ou 1) gerado pela função ponderar lista com a entrada anterior.
        odd (float): Valor real com duas casas decimais. Ele é determinado pela entrada dos dados, ou usuário.
    Returns:
        array: Array contendo as variaveis df1 e mediap_array atualizadas, também retorna a acuracia atualizada.
    """

    if resultado == 1:
        if odd >= 2:
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
    Implementação da lógica fuzzy para classificar as odds
    """
    odd_range = np.arange(1, 4.1, 0.1)
    
    # Conjuntos fuzzy
    baixo = fuzz.trimf(odd_range, [1, 1, 2])
    medio = fuzz.trimf(odd_range, [1.5, 2.5, 3.5])
    alto = fuzz.trimf(odd_range, [2.5, 4, 4])
    
    # Graus de pertinência
    pert_baixo = fuzz.interp_membership(odd_range, baixo, odd)
    pert_medio = fuzz.interp_membership(odd_range, medio, odd)
    pert_alto = fuzz.interp_membership(odd_range, alto, odd)
    
    # Classificação
    if pert_alto > pert_medio and pert_alto > pert_baixo:
        return 1  # Alta confiança na subida
    elif pert_medio > pert_baixo:
        return 0.5  # Média confiança
    else:
        return 0  # Baixa confiança

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
        #odd = float(data['Odd'][i])
        odd = float(data['Entrada'][i].replace(",",'.'))
        if odd == 0:
            odd = 1
        print(f'Entrada: {odd}')
    else:
        odd = float(input("Entrada -> ").replace(",",'.'))

    if odd == 0:
        return array2s, array2n, odd
    
    if odd >= 4:
        odd = 4
    
    corte1 = fuzzy_classification(odd)
    array2s.append(odd) # Aplicando lógica fuzzy
    array2n.append(corte1)
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
        if odd >= 2:
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
        if odd < 2:
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


def lista_predicao(t, modelos, array1):
    """
    Gera uma lista com possíveis predições contínuas entre 0 e 1.
    
    Args:
        t (int): Quantidade de modelos na lista original.
        modelos (list): Lista contendo modelos treinados.
        array1 (np.array): Lista contendo os últimos valores.

    Returns:
        list: Lista que contém a predição de cada modelo.
    """
    y_pred1 = []
    for sk in range(t):
        if modelos[sk] is not None:
            posicao = 60 * sk + 60
            print(sk, posicao)

            matriz1s = matriz(posicao, array1)

            x_new = np.array(matriz1s[-1, 1:])  # Pegamos a última entrada para predição
            x_new = x_new.astype("float32")
            x_new = np.expand_dims(x_new, -1)
            x_new = np.reshape(x_new, (-1, (matriz1s.shape[1]) - 1, 1, 1))

            predictions = modelos[sk].predict(x_new)
            y_pred = predictions.flatten()[0]  # Pegamos o valor contínuo entre 0 e 1
            
            y_pred1.append(y_pred)

    print(y_pred1)
    return y_pred1

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split

def reden(array1, array3, m, n, salvar=True, carregar=False):
    """
    Treina uma rede neural para prever valores contínuos entre 0 e 1, com suporte a salvamento e carregamento.

    Args:
        array1 (numpy.array): Valores reais entre 0 e 1 (target).
        array3 (numpy.array): Entradas preditoras.
        m (int): Número de amostras.
        n (int): Número de características por amostra.
        salvar (bool): Se True, salva o modelo treinado.
        carregar (bool): Se True, carrega o modelo previamente salvo.

    Returns:
        keras.Model: Modelo treinado.
    """

    # Diretório para salvar os modelos
    dir_modelo = "modelos_salvos"
    modelo_path = os.path.join(dir_modelo, "ultimo_modelo.keras")
    pesos_path = os.path.join(dir_modelo, "pesos.weights.h5")
    historico_path = os.path.join(dir_modelo, "historico.json")

    if carregar and os.path.exists(modelo_path):
        print("Carregando modelo pré-treinado...")
        model = keras.models.load_model(modelo_path)
        return model

    # Criando diretório se não existir
    os.makedirs(dir_modelo, exist_ok=True)

    # Dividindo os dados em treino e teste
    X = np.array(array3)
    y = np.array(array1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Ajustando dimensões para entrada no modelo
    x_train = np.expand_dims(x_train, -1).astype("float32")
    x_test = np.expand_dims(x_test, -1).astype("float32")
    input_shape = (n - 1, 1)  # Formato esperado de entrada

    # Definição do modelo
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(264, use_bias=True),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),  # Dropout adaptativo

        layers.Dense(128, use_bias=True),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.2),

        layers.Dense(64, use_bias=True),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),

        layers.Dense(1, activation="sigmoid"),  # Saída contínua entre 0 e 1
    ])

    model.compile(
        loss="mean_squared_error",  # Ajustado para regressão
        optimizer=Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        metrics=['mae', 'mse']
    )

    # Treinamento
    batch_size = min(2**10, x_train.shape[0])  # Ajusta batch_size ao tamanho do conjunto
    epochs = 100

    historico = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )

    # Avaliação
    score = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test MSE: {score[1]:.4f}")
    print(f"Test MAE: {score[0]:.4f}")

    # Salvando modelo, pesos e histórico
    if salvar:
        model.save(modelo_path)  # Salva no novo formato .keras
        model.save_weights(pesos_path)  # Salva os pesos corretamente com a extensão .weights.h5

        with open(historico_path, "w") as f:
            json.dump(historico.history, f)

        print("Modelo, pesos e histórico salvos com sucesso!")

    return model



def ponderar_lista(lista):
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

    # Calcular soma ponderada e total de pesos
    soma_ponderada = sum(lista)
    total_pesos = n
    
    qrange = 0.55

    # Retornar 1 se média ponderada >= 0.5, senão 0
    return 1 if soma_ponderada / total_pesos >= qrange else 0


## Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
data = pd.read_csv('/home/darkcover/Documentos/Out/dados/Saidas/FUNCOES/DOUBLE - 17_09_s1.csv')

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j2, j3 = 0,0,0,0

media_parray, acerto01 = [], []

acerto2, acerto3, core1 = 0,0,0

recurso1, recurso2 = [None]*5000, [None]*5000

array_geral = np.zeros(6, dtype=float)
df1 = pd.DataFrame({'lautgh1': np.zeros(60, dtype = int), 'lautgh2': np.zeros(60, dtype = int)})
modelos, acumu, atrasado = [None]*50, [0]*50, [0]*60

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

contagem = 0

array_contagem = []

limite_i = 360
contagem2 = 5

while i <= 210000:
    print(24*'---')
    #print(len(media_parray))
    if len(media_parray) < 59:
        m1 = 0
        core1 = 0
    else:
        m1 = media_parray[len(media_parray) - 60]
        core1 = i % 60

    print(f'Número da Entrada - {i} | Acuracia_{core1}: {round(m1,4)}')
    
    array2s, array2n, odd = coletarodd(i, inteiro, data, array2s, array2n)
    modificar_ag(odd)
    if odd == 0:
        break

    if i >= (limite_i + 1):
        print(24*"-'-")
        
        array_geral = placargeral(resultado1, odd, array_geral)
        media_parray = placar60(df1, core1, media_parray, resultado1, odd)
        
        if core1 == 0:
            core11 = 60
        else:
            core11 = core1
            
        print(f'Acuracia modelo Geral: {round(array_geral[0],4)} | Acuracia_{core11}: {round(media_parray[-1],4)} \nPrecisao modelo Geral: {round(array_geral[1],4)}')
        
        print(24*"-'-")
        if resultado1 == 1:
            if odd >= 2:
                trick1 = 1
            else:
                trick1 = 0
            
            if resultado1 == trick1:
                contagem = contagem + 1
            else:
                contagem = contagem - 1
        print(24*'-')    
        print(f'Contagem: {contagem} | T. Data: {len(data)}')
        array_contagem.append(contagem)
        if  contagem >= 5 or contagem <= -5:
            print("ATENÇÃO ...")
        print(24*"-'-")


    if i >= limite_i and (i % 60) == 0:
        print('***'*20)
        print(f'Carregando dados ...')
        info = []
        cronor = (i + 300) // 5
        lista = [name for name in range(60, cronor, 60)]

        if len(lista) >= (len(modelos) - 25):
            print(f'T. Lista: {len(lista)} | T. Mod. Real: {len(modelos)} | T. Mod. Ajustado: {len(modelos)}')
            cronor1 = [None]*50
            modelos = modelos.extend(cronor1)
            acumu = acumu.extend(cronor1)

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
            
            print(f'Treinamento {click} realizado com sucesso ...')
        print('***'*20)


    if i >= limite_i:
        y_pred1 = lista_predicao(len(modelos), modelos, array2s)
        resultado1 = ponderar_lista(y_pred1)
                
        print(24*'*-')
        print(f'Predição da Entrada: {resultado1}')
        print(24*'*-')
    
    i += 1


# Criando DataFrames
data_final = pd.DataFrame({"Contagem": array_contagem})
#data_acuracia = pd.DataFrame({"Acuracia": array_acuracia})

# Escrevendo ambos no mesmo arquivo, mas em sheets diferentes
with pd.ExcelWriter('order.xlsx', engine='xlsxwriter') as writer:
    data_final.to_excel(writer, sheet_name='Contagem', index=False)
#    data_acuracia.to_excel(writer, sheet_name='Acuracia', index=False)