import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping

# Libs
import time
import warnings

import time
# Configs
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

## Funções

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
    if odd >= 3:
        odd = 3

    array2s.append(odd)
    if odd >= 2:
        corte2 = 1
    else:
        corte2 = 0

    array2n.append(corte2)

    return array2s, array2n, odd

def matriz(num_linhas, array):
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
    Gera uma lista com possíveis predições.
    Args:
        t (int): Quantidade de modelos contidos na lista original.
        modelos (np.array): Array que contém modelos treinados.
        array1 (np.array): Lista contendo os últimos valores.
    Returns:
        np.array: Array que contém a predição de cada modelo da lista original.
    """
    y_pred1 = []
    tam = len(array1)
    for sk in range(0,t):
        if modelos[sk] is not None:
            posicao = 60*sk + 60
            ajuste = tam % posicao
            trick1 = tam - ajuste
            array_ajustado = array1[:trick1]
            print(sk, posicao,trick1, len(array_ajustado), ajuste)
            matriz1s = matriz(posicao,array_ajustado)
            if ajuste == 0:
                trick2 = -1
            else:
                trick2 = ajuste - 1
            cc = colunas[sk]
            n = matriz1s.shape[1]
            x_new = np.array(matriz1s[trick2,(n - cc) + 1:])
            x_new = x_new.astype("float32")
            x_new = np.expand_dims(x_new, -1)
            x_new = np.reshape(x_new, (-1, (cc-1), 1, 1))

            predictions = modelos[sk].predict(x_new)

            y_pred = np.argmax(predictions, axis=1)
            y_pred1.append(y_pred[0])
    return y_pred1

def calcular_media_array(array, m):
    ###FIX: Fazer com que ele retorne um array com o historico das médias.
    ###TARGET: Adicionar o historico das médias como vetor de entrada dos arrays.
    """
    Função que determina a média do array baseada na quantidade de linhas.
    
    Args:
        array (numpy.array): Vetor com entradas reais.
        m (int): Tamanho da média a ser calculado.
    
    Returns:
        np.array: Vetor unidimensional, com valores da média.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("O argumento 'array' deve ser um numpy array.")
    
    if m <= 0:
        raise ValueError("O valor de 'm' deve ser maior que zero.")
    
    medias = np.array([np.mean(array[-m:])])
    
    return medias

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
    # Calculando a média das últimas 60 entradas para cada amostra
    medias = np.array([np.mean(array3[max(0, i-60):i], axis=0) for i in range(m)])
    
    # Concatenando a média como uma nova feature
    X = np.concatenate((array3, medias), axis=1)
    y = np.array(array1)
    
    # Dividindo os dados em treino e teste
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
        layers.Dense(264, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        metrics=['accuracy', Precision(name="precision"), Recall(name="recall")]
    )

    # Treinamento
    batch_size = 2**10
    epochs = 50
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    # Avaliação
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")
    print(f"Precision: {score[2]:.4f}")
    print(f"Recall: {score[3]:.4f}")

    return [model, score[2]]

def ponderar_lista(lista, base=1.25):
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

    # Retornar 1 se média ponderada >= 0.55, senão 0
    return 1 if soma_ponderada / total_pesos >= 0.5 else 0


## Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
data = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k_1.csv')

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j2, j3 = 0,1,0,0

media_parray, acerto01 = [], []

acerto2, acerto3, core1 = 0,0,0

recurso1, recurso2 = [None]*5000, [None]*5000

array_geral = np.zeros(6, dtype=float)
df1 = pd.DataFrame({'lautgh1': np.zeros(60, dtype = int), 'lautgh2': np.zeros(60, dtype = int)})
modelos, acumu, atrasado, colunas = [None]*50, [0]*50, [0]*60, [0]*50

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

while i <= 210000:
    print(24*'---')
    #print(len(media_parray))
    if len(media_parray) < 59:
        m1 = 0
        core1 = 0
    else:
        m1 = media_parray[len(media_parray) - 60]
        core1 = i % 60

    print(f'Número da Entrada - {i} | Acuracia_{core1 + 1}: {round(m1,4)}')
    
    array2s, array2n, odd = coletarodd(i, inteiro, data, array2s, array2n)
    if odd == 0:
        break

    if i >= 301:
        print(24*"-'-")
        
        array_geral = placargeral(resultado, odd, array_geral)
        media_parray = placar60(df1, core1, media_parray, resultado, odd)
        
        if core1 == 0:
            core11 = 60
        else:
            core11 = core1
        print(f'Acuracia modelo Geral: {round(array_geral[0],4)} | Acuracia_{core11}: {round(media_parray[-1],4)} \nPrecisao modelo Geral: {round(array_geral[1],4)}')
        print(24*"-'-")

    if i >= 300 and (i % 60) == 0:
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
                trap = i // click
                if trap >= 5:
                    info.append(click)
        
        print(f'{12*"*-"} \nPosições que devem ser carregadas: {info} \n{12*"*-"}')
        for click in info:
            print(f'Treinamento para {click}')
            matrix1s, matrix1n = matriz(click, array2s[:i]), matriz(click, array2n[:i])
            posicao0 = int((click / 60) - 1)
            array1, array3 = matrix1n[:,-1], matrix1s[:,:-1]
            m, n = matrix1n.shape
            print(f'Matrix_{click}: {[matrix1n.shape, matrix1s.shape]} | Indice: {matrix1n.shape[1]} | Posicao: {posicao0}')
            models = reden(array1,array3, m, n)
            
            if i >= 420:
                if acumu[posicao0] < models[1]:
                    modelos[posicao0] = models[0]
                    acumu[posicao0] = models[1]
                    colunas[posicao0] = n
                    print('REDE NEURAL POSICIONAL ATUALIZADA...')
            else:
                modelos[posicao0] = models[0]
                acumu[posicao0] = models[1]
                colunas[posicao0] = n
            #modelos[posicao0] = models[0]
            #acumu[posicao0] = models[1]
            print(f'Treinamento {click} realizado com sucesso ... {acumu[posicao0]} \n')
            time.sleep(1)
        print('***'*20)


    if i >= 300:
        y_pred1 = lista_predicao(len(modelos), modelos, array2s)
        resultado = ponderar_lista(y_pred1)
        
        print(24*'*-')
        print(f'Entrada Cadastrada: {resultado}')
        print(24*'*-')
    
    i += 1