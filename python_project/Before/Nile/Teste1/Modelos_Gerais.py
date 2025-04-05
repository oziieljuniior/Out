import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

data = pd.read_csv("/home/darkcover/Documentos/Out/dados/odds_200k_1.csv")
array_binario = []
array_float = []
i1 = 0
contagem = 0

array_contagem = []
array_acuracia = []

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
        matriz_float = matriz(60, array_float[:i])
        matriz_binaria = matriz(60, array_binario[:i])

        X1 = matriz_float[:,:(matriz_float.shape[1] - 1)]
        y1 = matriz_binaria[:,-1]

        # Normalizar os recursos (importante para redes neurais)
        scaler1 = MinMaxScaler()
        X1 = scaler1.fit_transform(X1)

        # Separar os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
        # Treinando um modelo de Regressão Logística
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Fazendo previsões
        y_pred = model.predict(X_test)

        # Avaliando o desempenho do modelo
        accuracy = accuracy_score(y_test, y_pred)
        array_acuracia.append(accuracy)
        print(f'Acurácia: {accuracy}')
    if i >= 300:
        #print(X1.shape)
        X_real = np.array(X1[i1 + 1,:]).reshape(1,-1)
        y_pred = model.predict(X_real)
        print(f'y_pred: {y_pred[0]}')

        i1 += 1
        if i1 == 59:
            i1 = 0

        
        
data_final = pd.DataFrame({"Contagem":array_contagem})

data_final.to_excel('order1.xlsx')

data_acuracia = pd.DataFrame({"Acuracia": array_acuracia})
data_acuracia.to_excel('order2.xlsx')
