import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    corte2 = fuzzy_classification(odd)  # Aplicando lógica fuzzy
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

data = pd.read_csv("/home/darkcover/Documentos/Out/dados/odds_200k.csv")
array_binario, array_float, array_contagem, array_acuracia = [], [], [], []
i1, i, contagem = 0, 0, 0
t = len(data)

while i <= t:
    if i >= len(data):
        break
    print(36*'*')
    print(f'Rodada: {i}')
    array_float, array_binario, odd = coletarodd(i, len(data), data, array_float, array_binario)
    if odd == 0:
        break

    if i >= 301:
        if y_pred[0] == 1:
            trick1 = 1 if odd >= 2 else 0
            contagem += 1 if y_pred[0] == trick1 else -1
        print(24*'-')    
        print(f'Contagem: {contagem} | T. Data: {len(data)}')
        array_contagem.append(contagem)
        if abs(contagem) >= 10:
            data = data.iloc[i-301:].reset_index(drop=True)
            i = 0
            contagem = 0
        print(24*'-') 

    if i % 60 == 0 and i >= 300:
        matriz_float = matriz(60, array_float[:i])
        matriz_binaria = matriz(60, array_binario[:i])

        X1 = matriz_float[:, :(matriz_float.shape[1] - 1)]
        y1 = matriz_binaria[:, -1]
        
        scaler1 = MinMaxScaler()
        X1 = scaler1.fit_transform(X1)

        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

        y_train = np.round(y_train).astype(int)  # Garante que os valores sejam 0 ou 1
        y_test = np.round(y_test).astype(int)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        array_acuracia.append(accuracy)
        
    if i >= 300:
        #print(X1.shape)
        X_real = np.array(X1[i1 + 1,:]).reshape(1,-1)
        y_pred = model.predict(X_real)
        print(f'y_pred: {y_pred[0]}')

        i1 += 1
        if i1 == 59:
            i1 = 0
    
    i += 1
    
# Criando DataFrames
data_final = pd.DataFrame({"Contagem": array_contagem})
data_acuracia = pd.DataFrame({"Acuracia": array_acuracia})

# Escrevendo ambos no mesmo arquivo, mas em sheets diferentes
with pd.ExcelWriter('order.xlsx', engine='xlsxwriter') as writer:
    data_final.to_excel(writer, sheet_name='Contagem', index=False)
    data_acuracia.to_excel(writer, sheet_name='Acuracia', index=False)

# Visualização gráfica
plt.figure(figsize=(10,5))
plt.plot(array_contagem, label='Contagem', color='blue')
plt.xlabel('Iteração')
plt.ylabel('Contagem')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(array_acuracia, label='Acurácia', color='red')
plt.xlabel('Iteração')
plt.ylabel('Acurácia')
plt.legend()
plt.grid()
plt.show()
