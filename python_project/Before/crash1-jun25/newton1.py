import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

##Funções
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

#Produção
data = pd.read_csv('/home/darkcover/Documentos/Out/Documentos/dados/odds_200k.csv')
data_T1 = data.iloc[:100000]['Odd'].reset_index(drop=True)
data_T2 = []
for i in range(len(data_T1)):
    if data_T1[i] >= 2:
        data_T2.append(2)
    else:
        if data_T1[i] == 0:
            data_T2.append(1)
        else:
            data_T2.append(data_T1[i])

data_T2 = {'Odd': data_T2}
matriz1 = matriz(15, data_T2['Odd'])

print(matriz1.shape)


## Treinamento do modelo de árvore de decisão
# Considerando matriz1 já existente
X = matriz1[:, :-1]  # colunas 0 a 8
y = matriz1[:, -1]   # coluna 9

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Prever e avaliar
y_pred = tree.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")


odd = 1
count, count1, count2 = 0,0,0
order = []
while odd != 0:
    print(f'Insira as 9 últimas entradas: ')
    while count < 9:
        odd = float(input(f'Insira a {count+1}ª entrada: '))
        order.append(odd)
        count += 1
    print(f'Order: {order}')
    pred = tree.predict([order])
    print(f'Próximo valor será: {pred}')
    print(12*'**')
    odd = float(input('Insira o próximo valor: '))
    if pred >= 1.68:
        count1 += 1
        if odd >= 1.5:
            count2 += 1
    if count1 == 0:
        print('Nenhum valor acima de 1.68 foi inserido.')
    else:
        print(f'Contagem: {count1} | Contagem 1: {count2} \nPercentual: {count2/count1:.2%} | Total: {count1}')
    order.append(odd)
    order = order[1:]
    print(order)


