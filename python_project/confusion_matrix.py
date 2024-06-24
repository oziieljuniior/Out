import pandas as pd
import numpy as np

data = pd.read_csv('/home/darkcover/Documentos/Out/dados/data_final1.csv')

data = data.drop(columns=['Unnamed: 0'])
data = data.rename(columns={'Odd_Categoria': 'odd_saida'})

data = data.drop(0).reset_index(drop=True)


print(data)


i, j, k, l = 0,0,0,0
for (apostar, acerto) in zip(data['apostar'], data['acerto']):
    #Não apostar e acertar
    if apostar == 0 and acerto == 1:
        i += 1
    #Não apostar e errar
    elif apostar == 0 and acerto == 0:
        j += 1
    #Apostar e acertar
    elif apostar == 1 and acerto == 1:
        k += 1
    #Apostar e errar
    elif apostar == 1 and acerto == 2:
        l += 1



matriz_confusa = np.array([
    [i, j],
    [k, l],
])

matriz_confusa_ponderada = np.array([
    [i/200000, j/200000],
    [k/200000, l/200000],
])
soma = i + j + k + l
apostas = k + l
acertos = k / apostas
erros = l / apostas
print(f'Matriz de Confusão: \n {matriz_confusa}\nMatriz de confusão ponderada: \n {matriz_confusa_ponderada} \nPonderada em relação as apostas:\n {acertos, erros} \nQuantidade de entradas totais: {soma}')