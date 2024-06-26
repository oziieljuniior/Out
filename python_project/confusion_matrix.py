import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/home/darkcover/Documentos/Out/dados/data_final1.csv')

#data = data.drop(columns=['Unnamed: 0'])

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
    elif apostar == 1 and acerto == 0:
        l += 1



matriz_confusa = np.array([
    [i, j],
    [k, l],
])

matriz_confusa_ponderada = np.array([
    [i/199999, j/199999],
    [k/199999, l/199999],
])
soma = i + j + k + l
apostas = k + l
acertos = k / apostas
erros = l / apostas
print(f'Matriz de Confusão: \n {matriz_confusa}\nMatriz de confusão ponderada: \n {matriz_confusa_ponderada} \nPonderada em relação as apostas:\n {acertos, erros} \nQuantidade de entradas totais: {soma}')

# Calculando a matriz de correlação
corr_matrix = data.corr()

# Criando o heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=.5)
plt.title('Correlation heatmap')
plt.show()