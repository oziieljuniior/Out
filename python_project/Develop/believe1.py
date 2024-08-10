import pandas as pd
import numpy as np
import zlib
import time
from scipy.stats import chisquare

data = pd.read_csv("/home/darkcover/Documentos/Out/dados/odds_200k.csv")

data = data.drop(columns=['Unnamed: 0'])
data = data.rename(columns={'Odd_Categoria': 'odd_saida'})
print("Data Carregada ...")

array1, array2, array3, arraygeral = [], [], [], []
count1, count2, count3, count4 = 108, 52, 160, 320
#Carregar funções
def kolmogorov_complexity(data):
    data_bytes = bytes(data)
    compressed_data = zlib.compress(data_bytes)
    return len(compressed_data)

def martin_lof_randomness(data):
    # Exemplo simples: Teste do qui-quadrado
    expected = [len(data) / 2, len(data) / 2]
    observed = [sum(data), len(data) - sum(data)]
    chi2, p = chisquare(observed, f_exp=expected)
    return p > 0.05  # True se passar no teste de aleatoriedade


for i in range(len(data)):
    if data['odd_saida'][i] >= 5:
        zeroum = 1
    else:
        zeroum = 0
    array1.append(zeroum)
    array2.append(zeroum)
    arraygeral.append(zeroum)
    if len(array1) == 160:
        for j in range(i, i + 160):
            prob1 = count1/count3
            prob2 = count2/count3
            if data['odd_saida'][j] >= 5:
                zeroum = 1
            else:
                zeroum = 0
            array2.append(zeroum)
            array3.append(zeroum)
            print(f'i: {i} \nprob1: {prob1} \nprob2: {prob2} \nsaida: {zeroum}')
            #time.sleep(0.25)
            if zeroum == 1:
                count1 = count1 - 1
            else:
                count2 = count2 - 1
            count3 = count3 - 1
        # Avaliação de Kolmogorov
        kolmogorov_result = kolmogorov_complexity(array3)

        # Avaliação de Martin-Löf
        martin_lof_result = martin_lof_randomness(array3)
        media1 = np.mean(array1)
        media2 = np.mean(array2)
        media3 = np.mean(array3)
    
        print(f'Array1: {array1} \nTamanho do array1: {len(array1)} \nMedia array1: {media1} \nArray2: {array2} \nTamanho do array2: {len(array2)} \nMedia array2: {media2} \nArray3: {array3} \nTamanho do array3: {len(array3)} \nMedia array3: {media3} \nComplexidade de Kolmogorov: {kolmogorov_result} \nPassa no teste de Martin-Löf? {martin_lof_result}')

        array1, array2, array3 = [], [], []
        
        time.sleep(60)
        