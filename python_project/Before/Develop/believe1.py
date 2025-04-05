import pandas as pd
import numpy as np
import zlib
import time
from scipy.stats import chisquare

data = pd.read_csv("/home/darkcover/Documentos/Out/dados/odds_200k.csv")

data = data.drop(columns=['Unnamed: 0'])
data = data.rename(columns={'Odd_Categoria': 'odd_saida'})
print("Data Carregada ...")

array1, array2, array3, array4, array5,  arraygeral = [], [], [], [], [], []
count1, count2, count3, count4 = 108, 52, 160, 320

t1 = len(data)
for i in range(t1):
    if data['odd_saida'][i] >= 5:
        zeroum = 1
    else:
        zeroum = 0
    array1.append(zeroum)
    array2.append(zeroum)
    arraygeral.append(zeroum)
    if len(array1) == 160:
        quest1 = t1 - i
        if quest1 >= 0:
            for j in range(i, i + 160):
                prob1 = count1/count3
                prob2 = count2/count3
                if prob2 <= 0 or prob1 > 1:
                    break
                if data['odd_saida'][j] >= 5:
                    zeroum = 1
                else:
                    zeroum = 0
                array2.append(zeroum)
                array3.append(zeroum)
                if prob1 >= 0.72:
                    array4.append(zeroum)
                    array5.append(zeroum)
                    media4 = np.mean(array4)
                    media5 = np.mean(array5)
                    print(f'j: {j} \nprob1: {prob1} \nprob2: {prob2} \nsaida: {zeroum} \nTaxa de acerto: {media4} \nTaxa de acerto local: {media5}')
                #time.sleep(0.25)
                if zeroum == 1:
                    count1 = count1 - 1
                else:
                    count2 = count2 - 1
                count3 = count3 - 1
            
            array1, array2, array3, array5 = [], [], [], []
            count1, count2, count3 = 108, 52, 160
print(len(array4))
            
            
            