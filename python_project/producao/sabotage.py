import pandas as pd
import numpy as np

array1, array2, array3, array4, array5,  arraygeral = [], [], [], [], [], []
count1, count2, count3, count4, i1, j1 = 108, 52, 160, 320, 1, 0

data = pd.read_csv("/home/darkcover/Documentos/Out/dados/odds_200k.csv")

data = data.drop(columns=['Unnamed: 0'])
data = data.rename(columns={'Odd_Categoria': 'odd_saida'})
print("Data Carregada ...")

for i in range(160):
    if data['odd_saida'][i] >= 5:
        zeroum = 1
    else:
        zeroum = 0
    array1.append(zeroum)
    
while i1 != 0:
    i1 = input("Insira a última entrada determinada: ")

    i1 = i1.replace(',','.')

    i1 = float(i1)

    if i1 == 0:
        break

    if i1 < 10:
        if i1 < 5:
            if i1 < 3.5:
                if i1 < 2.6:
                    if i1 < 2.1:
                        if i1 < 1.7:
                            if i1 < 1.45:
                                if i1 < 1.3:
                                    if i1 < 1.15:
                                        if i1 < 1.05:
                                            odd_saida = 1
                                        else:
                                            odd_saida = 2
                                    else:
                                        odd_saida = 3
                                else:
                                    odd_saida = 4
                            else:
                                odd_saida = 5
                        else:
                            odd_saida = 6
                    else:
                        odd_saida = 7
                else:
                    odd_saida = 8
            else:
                odd_saida = 9
        else:
            odd_saida = 10
    else:
        odd_saida = 11
    
    if odd_saida >= 5:
        zeroum = 1
    else:
        zeroum = 0
    
    array1.append(zeroum)
    array2.append(zeroum)
    array3.append(zeroum)
    arraygeral.append(zeroum)
    
    j1 = j1 + 1

    prob1 = count1/count3 #Acerto de um array de 160 entradas
    prob2 = count2/count3 #Erro de um array de 160 entradas
    prob3 = sum(array1) / len(array1) #Media do array1
    prob4 = sum(array1) / 320 #Media do array1 com 320 entradas
    
    if prob1 <= 0 or prob2 <= 0 or prob1 > 1 or prob2 >1 or len(array1) == 320:
        array1 = array2
        arra2, array3, array5 = [], [], []
        count1, count2, count3, j1 = 108, 52, 160, 0

    if prob1 >= 0.75:
        array4.append(zeroum)
        array5.append(zeroum)
        media4 = np.mean(array4)
        media5 = np.mean(array5)
        print(f'j: {j1} \nProb Geral: {prob3} \nProb Localizado: {prob4} \nprob1: {prob1} \nprob2: {prob2} \nsaida: {zeroum} \nTaxa de acerto: {media4} \nTaxa de acerto local: {media5}')
        
    if zeroum == 1:
        count1 = count1 - 1
    if zeroum == 0:
        count2 = count2 - 1
        
    count3 = count3 - 1
        
    if prob1 < 0.75:
        print(f'j: {j1} \nProb Geral: {prob3} \nProb Localizado: {prob4} \nprob1: {prob1} \nprob2: {prob2} \nsaida: {zeroum}')    
            
            
            