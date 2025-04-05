## Criar planilha de jogos ~ fase 2
#Import de bibliotecas
import pandas as pd
import numpy as np

data_inicial = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
data_inicial = data_inicial.drop(columns=['Unnamed: 0'])
data_inicial = data_inicial.rename(columns={'Odd_Categoria': 'odd_saida'})
print("Data Carregada ...")

array1, array2, array3, array4, array5, array6, array7, array8, array9, array10 = [], [], [], [], [], [], [], [], [], []


for i in data_inicial['Odd']:
    print(i)
    if i < 10:
        array10.append(0)
    else:
        array10.append(1)
        
    if i < 5:
        array9.append(0)
    else:
        array9.append(1)
    
    if i < 3.5:
        array8.append(0)
    else:
        array8.append(1)
    
    if i < 2.6:
        array7.append(0)
    else:
        array7.append(1)
    
    if i < 2.1:
        array6.append(0)
    else:
        array6.append(1)
    
    if i < 1.7:
        array5.append(0)
    else:
        array5.append(1)                            

    if i < 1.45:
        array4.append(0)
    else:
        array4.append(1)

    if i < 1.3:
        array3.append(0)
    else:
        array3.append(1)
    
    if i < 1.15:
        array2.append(0)
    else:
        array2.append(1)                                        

    if i < 1.05:
        array1.append(0)
    else:
        array1.append(1)
        
print(len(array1),len(array2),len(array3),len(array4),len(array5),len(array6),len(array7),len(array8),len(array9),len(array10))
media1,media2,media3,media4,media5,media6,media7,media8,media9,media10 = sum(array1)/199999,sum(array2)/199999,sum(array3)/199999,sum(array4)/199999,sum(array5)/199999,sum(array6)/199999,sum(array7)/199999,sum(array8)/199999,sum(array9)/199999,sum(array10)/199999
print(media1,media2,media3,media4,media5,media6,media7,media8,media9,media10)