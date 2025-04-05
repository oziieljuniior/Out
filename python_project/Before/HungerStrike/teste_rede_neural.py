import pandas as pd
import numpy as np

from statsmodels.tsa.ar_model import AutoReg

data = pd.read_csv('/home/darkcover/Documentos/Data/Out/Entrada.csv')

i = 0
inteiro = int(input("Insira a posição da data ---> "))
array1, array2 = [], []


while i <= 10000:
    print(24*'***')
    
    print(f'Posição data - {i}')
    if i <= inteiro:
        odd = data['Entrada'][i].replace(",",".")
        print(f'Entrada -> {odd}')
    else:
        odd = input("Entrada -> ").replace(",",".")
    odd = float(odd)
    if odd == 0:
        break

    if odd >= 2:
        array1.append(1)
    else:
        array1.append(0)

    if i >= 60:
        array2 = array1[i - 60: i]
        