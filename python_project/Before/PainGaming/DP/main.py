import pandas as pd
import numpy as np
from modulos.Matriz import Ordernar
from modulos.Coletar import Odd
from modulos.RedeNeural import DP
import time

#Carregar data - /content/drive/MyDrive/Out/dados/odds_200k.csv
data = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
array = data['Odd']
lista_modelos, ordernar_precisao = [None]*50, [0]*50

for i in range(3000):
    if i >= 600 and (i % 60) == 0:
        array1 = array[:i]
        #print(len(array1))
        trick1 = Odd(array1)
        array1 = trick1.array_float()
        array_float, array_class = trick1.array_truncado(), trick1.array_int()
        
        trick2_float, trick2_class = Ordernar(array_float), Ordernar(array_class)
                
        lista_matriz_float, lista_matriz_class = trick2_float.tranformar(), trick2_class.tranformar()
        
        for order in range(len(lista_matriz_float)):
            matriz_float, matriz_class = np.array(lista_matriz_float[order]), np.array(lista_matriz_class[order])
            m,n = np.array(matriz_float).shape
            print(m,n)
            array3 = matriz_float[:,:-1]
            array1 = matriz_class[:,-1]
            trick3 = DP(array1,array3,n)
            modelss = trick3.reden()
