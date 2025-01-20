from typing import List
import pandas as pd
from modulos.Coletar import Odd
from modulos.Matriz import Ordernar60

#Carregar data - /content/drive/MyDrive/Out/dados/odds_200k.csv
data = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
limite: int = int(input("Insira o valor limite -> "))
print(24*'*-','*')
print(f'Carregando dados ...')
array_1: List[float] = data['Odd']
dados = Odd(array_1)
arrayf, arrayi = dados.array_float(), dados.array_int
ajuste = arrayf[:199980]
print(len(ajuste))



matrizf = Ordernar60(ajuste).tranformar()
