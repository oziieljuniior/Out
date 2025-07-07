# Libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Placares import Placar # Importando a classe Placar do módulo Placares
from Modulos.Vetores import AjustesOdds
from Modulos.RedeNeural import Modelos

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np

import time

    
### Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
data = pd.read_csv('/home/darkcover/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10/Vitoria1_10 - game_teste3x1.csv')

array1, i = [], 0

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

## Variáveis para salvar em um dataframe
data_matriz_float, data_matriz_int, array_geral_float, historico_janelas = [], [], [], [] 

placar = Placar()  # Inicializando o placar
vetores = AjustesOdds(array1)  # Inicializando a classe de ajustes de odds
### Produção
while i <= 210000:
    print(24*'---')
    print(f'Rodada - {i}')

######## -> Vetor de Entradas Unidimensional ##########        
    arrayodd, odd = vetores.coletarodd(i, inteiro, data)
    array_geral_float.append(odd)

    if odd == 0:
        break
######################################################

######## -> Placar ###################################      
    if i >= 241:
        print(24*"-'-")
        array_placar = placar.atualizar_geral(i, resultado, odd)
        print(f'Precisão Geral: {array_placar["Precisao_Geral"]:.2f}% \nPrecisão Rede Neural: {array_placar["Precisao_Sintetica"]:.2f}%')
        print(24*"-'-")
######################################################

######## -> Treinamento da Rede Neural ###############
    if i >= 240 and (i % 120) == 0:
        print('***'*20)
        ##### -> Vetores de Entradas #################
        print(f'Carregando dados ...')
        matriz_final_float, matriz_final_int = vetores.tranforsmar_final_matriz(arrayodd)
        print(f'Matrix: {[matriz_final_float.shape, matriz_final_int.shape]}')
        data_matriz_float.append(matriz_final_float), data_matriz_int.append(matriz_final_int)
        n = matriz_final_float.shape[1]
        array1, array2 = matriz_final_float, matriz_final_int
        ##############################################
        ##### -> Treinamento da Rede Neural ##########
        i0 = 0
        while i0 <= 10:
            models, metricas = Modelos.treinar_ou_retreinar(array1, array2, reset=True)
            #print(f'Modelo treinado com sucesso! {metricas}')
            if metricas['accuracy'] >= 0.60 and metricas['f1_score'] >= 0.70:
                print('Modelo atingiu precisão desejada, salvando modelo...')
                models.save('modelo_acumulado.keras')
                i0 = 11
            i0 += 1
        ##############################################
######################################################
            
    if i >= 240:
        #### -> Predição da Rede Neural ##############
        y_pred, y_prob = Modelos.prever(arrayodd, threshold=0.7)  # array1 com no mínimo 120 elementos
        order = 1 - y_prob
        if order >= 0.5:
            resultado = 1
        else:
            resultado = 0
        print(24*'*-')
        print(f'Proxima Entrada: {resultado}')
        print(24*'*-')
        ##############################################

    i += 1


