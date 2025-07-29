# Libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Placares import Placar # Importando a classe Placar do módulo Placares
from Modulos.Vetores import AjustesOdds
from Modulos.RedeNeural import Modelos

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import time

import matplotlib.pyplot as plt


### Carregar Gráfico --------------------------------------------
plt.ion()                       # ativa interação
fig, ax = plt.subplots()
ax.set_xlabel("Iteração")
ax.set_ylabel("Precisão (%)")
ax.set_ylim(0, 100)
line_geral, = ax.plot([], [], label="Precisão Geral")
line_modelo, = ax.plot([], [], label="Precisão Modelo")
ax.legend(loc="lower right")
x_data, y_geral, y_modelo = [], [], []

# ---------------------------------------------------------------


### Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
#/home/darkcover/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10 - 11-07-25_teste1.csv
#/home/darkcover/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10/Vitoria1_10 - game_teste3x1.csv
#/home/darkcover01/Documentos/Out/Documentos/dados/odds_200k.csv
#/home/darkcover01/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10 - 11-07-25_teste1.csv', usecols=["Entrada"], dtype=str
data = pd.read_csv('/home/darkcover01/Documentos/Out/Documentos/dados/odds_200k.csv')

array1, i = [], 0

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

## Variáveis para salvar em um dataframe
data_matriz_float, data_matriz_int, array_geral_float, historico_janelas = [], [], [], [] 
df_metricas_treino = pd.DataFrame(columns=["rodada", "modelo", "accuracy", "precision", "recall", "f1_score", "precision 0", "precision 1", "recall 0", "recall 1", "f1_score 0", "f1_score 1"])
df_acuracia = pd.DataFrame(columns=["Iteração", "Precisão Geral", "Precisão Modelo"])
datasave = pd.DataFrame({'Modelo N': [], 'Reportes Modelos': []})
placar = Placar()  # Inicializando o placar
vetores = AjustesOdds(array1)  # Inicializando a classe de ajustes de odds
Modelos = Modelos()


### Produção
while i <= 210000:
    print(24*'---')
    print(f'Rodada - {i}')

######## -> Vetor de Entradas Unidimensional ##########        
    arrayodd, odd = vetores.coletarodd(i, inteiro, data, alavanca=False)
    array_geral_float.append(odd)

    if odd == 0:
        break
######################################################

######## -> Placar ###################################      
    if i >= 24001:
        print(24*"-'-")
        array_placar = placar.atualizar_geral(i, resultado, odd)
        print(f'Precisão Geral: {array_placar["Precisao_Geral"]:.2f}% \nPrecisão Modelo: {array_placar["Precisao_Sintetica"]:.2f}%')
        x_data.append(i)                                   # ou rodada, se preferir
        y_geral.append(array_placar["Precisao_Geral"])
        y_modelo.append(array_placar["Precisao_Sintetica"])

        line_geral.set_data(x_data, y_geral)
        line_modelo.set_data(x_data, y_modelo)
        ax.relim()                 # recalcula limites
        ax.autoscale_view()        # aplica limites
        plt.pause(0.01)            # deixa o evento de GUI atualizar

        df_acuracia.loc[len(df_acuracia)] = {
            "Iteração": i,
            "Precisão Geral": array_placar["Precisao_Geral"],
            "Precisão Modelo": array_placar["Precisao_Sintetica"]
        }
        print(24*"-'-")
######################################################

######## -> Treinamento da Modelo ###############
    if i >= 24000 and (i % 120) == 0:
        print('***'*20)
        ##### -> Vetores de Entradas #################
        print(f'Carregando dados ...')
        matriz_final_float, matriz_final_int = vetores.tranforsmar_final_matriz(arrayodd)
        print(f'Matrix: {[matriz_final_float.shape, matriz_final_int.shape]}')
        data_matriz_float.append(matriz_final_float), data_matriz_int.append(matriz_final_int)
        n = matriz_final_float.shape[1]
        array1, array2 = matriz_final_float, matriz_final_int
        ##############################################
        ##### -> Treinamento do modelo ##########
        X = pd.DataFrame(array1)  # suas features
        y = array2.flatten()      # saída binária

        # 2. Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logreg, dicionario = Modelos.treinar_ou_retreinar(X_train, y_train)
        
        ##############################################
######################################################
            
    if i >= 24000:
        #### -> Predição da Modelo ##############
        print(24*'*-')
        #Apredicao = vetores.transformar_entrada_predicao(arrayodd)
        #print(f'Predição: {type(Apredicao)} | {len(Apredicao)}')
        res = Modelos.prever(arrayodd)
        if res[0] == 1:
            resultado = 0
        else:
            resultado = 1
        
        print(f'Proxima Entrada: {resultado}')
        print(24*'*-')
        ##############################################

    i += 1


df_acuracia.to_csv('acuracia.csv', index=False)

plt.ioff()
plt.savefig("evolucao_precisao.png", dpi=150)
plt.show()
