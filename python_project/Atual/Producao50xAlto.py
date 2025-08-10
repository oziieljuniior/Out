# Libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.PlacaresAlto import Placar # Importando a classe Placar do módulo Placares
from Modulos.VetoresAlto import AjustesOdds

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import time

# --- 1) definir antes do while ---------------------------------
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logreg = make_pipeline(
    LogisticRegression(
        max_iter=1000000,             # começa “raso” para convergir rápido
        warm_start=True,   
        C=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)

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
    if i >= 12001:
        print(24*"-'-")
        array_placar = placar.atualizar_geral(i, resultado, odd)
        print(f'Precisão Geral: {array_placar["Precisao_Geral"]:.2f}% \nPrecisão Modelo: {array_placar["Precisao_Sintetica"]:.2f}%')
        
        df_acuracia.loc[len(df_acuracia)] = {
            "Iteração": i,
            "Precisão Geral": array_placar["Precisao_Geral"],
            "Precisão Modelo": array_placar["Precisao_Sintetica"]
        }
        print(24*"-'-")
######################################################

######## -> Treinamento da Modelo ###############
    if i >= 12000 and (i % 600) == 0:
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
        logreg.fit(X_train, y_train)     # treinamento inicial

        # 3. Modelo linear base (regressão logística)
        #logreg = LogisticRegression(max_iter=3000, C=1.0,class_weight='balanced', random_state=42)
        
        #logreg.fit(X_train, y_train)
        y_pred_lr = logreg.predict(X_test)
        
        report = classification_report(y_test, y_pred_lr, output_dict=True)

        print("Modelo Linear - Regressão Logística")
        print(classification_report(y_test, y_pred_lr))
        df_metricas_treino.loc[len(df_metricas_treino)] = {
            "rodada": (i // 600) - 3,  # Armazenando a rodada
            "i": i,
            "modelo": "Regressão Logística",
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "precision 0": report["0"]["precision"],
            "precision 1": report["1"]["precision"],
            "recall 0": report["0"]["recall"],
            "recall 1": report["1"]["recall"],
            "f1_score 0": report["0"]["f1-score"],
            "f1_score 1": report["1"]["f1-score"]
        }
    
        ##############################################
######################################################
            
    if i >= 12000:
        #### -> Predição da Modelo ##############
        print(24*'*-')
        Apredicao = vetores.transformar_entrada_predicao(arrayodd)
        #print(f'Predição: {type(Apredicao)} | {len(Apredicao)}')
        res = logreg.predict(Apredicao)
        if res[0] == 1:
            resultado = 0
        else:
            resultado = 1
        
        print(f'Proxima Entrada: {resultado}')
        print(24*'*-')
        ##############################################

    i += 1


df_metricas_treino.to_csv('metricas_treino.csv', index=False)
df_acuracia.to_csv('acuracia.csv', index=False)
