# Libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Placares import Placar # Importando a classe Placar do módulo Placares
from Modulos.Vetores import AjustesOdds

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

    
### Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
#/home/darkcover/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10 - 11-07-25_teste1.csv
#/home/darkcover/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10/Vitoria1_10 - game_teste3x1.csv
data = pd.read_csv('/home/darkcover01/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10 - 11-07-25_teste1.csv')

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
    if i >= 240 and (i % 60) == 0:
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
        X = pd.DataFrame(array1)  # suas features
        y = array2.flatten()      # saída binária

        # 2. Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit, cross_validate
        from sklearn.metrics import (classification_report, confusion_matrix,
                                    RocCurveDisplay, PrecisionRecallDisplay)
        # 1. Pipeline com pré-processamento e modelo
        pipe = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty='l2',
                                                C=1.0,          # ajustar via grid
                                                class_weight='balanced',
                                                max_iter=1000,
                                                random_state=42))

        tscv = TimeSeriesSplit(n_splits=5)

        cv_res = cross_validate(pipe, X, y,
                                cv=tscv,
                                scoring=['accuracy', 'f1', 'roc_auc'],
                                return_estimator=True)

        print("ACC média:", cv_res['test_accuracy'].mean())
        print("F1 média :", cv_res['test_f1'].mean())
        print("ROC-AUC  :", cv_res['test_roc_auc'].mean())
        
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:,1]
        # Escolher threshold
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(y_test, y_prob)
        # Ex.: threshold para recall classe 0 ≥ 0.80
        thr_opt = next(t for p,r,t in zip(prec, rec, thr) if r >= 0.80)
        y_pred = (y_prob >= thr_opt).astype(int)
        print(classification_report(y_test, y_pred))


        # 3. Modelo linear base (regressão logística)
        logreg = LogisticRegression(penalty='l2',
                                                C=1.0,          # ajustar via grid
                                                class_weight='balanced',
                                                max_iter=2000,
                                                random_state=42)
        logreg.fit(X_train, y_train)
        y_pred_lr = logreg.predict(X_test)

        print("Modelo Linear - Regressão Logística")
        print(classification_report(y_test, y_pred_lr))
        
        ##############################################
######################################################
            
    if i >= 240:
        #### -> Predição da Rede Neural ##############
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


