# Libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.PlacaresAlto import Placar # Importando a classe Placar do módulo Placares
from Modulos.VetoresAlto import AjustesOdds
from Modulos.VetoresAltoa import AjustesOdds as AjustesOddsA
from Modulos.PlacaresAltoa import Placar as PlacarA

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- 1) definir antes do while ---------------------------------
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logreg = make_pipeline(
    LogisticRegression(
        max_iter=50000,             # começa “raso” para convergir rápido
        warm_start=False,   
        C=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)

logreg1 = make_pipeline(
    LogisticRegression(
        max_iter=50000,             # começa “raso” para convergir rápido
        warm_start=False,   
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
placar, placar10x = Placar(), PlacarA()  # Inicializando o placar
vetores, vetores10x = AjustesOdds(array1), AjustesOddsA(array1)   # Inicializando a classe de ajustes de odds


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
        array_placar1 = placar10x.atualizar_geral(i, resultado10x, odd)
        print(f'Precisão Geral 15x: {array_placar["Precisao_Geral"]:.2f}% | Precisão Modelo 15x: {array_placar["Precisao_Sintetica"]:.2f}%')
        print(f'Precisão Geral 10x: {array_placar1["Precisao_Geral"]:.2f}% | Precisão Modelo 10x: {array_placar1["Precisao_Sintetica"]:.2f}%')
    
        df_acuracia.loc[len(df_acuracia)] = {
            "Iteração": i,
            "Precisão Geral": array_placar["Precisao_Geral"],
            "Precisão Modelo": array_placar["Precisao_Sintetica"]
        }
        print(24*"-'-")
######################################################

######## -> Treinamento da Modelo ###############
    if i >= 12000 and (i % 1200) == 0:
        print('***'*20)
        ##### -> Vetores de Entradas #################
        print(f'Carregando dados ...')
        matriz_final_float, matriz_final_int = vetores.tranforsmar_final_matriz(arrayodd)
        matriz_final_float10x, matriz_final_int10x = vetores10x.tranforsmar_final_matriz(arrayodd)
        print(f'Matrix15x: {[matriz_final_float.shape, matriz_final_int.shape]}')
        print(f'Matrix10x: {[matriz_final_float10x.shape, matriz_final_int10x.shape]}')
        
        data_matriz_float.append(matriz_final_float), data_matriz_int.append(matriz_final_int)
        
        array1, array2 = matriz_final_float, matriz_final_int
        array110x, array210x = matriz_final_float10x, matriz_final_int10x
        ##############################################
        ##### -> Treinamento do modelo ##########
        X = pd.DataFrame(array1)  # suas features
        y = array2.flatten()      # saída binária
        X10x = pd.DataFrame(array110x)  # suas features 10x
        y10x = array210x.flatten()      # saída binária 10x        

        # 2. Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logreg.fit(X_train, y_train)     # treinamento inicial
        X_train10x, X_test10x, y_train10x, y_test10x = train_test_split(X10x, y10x, test_size=0.2, random_state=42)
        logreg1.fit(X_train10x, y_train10x)  # treinamento inicial

        y_pred_lr = logreg.predict(X_test)
        y_pred_lr10x = logreg1.predict(X_test10x)
        
        report = classification_report(y_test, y_pred_lr, output_dict=True)

        print("Modelo Linear15x - Regressão Logística")
        print(classification_report(y_test, y_pred_lr))
        df_metricas_treino.loc[len(df_metricas_treino)] = {
            "rodada": (i // 1200) - 3,  # Armazenando a rodada
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
        print("Modelo Linear10x - Regressão Logística")
        print(classification_report(y_test10x, y_pred_lr10x))

        ##############################################
######################################################
            
    if i >= 12000:
        #### -> Predição da Modelo ##############
        print(24*'*-')
        Apredicao = vetores.transformar_entrada_predicao(arrayodd)
        Apredicao10x = vetores10x.transformar_entrada_predicao(arrayodd)
        
        res = logreg.predict(Apredicao)
        if res[0] == 1:
            resultado = 0
        else:
            resultado = 1
        
        res10x = logreg1.predict(Apredicao10x)
        if res10x[0] == 1:
            resultado10x = 0
        else:
            resultado10x = 1
        
        print(f'Proxima Entrada15x: {resultado} | Proxima Entrada10x: {resultado10x}')
        print(24*'*-')
        
        print(f'Predições Futuras')
        array_futuro10x, array_futuro15x = arrayodd.copy(), arrayodd.copy()
        resultado15x1, resultado10x1 = resultado, resultado10x
        for i in range(5):
            if resultado10x1 == 1:
                valorsintetico10x1 = 11
            else:
                valorsintetico10x1 = 2
            
            if resultado == 1:
                valorsintetico15x1 = 16
            else:
                valorsintetico15x1 = 2
            
            array_futuro10x.append(valorsintetico10x1), array_futuro15x.append(valorsintetico15x1)
            
            Apredicao_futuro10x, Apredicao_futuro15x = vetores10x.transformar_entrada_predicao(array_futuro10x), vetores.transformar_entrada_predicao(array_futuro15x)
            res = logreg.predict(Apredicao_futuro15x)
            if res[0] == 1:
                resultado15x1 = 0
            else:
                resultado15x1 = 1
            
            res10x = logreg1.predict(Apredicao10x)
            if res10x[0] == 1:
                resultado10x1 = 0
            else:
                resultado10x1 = 1
            print(i*'.')
        
        print(f'Array Futuro10x: {array_futuro10x[-10:]} \nArray Futuro15x: {array_futuro15x[-10:]}')
        
        print(12*'##')    
            
        
        
        ##############################################

    i += 1


df_metricas_treino.to_csv('metricas_treino.csv', index=False)
df_acuracia.to_csv('acuracia.csv', index=False)
