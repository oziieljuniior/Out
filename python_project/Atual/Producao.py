# Libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Placares import Placar # Importando a classe Placar do módulo Placares
from Modulos.Placares5x import Placar as Placar5x  # Importando a classe Placar5x do módulo Placares5x
from Modulos.Vetores import AjustesOdds as Vetor3x
from Modulos.Vetores5x import AjustesOdds as Vetor5x  # Importando a classe Vetores5x do módulo Vetores5x
from Modulos.Vetores2x import AjustesOdds as Vetor2x  # Importando a classe Vetores2x do módulo Vetores2x
from Modulos.Placares2x import Placar as Placar2x  # Importando a classe Placar2x do módulo Placares2x

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import time

from sklearn.metrics import f1_score

def achar_threshold(y_true, proba, beta=1.0):
    melhor, thr_best = -1, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (proba >= thr).astype(int)
        f = f1_score(y_true, pred, zero_division=0, average="binary")
        if f > melhor:
            melhor, thr_best = f, thr
    return thr_best

### Carregar Modelo
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

logreg2 = make_pipeline(
    LogisticRegression(
        max_iter=50000,             # começa “raso” para convergir rápido
        warm_start=False,   
        C=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)


### Carregar data
data = pd.read_csv('/home/darkcover01/Documentos/Out/Documentos/dados/odds_200k.csv')

array1, i = [], 0

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

## Variáveis para salvar em um dataframe
data_matriz_float, data_matriz_int, array_geral_float, historico_janelas = [], [], [], [] 
arrayodd, arrayodd1 = [], []
df_metricas_treino = pd.DataFrame(columns=["rodada", "modelo", "accuracy", "precision", "recall", "f1_score", "precision 0", "precision 1", "recall 0", "recall 1", "f1_score 0", "f1_score 1"])
df_acuracia = pd.DataFrame(columns=["Iteração", "Precisão Geral", "Precisão Modelo"])
datasave = pd.DataFrame({'Modelo N': [], 'Reportes Modelos': []})
placar, placar5x, placar2x = Placar(), Placar5x(), Placar2x()  # Inicializando o placar
vetores, vetores1, vetores2 = Vetor3x(array1), Vetor5x(array1), Vetor2x(array1)  # Inicializando a classe de ajustes de odds


### Produção
while i <= 210000:
    print(24*'---')
    print(f'Rodada - {i}')

######## -> Vetor de Entradas Unidimensional ##########        
    arrayodd, odd = vetores.coletarodd(i, inteiro, data, alavanca=False)
    #arrayodd1, odd = vetores1.coletarodd(i, inteiro, data, alavanca=False)
    array_geral_float.append(odd)

    if odd == 0:
        break
######################################################

######## -> Placar ###################################      
    if i >= 6001:
        print(24*"-'-")
        array_placar = placar.atualizar_geral(i, resultado, odd)
        array_placar2x = placar2x.atualizar_geral(i, resultado, odd)
        array_placar5x = placar5x.atualizar_geral(i, resultado5x, odd)
        print(f'Precisão Geral 2x: {array_placar2x["Precisao_Geral"]:.2f}% | Precisão Modelo 2x: {array_placar2x["Precisao_Sintetica"]:.2f}%')
        print(f'Precisão Geral 3x: {array_placar["Precisao_Geral"]:.2f}% | Precisão Modelo 3x: {array_placar["Precisao_Sintetica"]:.2f}%')
        print(f'Precisão Geral 5x: {array_placar5x["Precisao_Geral"]:.2f}% | Precisão Modelo 5x: {array_placar5x["Precisao_Sintetica"]:.2f}%')
        df_acuracia.loc[len(df_acuracia)] = {
            "Iteração": i,
            "Precisão Geral 3x": array_placar["Precisao_Geral"],
            "Precisão Modelo 3x": array_placar["Precisao_Sintetica"],
            "Precisão Geral 5x": array_placar5x["Precisao_Geral"],
            "Precisão Modelo 5x": array_placar5x["Precisao_Sintetica"]
        }
        print(24*"-'-")
######################################################

######## -> Treinamento da Modelo ###############
    if i >= 6000 and (i % 1200) == 0:
        print('***'*20)
        ##### -> Vetores de Entradas #################
        print(f'Carregando dados ...')
        matriz_final_float, matriz_final_int = vetores.tranforsmar_final_matriz(arrayodd)
        matriz_final_float2x, matriz_final_int2x = vetores2.tranforsmar_final_matriz(arrayodd)
        matriz_final_float5x, matriz_final_int5x = vetores1.tranforsmar_final_matriz(arrayodd)
        print(f'Matrix: {[matriz_final_float.shape, matriz_final_int.shape]}')
        print(f'Matrix 2x: {[matriz_final_float2x.shape, matriz_final_int2x.shape]}')
        print(f'Matrix: {[matriz_final_float5x.shape, matriz_final_int5x.shape]}')

        array1, array2 = matriz_final_float, matriz_final_int
        array1_2x, array2_2x = matriz_final_float2x, matriz_final_int2x
        array1_5x, array2_5x = matriz_final_float5x, matriz_final_int5x
        
        ##############################################
        ##### -> Treinamento do modelo ##########
        X = pd.DataFrame(array1)  # suas features
        y = array2.flatten()      # saída binária
        X1 = pd.DataFrame(array1_5x)  # suas features
        y1 = array2_5x.flatten()      # saída binária
        X2 = pd.DataFrame(array1_2x)  # suas features
        y2 = array2_2x.flatten()      # saída binária

        # 2. Divisão treino/teste
        cut = int(0.8 * len(X))
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y[:cut], y[cut:]
        X_train5x, X_test5x = X1.iloc[:cut], X1.iloc[cut:]
        y_train5x, y_test5x = y1[:cut], y1[cut:]
        X_train2x, X_test2x = X2.iloc[:cut], X2.iloc[cut:]
        y_train2x, y_test2x = y2[:cut], y2[cut:]

        # 3. Modelo linear base (regressão logística)
        #logreg = LogisticRegression(max_iter=10_000, C=1.0,class_weight='balanced', warm_start=True, random_state=42)
        logreg.fit(X_train, y_train)
        logreg1.fit(X_train5x, y_train5x)
        logreg2.fit(X_train2x, y_train2x)

        proba_val = logreg.predict_proba(X_test)[:,1]
        thr = achar_threshold(y_test, proba_val, beta=1.0)
        proba_val1 = logreg1.predict_proba(X_test5x)[:,1]
        thr1 = achar_threshold(y_test5x, proba_val1, beta=1.0)
        proba_val2 = logreg2.predict_proba(X_test2x)[:,1]
        thr2 = achar_threshold(y_test2x, proba_val2, beta=1.0)

        y_pred_lr = logreg.predict(X_test)
        y_pred_lr1 = logreg1.predict(X_test5x)
        y_pred_lr2 = logreg2.predict(X_test2x)
        
        report = classification_report(y_test, y_pred_lr, output_dict=True)
        report1 = classification_report(y_test5x, y_pred_lr1, output_dict=True)
        report2 = classification_report(y_test2x, y_pred_lr2, output_dict=True)

        print("Modelo Linear 2x - Regressão Logística")
        print(classification_report(y_test2x, y_pred_lr2))
        
        print("Modelo Linear 3x - Regressão Logística")
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
        
        print("Modelo Linear 5x - Regressão Logística")
        print(classification_report(y_test5x, y_pred_lr1))
        
        from sklearn.metrics import roc_auc_score, average_precision_score

        roc = roc_auc_score(y_test, proba_val)
        pr  = average_precision_score(y_test, proba_val)
        roc1 = roc_auc_score(y_test5x, proba_val1)
        pr1  = average_precision_score(y_test5x, proba_val1)
        roc2 = roc_auc_score(y_test2x, proba_val2)
        pr2  = average_precision_score(y_test2x, proba_val2)
        
        print(f'ROC AUC 2x: {roc2:.4f} | PR AUC 2x: {pr2:.4f}')
        print(f'ROC AUC 3x: {roc:.4f} | PR AUC 3x: {pr:.4f}')
        print(f'ROC AUC 5x: {roc1:.4f} | PR AUC 5x: {pr1:.4f}')
        # Salve no df_metricas_treino:
        # "roc_auc": roc, "pr_auc": pr

        ##############################################
######################################################
            
    if i >= 6000:
        #### -> Predição da Modelo ##############
        print(24*'*-')
        Apredicao = vetores.transformar_entrada_predicao(arrayodd)
        Apredicao2x = vetores2.transformar_entrada_predicao(arrayodd)
        Apredicao5x = vetores1.transformar_entrada_predicao(arrayodd)
        #print(f'Predição: {type(Apredicao)} | {len(Apredicao)}')

        proba_pred = logreg.predict_proba(Apredicao)[:,1]
        proba_pred5x = logreg1.predict_proba(Apredicao5x)[:,1]
        proba_pred2x = logreg2.predict_proba(Apredicao2x)[:,1]
        res = (proba_pred >= thr).astype(int)
        res5x = (proba_pred5x >= thr1).astype(int)
        res2x = (proba_pred2x >= thr2).astype(int)
        
        if res2x[0] == 1:
            resultado2x = 0
        else:
            resultado2x = 1

        if res[0] == 1:
            resultado = 0
        else:
            resultado = 1
        if res5x[0] == 1:
            resultado5x = 0
        else:
            resultado5x = 1
        
        print(f'Proxima Entrada 2x: {resultado2x} | Proxima Entrada 3x: {resultado} | Proxima Entrada 5x: {resultado5x}')
        print(24*'*-')
        
        print(f'Predições Futuras')
        array_futuro2x, array_futuro3x, array_futuro5x = arrayodd.copy(), arrayodd.copy(), arrayodd.copy()
        resultado2x1, resultado3x1, resultado5x1 = resultado2x, resultado, resultado5x
        for i in range(4):
            if resultado2x1 == 1:
                valorsintetico2x1 = 2.5
            else:
                valorsintetico2x1 = 1.45
            
            if resultado == 1:
                valorsintetico3x1 = 4
            else:
                valorsintetico3x1 = 1.75
            
            if resultado == 1:
                valorsintetico5x1 = 6
            else:
                valorsintetico5x1 = 2
            
            array_futuro2x.append(valorsintetico2x1), array_futuro3x.append(valorsintetico3x1), array_futuro5x.append(valorsintetico5x1)    
            
            Apredicao = vetores.transformar_entrada_predicao(array_futuro2x)
            Apredicao2x = vetores2.transformar_entrada_predicao(array_futuro3x)
            Apredicao5x = vetores1.transformar_entrada_predicao(array_futuro5x)
            #print(f'Predição: {type(Apredicao)} | {len(Apredicao)}')

            proba_pred = logreg.predict_proba(Apredicao)[:,1]
            proba_pred5x = logreg1.predict_proba(Apredicao5x)[:,1]
            proba_pred2x = logreg2.predict_proba(Apredicao2x)[:,1]
            res = (proba_pred >= thr).astype(int)
            res5x = (proba_pred5x >= thr1).astype(int)
            res2x = (proba_pred2x >= thr2).astype(int)
            
            if res2x[0] == 1:
                resultado2x1 = 0
            else:
                resultado2x1 = 1

            if res[0] == 1:
                resultado3x1 = 0
            else:
                resultado3x1 = 1
            if res5x[0] == 1:
                resultado5x1 = 0
            else:
                resultado5x1 = 1
        
        print(f'Array Futuro2x: {array_futuro2x[-10:]} \nArray Futuro3x: {array_futuro3x[-10:]} \nArray Futuro5x: {array_futuro5x[-10:]}')
        
        print(12*'##') 
        ##############################################

    i += 1


df_metricas_treino.to_csv('metricas_treino.csv', index=False)
df_acuracia.to_csv('acuracia.csv', index=False)
