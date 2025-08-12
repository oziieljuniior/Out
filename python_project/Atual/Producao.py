# Libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Placares import Placar # Importando a classe Placar do módulo Placares
from Modulos.Vetores import AjustesOdds

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
    StandardScaler(with_mean=False),
    LogisticRegression(
        penalty="l2",
        C=0.1,                 # ajuste via grid depois
        solver="lbfgs",        # estável p/ l2
        class_weight="balanced",  # funciona bem p/ 30/70; para 50/50 pode remover
        max_iter=50_000,       # 1e6 é desnecessário; se não convergir, é escala/colinearidade
        random_state=42,
        warm_start=False       # raramente ajuda na LR
    )
)


### Carregar data
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
    if i >= 6001:
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
    if i >= 6000 and (i % 600) == 0:
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
        cut = int(0.8 * len(X))
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y[:cut], y[cut:]

        # 3. Modelo linear base (regressão logística)
        #logreg = LogisticRegression(max_iter=10_000, C=1.0,class_weight='balanced', warm_start=True, random_state=42)
        logreg.fit(X_train, y_train)

        proba_val = logreg.predict_proba(X_test)[:,1]
        thr = achar_threshold(y_test, proba_val, beta=1.0)

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

        from sklearn.metrics import roc_auc_score, average_precision_score

        roc = roc_auc_score(y_test, proba_val)
        pr  = average_precision_score(y_test, proba_val)

        print(f'ROC AUC: {roc:.4f} | PR AUC: {pr:.4f}')
        # Salve no df_metricas_treino:
        # "roc_auc": roc, "pr_auc": pr

        ##############################################
######################################################
            
    if i >= 6000:
        #### -> Predição da Modelo ##############
        print(24*'*-')
        Apredicao = vetores.transformar_entrada_predicao(arrayodd)
        #print(f'Predição: {type(Apredicao)} | {len(Apredicao)}')

        proba_pred = logreg.predict_proba(Apredicao)[:,1]
        res = (proba_pred >= thr).astype(int)

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
