# Libs
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Placares import Placar  # Importando a classe Placar do módulo Placares
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

# ==== NOVO: Bandit Contextual (agente) ====
from Modulos.Rl import BanditAgent, BanditConfig


def achar_threshold(y_true, proba, beta=1.0):
    melhor, thr_best = -1, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (proba >= thr).astype(int)
        f = f1_score(y_true, pred, zero_division=0, average="binary")
        if f > melhor:
            melhor, thr_best = f, thr
    return thr_best

### (Opcional) Modelo base inicial — será sobrescrito no treino periódico abaixo
logreg = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=50_000,
        random_state=42,
        warm_start=False,
    ),
)

### Carregar data
data = pd.read_csv('/home/darkcover01/Documentos/Out/Documentos/dados/odds_200k.csv')

array1, i = [], 0

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

## Variáveis para salvar em um dataframe
data_matriz_float, data_matriz_int, array_geral_float, historico_janelas = [], [], [], []
df_metricas_treino = pd.DataFrame(columns=[
    "rodada", "i", "modelo", "accuracy", "precision", "recall", "f1_score",
    "precision 0", "precision 1", "recall 0", "recall 1", "f1_score 0", "f1_score 1"
])
df_acuracia = pd.DataFrame(columns=["Iteração", "Precisão Geral", "Precisão Modelo"])
datasave = pd.DataFrame({'Modelo N': [], 'Reportes Modelos': []})
placar = Placar()  # Inicializando o placar
vetores = AjustesOdds(array1)  # Inicializando a classe de ajustes de odds

# ==== NOVO: inicializa agente bandit (classe operacional: odds > limiar) ====
LIMIAR_ODD = 3.0
agent = BanditAgent(
    cfg=BanditConfig(C_FP=4.0, G_TP=1.0, gamma=0.999, target_selection=0.33),
    limiar_odd=LIMIAR_ODD,
    seed=123,
)

# Variáveis auxiliares
action_atual = 0   # ação final (0/1) da rodada atual (decisão para a PRÓXIMA odd)
resultado = 0      # mantido por compatibilidade com o Placar
thr = 0.5          # será recalibrado no primeiro treino (i==600)

### Produção
while i <= 210000:
    print(24*'---')
    print(f'Rodada - {i}')

    ######## -> Vetor de Entradas Unidimensional ##########
    arrayodd, odd = vetores.coletarodd(i, inteiro, data, alavanca=False)
    array_geral_float.append(odd)

    if odd == 0:
        # flush opcional da decisão pendente com a última odd real
        agent.flush_pending_with(odd_current=float(array_geral_float[-2])) if len(array_geral_float) > 1 else None
        break
    ######################################################

    ######## -> Placar ###################################
    # Observação: o placar compara a decisão passada (resultado) com a odd ATUAL —
    # isso já implementa o atraso de 1 passo, e é compatível com o agente.
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

    ######## -> Treinamento do Modelo Base ###############
    if i >= 6000 and (i % 600) == 0:
        print('***'*20)
        print(f'Carregando dados ...')
        matriz_final_float, matriz_final_int = vetores.tranforsmar_final_matriz(arrayodd)
        print(f'Matrix: {[matriz_final_float.shape, matriz_final_int.shape]}')
        data_matriz_float.append(matriz_final_float), data_matriz_int.append(matriz_final_int)
        n = matriz_final_float.shape[1]
        array1, array2 = matriz_final_float, matriz_final_int

        # Treino/validação simples (holdout 80/20)
        X = pd.DataFrame(array1)
        y = array2.flatten()
        cut = int(0.8 * len(X))
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y[:cut], y[cut:]

        # Modelo linear base (regressão logística)
        #logreg = LogisticRegression(max_iter=10_000, C=1.0, class_weight='balanced', warm_start=True, random_state=42)
        logreg.fit(X_train, y_train)

        proba_val = logreg.predict_proba(X_test)[:, 1]
        thr = achar_threshold(y_test, proba_val, beta=1.0)  # limiar ótimo para a SUA convenção de y

        y_pred_lr = logreg.predict(X_test)
        report = classification_report(y_test, y_pred_lr, output_dict=True)

        print("Modelo Linear - Regressão Logística")
        print(classification_report(y_test, y_pred_lr))
        df_metricas_treino.loc[len(df_metricas_treino)] = {
            "rodada": (i // 6000) - 3,
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
            "f1_score 1": report["1"]["f1-score"],
        }

        from sklearn.metrics import roc_auc_score, average_precision_score
        roc = roc_auc_score(y_test, proba_val)
        pr  = average_precision_score(y_test, proba_val)
        
        print(f'ROC AUC: {roc:.4f} | PR AUC: {pr:.4f}')

    ######################################################

    ######## -> Predição + Bandit (gating) ###############
    if i >= 6000:
        print(24*'*-')
        Apredicao = vetores.transformar_entrada_predicao(arrayodd)   # (1, d_base)

        # Probabilidade do modelo base PARA A SUA CONVENÇÃO de y (no seu batch, y=1 quando odd < 3)
        proba_pred_lr = logreg.predict_proba(Apredicao)[:, 1]        # P(y=1) = P(odd < 3)
        yhat_lr = (proba_pred_lr >= thr).astype(int)                 # classe prevista segundo a SUA convenção

        # Converte para classe operacional: model_gt1 = 1 se o modelo sugere "odd > LIMIAR_ODD"
        model_gt1 = int(1 - yhat_lr[0])                              # invertendo a convenção do batch
        proba_gt1 = float(1.0 - proba_pred_lr[0])                    # P(odd > LIMIAR_ODD)

        # Monta o contexto do bandit: base + (proba_gt1, model_gt1)
        x_base = Apredicao.ravel().astype(np.float32)
        final_action, info = agent.step(
            x_base=x_base,
            proba_pred=proba_gt1,
            model_gt1=model_gt1,
            odd_current=float(odd),
            extras=None,               # adicione aqui suas features de histórico se desejar
            window_for_calib=600,
        )

        # Resultado final (decisão para a PRÓXIMA odd)
        resultado = final_action

        # Logs amigáveis
        print(
            f"p_model(> {LIMIAR_ODD:.1f})={proba_gt1:.3f} | sugestao_modelo={model_gt1} | "
            f"p_TS={info['p_TS']:.3f} | thr_util={info['threshold_util']:.3f} | FINAL={resultado} \n"
            f"acc={info['accuracy']:.3f} | prec={info['precision']:.3f} | rec={info['recall']:.3f} | "
            f"fpr={info['fpr']:.3f} | sel={info['selection_rate']:.3f}"
        )
        print(24*'*-')
    ######################################################

    i += 1

# Flush final (se houver decisão pendente)
if len(array_geral_float) >= 1:
    agent.flush_pending_with(odd_current=float(array_geral_float[-1]))

# Persistência de métricas
df_metricas_treino.to_csv('metricas_treino.csv', index=False)
df_acuracia.to_csv('acuracia.csv', index=False)
