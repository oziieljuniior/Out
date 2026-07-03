import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Importações locais do mesmo diretório Refatorado
from Placares import Placar, Placar60 
from Vetores import AjustesOdds

import time

class VovkAzouryWarmuthRegression:
    """
    Regressão Online baseada no algoritmo de Vovk-Azoury-Warmuth (VAW).
    Usa a fórmula de Sherman-Morrison para atualizar a matriz inversa A^-1 em O(d^2).
    Apresenta limites de regret logarítmicos ótimos O(d ln n) sob perda quadrática.
    """
    def __init__(self, input_dim: int, reg_param: float = 1.0):
        self.reg_param = reg_param
        self.d = input_dim
        # A_inv é inicializada como (1 / reg_param) * I
        self.A_inv = np.eye(input_dim) / reg_param
        # b é a soma cumulativa de y_s * x_s
        self.b = np.zeros(input_dim)
        # Cache para atualizar b no início do próximo passo predict()
        self.prev_x = None
        self.prev_y = None

    def predict(self, x: np.ndarray) -> float:
        """
        Realiza a predição para o vetor de recursos x_t atual.
        Consome os feedbacks pendentes do passo anterior para atualizar b_t.
        """
        # 1. Atualização pendente de b com o feedback do passo anterior
        if self.prev_x is not None and self.prev_y is not None:
            self.b += self.prev_y * self.prev_x
            self.prev_x = None
            self.prev_y = None

        # 2. Computa temporariamente a matriz A_t^-1 incluindo o x_t atual
        A_inv_t = self._update_inverse(self.A_inv, x)

        # 3. Calcula o vetor de pesos w_hat = A_t^-1 * b_t
        w_hat = np.dot(A_inv_t, self.b)

        # 4. Predição: y_hat = w_hat . x_t
        y_hat = np.dot(w_hat, x)
        return float(np.clip(y_hat, 0.0, 1.0))

    def predict_eval(self, x: np.ndarray) -> float:
        """
        Predição limpa para fins de avaliação/métricas. 
        Não altera o estado interno do modelo.
        """
        b_temp = self.b.copy()
        if self.prev_x is not None and self.prev_y is not None:
            b_temp += self.prev_y * self.prev_x
        A_inv_t = self._update_inverse(self.A_inv, x)
        w_hat = np.dot(A_inv_t, b_temp)
        y_hat = np.dot(w_hat, x)
        return float(np.clip(y_hat, 0.0, 1.0))

    def update_feedback(self, x: np.ndarray, y: float):
        """
        Recebe o rótulo real y_t e atualiza permanentemente a matriz inversa A_inv.
        Salva o x e y atuais para atualizar b_t no início do próximo predict().
        """
        self.A_inv = self._update_inverse(self.A_inv, x)
        self.prev_x = x
        self.prev_y = y

    def _update_inverse(self, A_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Atualiza a matriz inversa usando Sherman-Morrison.
        """
        num = np.dot(A_inv, x)
        den = 1.0 + np.dot(x, num)
        return A_inv - np.outer(num, num) / den

# Resolução de caminhos dinâmicos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(BASE_DIR, "..", "Atual", "data_treino", "Vitoria1_10 - 11-07-25_teste1.csv"))
output_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Documentos", "dados", "DoIt", "Order1.csv"))

# Carregar dados históricos
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError(f"Arquivo de dados não encontrado em: {data_path}")

# Inicializações de variáveis de controle e simulação
i = 0
array2s, array2n = [], []
array_geral_float = []
ordem_natural_count = 0

inteiro = int(input("Insira a entrada até onde o modelo deve ser carregado --> "))

# Instanciação dos módulos refatorados
vetores = AjustesOdds(array_geral_float, window_size=60)
placar = Placar(offset=360)
placar60_inst = Placar60()
media_parray = placar60_inst.media_parray
placar_dict = {'i_sintetico': 0, 'Contador_Geral': 0, 'Precisao_Geral': 0.0, 'Precisao_Sintetica': 0.0}

modelo = None
scaler = None
resultado = 0

# Loop principal de simulação sequencial
while i <= 210000:
    print(24 * '---')
    if len(media_parray) < 59:
        m = 0.0
    else:
        m = media_parray[len(media_parray) - 60]

    print(f'Número da Entrada - {i} | Acuracia_{(i % 60) + 1}: {round(m, 4)} | Contagem Geral: {placar_dict["i_sintetico"]} \nOrdem Natural: {ordem_natural_count}')
    
    # Coleta de Odds
    array2s, array2n, odd = vetores.coletarodd_rede_neural(i, inteiro, data, array2s, array2n)
    array_geral_float.append(odd)
    if odd == 0:
        break

    if odd >= 3:
        ordem_natural_count += 1

    # Atualização de placares e acompanhamento de métricas
    if i >= 361 and inteiro:
        print(24 * "-'-")
        placar_dict = placar.atualizar_geral(i, resultado, odd)
        media_parray = placar60_inst.atualizar(i, resultado, odd)
        
        core11 = 60 if i % 60 == 0 else (i % 60)
        print(f'Acuracia modelo Geral: {round(placar_dict["Precisao_Geral"], 4)} | Acuracia_{core11}: {round(media_parray[-1], 4)} \nPrecisao modelo Geral: {round(placar_dict["Precisao_Sintetica"], 4)}')
        print(24 * "-'-")

    # Inicialização do Modelo e Pré-treino sequencial no passo 360
    if i == 360:
        matriz_final_float, matriz_final_int = vetores.transformar_final_matriz(array_geral_float)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(matriz_final_float)
        y_train = np.ravel(matriz_final_int)
        
        print("\n[VovkAzouryWarmuth] Inicializando pesos VAW (Ridge Online) e realizando pré-treino...")
        modelo = VovkAzouryWarmuthRegression(input_dim=X_train.shape[1], reg_param=1.0)
        # Pré-treino sequencial O(d^2) simulando a chegada dos dados
        for t_idx in range(len(X_train)):
            x_sample = X_train[t_idx]
            y_sample = y_train[t_idx]
            modelo.predict(x_sample)
            modelo.update_feedback(x_sample, y_sample)

    # Avaliação estatística periódica a cada 60 passos
    if i >= 360 and (i % 60) == 0:
        matriz_final_float, matriz_final_int = vetores.transformar_final_matriz(array_geral_float)
        X_all = scaler.transform(matriz_final_float)
        y_all = np.ravel(matriz_final_int)
        
        split_index = int(len(X_all) * 0.7)
        X_test = X_all[split_index:]
        y_test = y_all[split_index:]
        
        # Predições no conjunto de teste usando o método de avaliação livre de efeitos colaterais
        p_test = np.array([modelo.predict_eval(x) for x in X_test])
        loss_test = -np.mean(y_test * np.log(p_test + 1e-15) + (1.0 - y_test) * np.log(1.0 - p_test + 1e-15))
        y_pred_test = (p_test >= 0.5).astype(int)
        accuracy_test = np.mean(y_pred_test == y_test)
        
        tp = np.sum((y_pred_test == 1) & (y_test == 1))
        fp = np.sum((y_pred_test == 1) & (y_test == 0))
        fn = np.sum((y_pred_test == 0) & (y_test == 1))
        
        precision_test = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_test = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        print(f"\n🔍 Resultados VAW no Teste (i={i}):")
        print(f"  Loss: {loss_test:.4f}")
        print(f"  Accuracy: {accuracy_test:.4f}")
        print(f"  Precision: {precision_test:.4f}")
        print(f"  Recall: {recall_test:.4f}")

    # Previsão da próxima entrada e atualização online contínua
    if i >= 360:
        print(24 * '*-')
        Apredicao = vetores.transformar_entrada_predicao(array_geral_float)
        x_scaled = scaler.transform(Apredicao)[0]
        
        # Realiza a predição usando o modelo VAW
        p = modelo.predict(x_scaled)
        
        # Rótulo de saída segue o mapeamento original (0 se p >= 0.5, senão 1)
        if p >= 0.5:
            resultado = 0
        else:
            resultado = 1
        
        print(f'Proxima Entrada (P={p:.4f}): {resultado}')
        print(24 * '*-')

        # Feedback e atualização online imediata baseada na perda observada
        y_true = 0.0 if odd >= 3 else 1.0
        modelo.update_feedback(x_scaled, y_true)

    i += 1

# Salvando resultados
print(f'Ordenando e salvando os dados...')
data_array_float = np.array(array2s)
data_array_int = np.array(array2n)

# Garantindo a existência do diretório de saída
os.makedirs(os.path.dirname(output_path), exist_ok=True)

data_final = pd.DataFrame({'Entrada': data_array_float, 'Resultado': data_array_int})
data_final.to_csv(output_path, index=False)
print(f'Simulação concluída. Resultados persistidos em: {output_path}')
