import os
import sys

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Placares import Placar, Placar60 # Importando a classe Placar e Placar60 do módulo Placares
from Modulos.Vetores import AjustesOdds

import skfuzzy as fuzz

# Libs
import time

# Métodos auxiliares locais removidos; agora importados de Modulos.Vetores e Modulos.Placares

def lista_predicao(i, t, modelos, array1, array2):
    """
    Gera uma lista com possíveis predições.
    """
    y_pred1 = []
    for sk in range(0,t):
        if modelos[sk] is not None:
            posicao = 60*sk + 60
            print(sk, posicao)
            matriz1s, matriz1n, posicao0 = tranforsmar_final_matriz(posicao,array1, array2)

            x_new = np.array(matriz1s[-1,1:])
            x_new = x_new.astype("float32")
            x_new = x_new.reshape(1, -1)
            
            predictions = modelos[sk].predict(x_new)

            y_pred = (predictions >= 0.5).astype(int)
            y_pred1.append(y_pred[0, 0])
    print(y_pred1)
    return y_pred1

# Classe personalizada de F1 Score
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Função principal
def reden(array1, array2):
    """
    Treina uma rede neural maximizando a detecção da classe 1, respeitando sequência, normalização e desbalanceamento.
    """
    # Convertendo entradas
    X = np.array(array1, dtype=np.float32)
    y = np.ravel(array2).astype(np.float32)
    
    split_index = int(len(X) * 0.7)
    X_train_raw, X_test_raw = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Normalização das features (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    input_shape = (X_train.shape[1],)

    # Pesos de classe para reforçar a classe 1 (se desejado usar em conjunto com binary crossentropy)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    # Modelo
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation=tf.keras.activations.swish),
        layers.Dense(1, activation="sigmoid"),  # O correto para classificação binária
    ])

    # Compilação
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        metrics=[
            'accuracy',
            Precision(name="precision"),
            Recall(name="recall")
        ]
    ) 

    # Parada Antecipada
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # Treinamento
    model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=150,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=2
    )

    # Avaliação
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🔍 Resultados no Teste:")
    print(f"Loss: {score[0]:.4f}")
    print(f"Accuracy: {score[1]:.4f}")
    print(f"Precision: {score[2]:.4f}")
    print(f"Recall: {score[3]:.4f}")

    return model, scaler


def ponderar_lista(lista, base=1.10):
    """
    Realiza uma ponderação dos elementos da lista com pesos exponenciais crescentes.

    Args:
        lista (list): Lista de inteiros contendo apenas 0 e 1.
        base (float): Base da função exponencial. Deve ser maior que 1.

    Returns:
        int: Resultado ponderado, 0 ou 1.
    """
    n = len(lista)
    if n == 0:
        raise ValueError("A lista não pode estar vazia.")

    # Calcular pesos exponenciais
    pesos = [base ** (n - i) for i in range(n)]

    # Calcular soma ponderada e total de pesos
    soma_ponderada = sum(elemento * peso for elemento, peso in zip(lista, pesos))
    total_pesos = sum(pesos)
    
    qrange = 1 / n

    # Retornar 1 se média ponderada >= 0.5, senão 0
    return 1 if soma_ponderada / total_pesos >= qrange else 0

# Lógica de matriz final local removida; integrada com os métodos da classe AjustesOdds

## Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
data = pd.read_csv('/home/darkcover/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10 - 11-07-25_teste1.csv')

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j2, j3 = 0,0,0,0

acerto2, acerto3, core1 = 0,0,0

modelos = [None]*5000
data_matrizes = [None]*5000
recurso1, recurso2 = [None]*5000, [None]*5000

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

data_matriz_float = []
data_matriz_int = []
data_array_float = []
data_array_int = []
array_geral_float = []

vetores = AjustesOdds(array_geral_float, window_size=60)
placar = Placar(offset=360)
placar60_inst = Placar60()
media_parray = placar60_inst.media_parray
placar_dict = {'i_sintetico': 0, 'Contador_Geral': 0, 'Precisao_Geral': 0.0, 'Precisao_Sintetica': 0.0}
ordem_natural_count = 0

while i <= 210000:
    print(24*'---')
    if len(media_parray) < 59:
        m = 0
        core1 = 0
    else:
        m = media_parray[len(media_parray) - 60]

    print(f'Número da Entrada - {i} | Acuracia_{(i % 60) + 1}: {round(m,4)} | Contagem Geral: {placar_dict["i_sintetico"]} \nOrdem Natural: {ordem_natural_count}')
    
    array2s, array2n, odd = vetores.coletarodd_rede_neural(i, inteiro, data, array2s, array2n)
    array_geral_float.append(odd)
    if odd == 0:
        break

    if odd >= 3:
        ordem_natural_count += 1

    if i >= 361 and inteiro:
        print(24*"-'-")
        
        placar_dict = placar.atualizar_geral(i, resultado, odd)
        media_parray = placar60_inst.atualizar(i, resultado, odd)
        
        core11 = 60 if i % 60 == 0 else (i % 60)
        print(f'Acuracia modelo Geral: {round(placar_dict["Precisao_Geral"],4)} | Acuracia_{core11}: {round(media_parray[-1],4)} \nPrecisao modelo Geral: {round(placar_dict["Precisao_Sintetica"],4)}')
        print(24*"-'-")

    if i >= 360 and (i % 60) == 0:
        matriz_final_float, matriz_final_int = vetores.tranforsmar_final_matriz(array_geral_float)
        modelo, scaler = reden(matriz_final_float, matriz_final_int)    

    if i >= 360:
        print(24*'*-')
        Apredicao = vetores.transformar_entrada_predicao(array_geral_float)
        Apredicao_scaled = scaler.transform(Apredicao)
        res = modelo.predict(Apredicao_scaled)
        if res[0, 0] >= 0.5:
            resultado = 0
        else:
            resultado = 1
        
        print(f'Proxima Entrada: {resultado}')
        print(24*'*-')


    i += 1

print(f'Ordenandos os dados ...')

data_array_float = np.array(array2s)
data_array_int = np.array(array2n)

data_final = pd.DataFrame({'Entrada': data_array_float, 'Resultado': data_array_int})
data_final.to_csv('/home/darkcover/Documentos/Out/Documentos/dados/DoIt/Order1.csv', index=False)



