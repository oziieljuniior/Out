import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa
from sklearn.metrics import f1_score


# Acesso aos módulos internos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Vetores import AjustesOdds

from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable


class Modelos:
    @staticmethod
    def treinar_ou_retreinar(array1, array2, reset=False, modelo_path="modelo_acumulado.keras"):
        """
        Treina ou continua o treinamento de uma rede neural com entrada 2D.
        """
        X = np.array(array1)
        y = np.ravel(array2).astype(int)
        input_shape = (X.shape[1], 1)

        x_train = np.expand_dims(X[:int(0.7 * len(X))], -1).astype("float32")
        x_test = np.expand_dims(X[int(0.7 * len(X)):], -1).astype("float32")

        y_train_cat = keras.utils.to_categorical(y[:len(x_train)], 2)
        y_test_cat = keras.utils.to_categorical(y[len(x_train):], 2)

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        if not reset and os.path.exists(modelo_path):
            print("🔄 Carregando modelo existente...")
            model = keras.models.load_model(modelo_path, custom_objects={'F1Score': Modelos.F1Score()})
        else:
            print("🚀 Criando novo modelo...")
            model = keras.Sequential([
                keras.Input(shape=input_shape),
                layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),

                layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),

                layers.LSTM(64, return_sequences=False),
                layers.Dropout(0.3),

                layers.Dense(32, activation="swish"),
                layers.Dropout(0.2),

                layers.Dense(2, activation="softmax")
            ])
            model.compile(
                loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.9, gamma=2.0),
                optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
                metrics=[
                    "accuracy",
                    Precision(name="precision"),
                    Recall(name="recall")
                ]
            )

        model.fit(
            x_train, y_train_cat,
            batch_size=512,
            epochs=30 if not reset else 50,
            validation_split=0.2,
            class_weight=class_weight_dict,
            verbose=2
        )

        # Avaliação
        score = model.evaluate(x_test, y_test_cat, verbose=0)

        y_true = np.argmax(y_test_cat, axis=-1)
        y_pred = np.argmax(model.predict(x_test), axis=-1)
        f1 = f1_score(y_true, y_pred)

        print(f"\n🔍 Avaliação:")
        print(f"Loss: {score[0]:.4f}, Accuracy: {score[1]:.4f}, Precision: {score[2]:.4f}, Recall: {score[3]:.4f}, F1: {f1:.4f}")

        model.save(modelo_path)
        print(f"✅ Modelo salvo em {modelo_path}")

        return model, {
            "accuracy": float(score[1]),
            "precision": float(score[2]),
            "recall": float(score[3]),
            "f1_score": float(f1),
        }

    @staticmethod
    def prever(array1, modelo_path="modelo_acumulado.keras", taxa_esperada=0.3):
        """
        Realiza predição sobre várias entradas com threshold dinâmico baseado na taxa esperada de classe 1.
        Ex: Para 60 entradas e taxa 0.3 → marca como 1 os 18 maiores valores.
        """
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {modelo_path}")

        model = keras.models.load_model(modelo_path)
        ajustador = AjustesOdds(array1)
        X_pred = ajustador.transformar_entrada_predicao(array1)

        y_probas = model.predict(X_pred)[:, 1]  # Probabilidade da classe 1

        # Threshold dinâmico: seleciona os top-k maiores
        qtd_1s = int(len(y_probas) * taxa_esperada)
        indices_top_k = np.argsort(y_probas)[-qtd_1s:]
        y_pred = np.zeros_like(y_probas, dtype=int)
        y_pred[indices_top_k] = 1

        print(f"📊 Taxa esperada: {taxa_esperada:.2f} → Previsto {np.sum(y_pred)} valores como 1.")
        print(f"ℹ️ Valores previstos (parcial):\n{y_pred[:10]}")
        print(f"ℹ️ Probabilidades (parcial):\n{y_probas[:10]}")

        return y_pred, y_probas

