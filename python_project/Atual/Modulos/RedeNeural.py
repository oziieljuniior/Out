import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score


# Acesso aos módulos internos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Vetores import AjustesOdds

from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable


class Modelos:
    @staticmethod
    @register_keras_serializable()
    def focal_loss(gamma=2., alpha=0.1):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = alpha * tf.pow(1 - y_pred, gamma)
            loss = weight * cross_entropy
            return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return focal_loss_fixed

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
            model = keras.models.load_model(modelo_path, custom_objects={"focal_loss_fixed": Modelos.focal_loss(alpha=0.1, gamma=2.0)})
        else:
            print("🚀 Criando novo modelo...")
            model = keras.Sequential([
                keras.Input(shape=input_shape),
                layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),

                layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),

                layers.LSTM(128, return_sequences=False),
                layers.Dropout(0.3),

                layers.Dense(64, activation="swish"),
                layers.Dropout(0.2),

                layers.Dense(2, activation="softmax")
            ])
            model.compile(
                loss= Modelos.focal_loss(alpha=0.1, gamma=2.0),
                optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
                metrics=[
                    "accuracy",
                    Precision(name="precision"),
                    Recall(name="recall")
                ]
            )

        model.fit(
            x_train, y_train_cat,
            batch_size=1024,
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

        model.save(modelo_path, include_optimizer=False)
        print(f"✅ Modelo salvo em {modelo_path}")

        return model, {
            "accuracy": float(score[1]),
            "precision": float(score[2]),
            "recall": float(score[3]),
            "f1_score": float(f1),
        }

    @staticmethod
    def prever(array1, modelo_path="modelo_acumulado.keras", threshold=0.5):
        """
        Realiza predição com base nas últimas 60 entradas.
        """
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {modelo_path}")

        model = keras.models.load_model(modelo_path, custom_objects={"focal_loss_fixed": Modelos.focal_loss(alpha=0.1, gamma=2.0)})
        ajustador = AjustesOdds(array1)
        X_pred = ajustador.transformar_entrada_predicao(array1)
        y_proba = model.predict(X_pred)[0][1]
        y_pred = int(y_proba > threshold)

        print(f"📈 Probabilidade classe 1: {y_proba:.4f}")
        return y_pred, y_proba
    
    