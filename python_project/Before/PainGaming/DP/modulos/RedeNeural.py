from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


class DP:
    def __init__(self, array1, array3, n):
        self.array1 = array1
        self.array3 = array3
        self.n = n
        
    def reden(self):
        """
        Função para treinar uma rede neural usando as entradas e saídas fornecidas.

        Args:
            array1 (numpy.array): Saídas (rótulos) binárias (0 ou 1).
            array3 (numpy.array): Entradas preditoras.
            m (int): Número de amostras.
            n (int): Número de características por amostra.

        Returns:
            keras.Model: Modelo treinado.
        """

        # Convertendo entradas para float e saídas para int
        array3 = np.array(self.array3, dtype=np.float32)
        array1 = np.array(self.array1, dtype=np.int32)
        n = self.n
        # Dividindo os dados em treino e teste
        X = array3
        y = array1
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalização dos dados
        x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
        x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

        # Ajustando dimensões para entrada no modelo
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        input_shape = (n - 1, 1)  # Formato esperado de entrada

        # Convertendo saídas para categóricas
        num_classes = 2
        y_train_cat = keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, num_classes)

        # Calculando pesos para balancear classes
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y_train)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

        # Definição do modelo
        model = keras.Sequential([
            keras.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ])

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy', Precision(name="precision"), Recall(name="recall")]
        )

        # Treinamento
        batch_size = 128
        epochs = 50
        history = model.fit(
            x_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            class_weight=class_weights_dict,
            verbose=1
        )

        # Salvando histórico de treinamento
        with open("training_history.txt", "w") as f:
            for epoch, metrics in enumerate(zip(history.history['loss'], history.history['accuracy'], 
                                                history.history['precision'], history.history['recall'])):
                f.write(f"Epoch {epoch + 1}: Loss={metrics[0]:.4f}, Accuracy={metrics[1]:.4f}, "
                        f"Precision={metrics[2]:.4f}, Recall={metrics[3]:.4f}\n")

        # Avaliação
        score = model.evaluate(x_test, y_test_cat, verbose=0)
        print(f"Test loss: {score[0]:.4f}")
        print(f"Test accuracy: {score[1]:.4f}")
        print(f"Precision: {score[2]:.4f}")
        print(f"Recall: {score[3]:.4f}")

        return [model, score[2]]