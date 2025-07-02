import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ====== Carregar e pré-processar dados ======
data = pd.read_csv('/home/darkcover/Documentos/Out/Documentos/dados/odds_200k.csv')
data_T1 = data.iloc[:50000]['Odd'].reset_index(drop=True)

# Tratar extremos
data_T2 = []
for odd in data_T1:
    if odd >= 2:
        data_T2.append(2)
    elif odd == 0:
        data_T2.append(1)
    else:
        data_T2.append(odd)

# Gerar matriz sequencial
def matriz(num_colunas, array1):
    if num_colunas > len(array1):
        raise ValueError("Número de colunas maior que tamanho do array.")
    return np.array([array1[i:i + num_colunas] for i in range(len(array1) - num_colunas + 1)])

matriz1 = matriz(60, data_T2)

# Entradas e saída
X_raw = matriz1[:, :-1]
y_raw = matriz1[:, -1]

# Binarização
X_bin = (X_raw >= 2).astype(int)
y_bin = (y_raw >= 2).astype(int)

# Redimensionar para entrada sequencial [batch, steps, features]
X_bin = np.expand_dims(X_bin, axis=-1)

# Divisão sequencial
X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, shuffle=False)

# ====== Modelo melhorado com Conv1D + Dense ======
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),
    Conv1D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='swish'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Avaliação
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)
print(classification_report(y_test, y_pred, digits=4))

# ====== Loop de previsão manual ======
odd = 1
order = []

print("\n🔍 Iniciando modo de previsão interativo (14 entradas binárias):")
while True:
    while len(order) < 59:
        odd = input(f"Insira a entrada #{len(order)+1}: ")
        if odd == "0":
            print("Encerrando...")
            exit()
        odd = float(odd.replace(',', '.'))
        order.append(2 if odd >= 2 else 0)

    input_seq = np.array(order).reshape(1, -1, 1)

    pred_prob = model.predict(input_seq)[0][0]
    pred_bin = int(pred_prob >= 0.5)
    print(f"🔮 Previsão (classe 1 se >= 2): {pred_bin} | Probabilidade: {pred_prob:.4f}")

    # Entrada real
    new_odd = input("Insira o valor real da próxima odd: ")
    new_odd = float(new_odd.replace(',', '.'))
    order.append(2 if new_odd >= 2 else 0)
    order = order[1:]
    print(f"Nova sequência: {order}\n")
