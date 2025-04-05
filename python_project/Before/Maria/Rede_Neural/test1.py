import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

# Exemplo de dados (1000 amostras, cada uma com 30 valores de entrada e 30 saídas binárias)
X = np.random.rand(1000, 180)  # 1000 amostras com 30 características de entrada cada
y = np.random.randint(0, 2, (1000, 30))  # 1000 amostras com 30 saídas binárias (0 ou 1)

# Criando o modelo
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(180,)))  # Primeira camada densa com 64 neurônios
model.add(Dense(32, activation='relu'))  # Segunda camada densa com 32 neurônios
model.add(Dense(30, activation='sigmoid'))  # Camada de saída com 30 neurônios, usando 'sigmoid' para prever 0s e 1s

# Compilando o modelo com uma métrica adicional
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo e armazenando o histórico
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Plotando a curva de perda
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Curva de Perda do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Dados de teste (1 amostras com 30 valores de entrada cada)
X_test = np.random.rand(1, 180)
y_test = np.random.randint(0, 2, (1, 30))  # Saídas binárias de teste

print(X_test)

# Avaliando o modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Perda no conjunto de teste: {test_loss}')
print(f'Acurácia no conjunto de teste: {test_acc}')

# Fazendo previsões
y_pred = model.predict(X_test)

# Arredondando as previsões para 0 ou 1
y_pred_rounded = np.round(y_pred)

# Comparando previsões com os valores reais
print(f'Valores Previstos (arredondados): {y_pred_rounded}')
print(f'Valores Reais: {y_test}')
