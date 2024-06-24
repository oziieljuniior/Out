import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, buffer_size=10000):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=buffer_size)
        self.model = self.build_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.Huber())

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.n_states,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.n_actions, activation='linear')
        ])
        return model

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            q_values = self.model.predict(state.reshape(1, -1), batch_size=1, verbose=0)
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), batch_size=1, verbose=0))
            target_f = self.model.predict(state.reshape(1, -1), batch_size=1, verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def calculate_reward(action, acerto):
    reward = 0
    if action == 1:  # Apostou
        if acerto == 1:
            reward = 1  # Recompensa por apostar corretamente
        elif acerto == 0:
            reward = -1  # Penalidade por apostar incorretamente
    return reward

def normalize_data(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def main():
    # Carregar os dados
    data = pd.read_csv('/home/darkcover/Documentos/Out/dados/data_final2.csv')

    # Selecionar as features e a variável de saída
    features = data[['odd_entrada', 'media5', 'media10', 'media20', 'media40', 'media80', 'media160', 'media320', 'media640']].values
    actions = data['apostar'].values  # Variável de saída: se deve apostar ou não
    acertos = data[['acerto', 'level', 'contagem']].values  # Variável de saída para calcular a recompensa

    features = normalize_data(features)

    # Adicionar level e contagem às features
    features = np.hstack((features, acertos[:, 1:]))

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test, acerto_train, acerto_test = train_test_split(features, actions, acertos, test_size=0.2, random_state=42)

    # Parâmetros de treinamento
    n_epochs = 2  # Aumentar o número de épocas
    batch_size = 128  # Tentar um tamanho de lote menor para mais atualizações

    # Crie o agente DQN
    n_states = X_train.shape[1]
    n_actions = 2  # Ações possíveis: apostar ou não apostar
    dqn_agent = DQNAgent(n_states, n_actions, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.985, epsilon_min=0.01, buffer_size=10000)

    epoch_accuracies = []
    epoch_directional_accuracies = []
    epoch_weighted_directional_accuracies = []

    # Treinamento do agente
    for epoch in range(n_epochs):
        correct_predictions = 0
        overestimations = 0
        weighted_correct_predictions = 0
        weighted_overestimations = 0

        for i in range(len(X_train) - 1):
            state = X_train[i, :]
            action = dqn_agent.get_action(state)
            true_action = y_train[i]
            acerto = acerto_train[i][0]
            reward = calculate_reward(action, acerto)
            next_state = X_train[i + 1, :]
            done = (i == len(X_train) - 2)

            dqn_agent.remember(state, action, reward, next_state, done)

            if done:
                dqn_agent.replay(batch_size)
                dqn_agent.update_epsilon()

            if action == true_action:
                correct_predictions += 1
                weighted_correct_predictions += 2  # Peso maior para acertos exatos
            elif action > true_action:
                overestimations += 1
                weighted_overestimations += 1  # Peso menor para erros

        epoch_accuracy = correct_predictions / len(X_train)
        epoch_directional_accuracy = (correct_predictions + overestimations) / len(X_train)
        epoch_weighted_directional_accuracy = (weighted_correct_predictions + weighted_overestimations) / len(X_train)

        epoch_accuracies.append(epoch_accuracy)
        epoch_directional_accuracies.append(epoch_directional_accuracy)
        epoch_weighted_directional_accuracies.append(epoch_weighted_directional_accuracy)

        print(f'Época {epoch + 1}/{n_epochs} - Precisão: {epoch_accuracy:.4f}, Acurácia Direcional: {epoch_directional_accuracy:.4f}, Acurácia Direcional Ponderada: {epoch_weighted_directional_accuracy:.4f}')

    # Teste o agente
    correct_predictions = 0
    overestimations = 0
    weighted_correct_predictions = 0
    weighted_overestimations = 0

    predicted_actions = []

    for i in range(len(X_test)):
        state = X_test[i, :]
        action = dqn_agent.get_action(state)
        true_action = y_test[i]

        predicted_actions.append(action)

        if action == true_action:
            correct_predictions += 1
            weighted_correct_predictions += 2
        elif action > true_action:
            overestimations += 1
            weighted_overestimations += 1

    accuracy = correct_predictions / len(X_test)
    directional_accuracy = (correct_predictions + overestimations) / len(X_test)
    weighted_directional_accuracy = (weighted_correct_predictions + weighted_overestimations) / len(X_test)

    print("Precisão:", accuracy)
    print("Acurácia Direcional:", directional_accuracy)
    print("Acurácia Direcional Ponderada:", weighted_directional_accuracy)

    # Matriz de Confusão
    conf_matrix = confusion_matrix(y_test, predicted_actions)
    print("Matriz de Confusão:\n", conf_matrix)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Não Apostar', 'Apostar'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epoch_accuracies, label='Precisão')
    plt.plot(epoch_directional_accuracies, label='Acurácia Direcional')
    plt.plot(epoch_weighted_directional_accuracies, label='Acurácia Direcional Ponderada')
    plt.xlabel('Época')
    plt.ylabel('Métrica')
    plt.title('Desempenho do Modelo ao Longo das Épocas')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    metrics = [accuracy, directional_accuracy, weighted_directional_accuracy]
    metric_labels = ['Precisão', 'Acurácia Direcional', 'Acurácia Direcional Ponderada']
    colors = ['blue', 'orange', 'green']
    plt.bar(metric_labels, metrics, color=colors)
    plt.xlabel('Métricas')
    plt.ylabel('Valor')
    plt.title('Métricas de Desempenho no Conjunto de Teste')
    plt.ylim(0, 1)
    plt.show()

    dqn_agent.model.save('AgenteAposta.keras')
    data_save = pd.DataFrame({'epoch_accuracies': epoch_accuracies,
                              'epoch_directional_accuracies': epoch_directional_accuracies,
                              'epoch_weighted_directional_accuracies': epoch_weighted_directional_accuracies})
    data_save.to_csv('/home/darkcover/Documentos/Out/dados/metricas_aposta.csv', index=False)

    # Teste o agente
    data2 = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k_1.csv')
    # Selecionar as features e a variável de saída
    features1 = data2[['odd_entrada', 'media5', 'media10', 'media20', 'media40', 'media80', 'media160', 'media320', 'media640']].values
    actions1 = data2['apostar'].values  # Variável de saída: se deve apostar ou não
    acertos1 = data2[['acerto', 'level', 'contagem']].values  # Variável de saída para calcular a recompensa

    features1 = normalize_data(features1)

    # Adicionar level e contagem às features
    features1 = np.hstack((features1, acertos1[:, 1:]))

    # Dividir os dados em treino e teste
    X_train1, X_test1, y_train1, y_test1, acerto_train1, acerto_test1 = train_test_split(features1, actions1, acertos1, test_size=0.9, random_state=42)

    correct_predictions1 = 0
    overestimations1 = 0
    weighted_correct_predictions1 = 0
    weighted_overestimations1 = 0

    predicted_actions1 = []

    for i in range(len(X_test1)):
        state = X_test1[i, :]
        action = dqn_agent.get_action(state)
        true_action = y_test1[i]
        print(i)
        predicted_actions.append(action)

        if action == true_action:
            correct_predictions1 += 1
            weighted_correct_predictions1 += 2
        elif action > true_action:
            overestimations1 += 1
            weighted_overestimations1 += 1

    accuracy = correct_predictions1 / len(X_test1)
    directional_accuracy = (correct_predictions1 + overestimations1) / len(X_test1)
    weighted_directional_accuracy = (weighted_correct_predictions1 + weighted_overestimations1) / len(X_test1)

    print("Precisão:", accuracy)
    print("Acurácia Direcional:", directional_accuracy)
    print("Acurácia Direcional Ponderada:", weighted_directional_accuracy)

    # Matriz de Confusão
    conf_matrix = confusion_matrix(y_test1, predicted_actions1)
    print("Matriz de Confusão:\n", conf_matrix)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Não Apostar', 'Apostar'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    
if __name__ == '__main__':
    main()
