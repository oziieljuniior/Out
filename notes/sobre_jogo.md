Claro! Vamos relembrar a proposta do jogo e como o agente DQN está sendo treinado para ele:

### Proposta do Jogo
O jogo envolve uma série de apostas com base em odds (probabilidades) de eventos. O jogador (ou agente) deve decidir em cada rodada se deve apostar ou não, baseado nas odds de entrada e várias médias de odds das últimas entradas. A aposta é considerada um acerto se a odd de saída for maior ou igual a 4, e um erro caso contrário.

#### Regras e Mecânica do Jogo:
1. **Rodadas e Níveis**:
   - O jogo tem uma série de rodadas.
   - O jogador começa no nível 1 e pode subir de nível se tiver uma sequência de acertos.
   - A contagem de acertos e erros influencia a mudança de nível. Se a contagem chegar a 15 acertos, o jogador sobe de nível. Se a contagem chegar a -10, o jogador volta ao nível 1.

2. **Entradas e Apostas**:
   - Para cada entrada (odd de entrada, odd de saída e outras características), o jogador decide se deve apostar.
   - A decisão de apostar é baseada nas médias móveis das últimas 5, 10, 20, 40, 80, 160, 320 e 640 entradas.
   - Se as médias estiverem dentro de certos limites, o jogador decide apostar na próxima rodada.

3. **Recompensas**:
   - Se o jogador aposta e a odd de saída é >= 4, é um acerto e a contagem aumenta.
   - Se o jogador aposta e a odd de saída é < 4, é um erro e a contagem diminui.

### Objetivo do Agente DQN
O agente DQN (Deep Q-Network) é treinado para tomar decisões de aposta com o objetivo de maximizar as recompensas acumuladas ao longo das rodadas. O agente aprende a identificar quando as odds indicam uma boa oportunidade de aposta e quando é melhor não apostar, com base em seu treinamento.

### Treinamento do Agente DQN
Para treinar o agente DQN para este jogo, seguimos estes passos:

1. **Definir o Ambiente**:
   - Cada estado do ambiente é uma representação das odds atuais e suas médias móveis.
   - As ações disponíveis para o agente são apostar (1) ou não apostar (0).

2. **Construir a Rede Neural**:
   - A rede neural do agente DQN aprende a mapear estados para ações com base nas recompensas recebidas.
   - Utilizamos técnicas como dropout e regularização para melhorar o aprendizado e evitar overfitting.

3. **Experiência Replay e Atualização de Epsilon**:
   - Utilizamos um buffer de experiência para armazenar as transições (estado, ação, recompensa, próximo estado) e amostramos mini-batches para treinamento.
   - O epsilon (exploração vs. exploração) é decaído gradualmente para garantir que o agente explore bem no início e depois se concentre em exploração.

4. **Função de Recompensa**:
   - A função de recompensa é ajustada para refletir os acertos e erros de apostas. Recompensas são atribuídas de acordo com o sucesso ou falha das apostas feitas.

### Sugestões Específicas para o Modelo Atual
Considerando a análise dos resultados e a natureza do jogo, aqui estão algumas sugestões específicas:

1. **Refinar a Função de Recompensa**:
   - Aumente a penalidade para apostas incorretas e recompensas para acertos para criar uma distinção mais clara nas ações do agente.

2. **Ajustar o Decaimento de Epsilon**:
   - Experimente diferentes taxas de decaimento para garantir que o agente explore suficientemente no início e depois explore.

3. **Analisar a Curva de Aprendizado**:
   - A análise das métricas de precisão, acurácia direcional e acurácia direcional ponderada ao longo das épocas indica que há pouca variação. Pode ser necessário ajustar a complexidade da rede ou fornecer mais dados para treinar.

4. **Incluir Mais Características**:
   - Adicione novas características que possam influenciar as decisões de aposta, como indicadores técnicos adicionais ou variáveis temporais.

### Implementação do Modelo de Treinamento

Aqui está o modelo ajustado para o treinamento do agente DQN:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.001, gamma=0.99, epsilon=1.0, buffer_size=10000):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=buffer_size)

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=self.n_states, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.n_actions, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_actions)
        q_values = self.model.predict(state.reshape(1, -1), batch_size=1, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state.reshape(1, -1), batch_size=1, verbose=0)[0]))
            target_f = self.model.predict(state.reshape(1, -1), batch_size=1, verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def calculate_reward(action, acerto):
    if action == 1:
        if acerto == 1:
            return 10
        elif acerto == 2:
            return -5
    return 0

def normalize_data(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def main():
    data = pd.read_csv('/home/darkcover/Documentos/Out/dados/data_final.csv')
    features = data[['odd_entrada', 'media5', 'media10', 'media20', 'media40', 'media80', 'media160', 'media320', 'media640', 'acerto', 'level', 'contagem']].values
    rewards = data['apostar'].values
    features = normalize_data(features)

    X_train, X_test, y_train, y_test = train_test_split(features, rewards, test_size=0.2, random_state=42)

    n_epochs = 25
    batch_size = 64

    n_states = X_train.shape[1]
    n_actions = 2
    dqn_agent = DQNAgent(n_states, n_actions)

    epoch_accuracies = []
    epoch_directional_accuracies = []
    epoch_weighted_directional_accuracies = []

    for epoch in range(n_epochs):
        correct_predictions = 0
        overestimations = 0
        weighted_correct_predictions = 0
        weighted_overestimations = 0

        for i in range(len(X_train) - 1):
            state = X_train[i]
            action = dqn_agent.get_action(state)
            true_action = y_train[i]
            acerto = state[-3]
            reward = calculate_reward(action, acerto)
            next_state = X_train[i + 1]
            done = (i == len(X_train) - 2)

            dqn_agent.remember(state, action, reward, next_state, done)
            if done:
                dqn_agent.replay(batch_size)

            if action == true_action:
                correct_predictions += 1
                weighted_correct_predictions += 2
            elif action > true_action:
                overestimations += 1
                weighted_overestimations += 1

        epoch_accuracy = correct_predictions / len(X_test)
        epoch_directional_accuracy = (correct_predictions + overestimations) / len(X_test)
        epoch_weighted_directional_accuracy = (weighted_correct_predictions + weighted_overestimations) / len(X_test)

        epoch_accuracies.append(epoch_accuracy)
        epoch_directional_accuracies.append(epoch_directional_accuracy)
        epoch_weighted_directional_accuracies.append(epoch_weighted_directional_accuracy)

        print(f'Época {epoch + 1}/{n_epochs} - Precisão: {epoch_accuracy:.4f},

 Acurácia Direcional: {epoch_directional_accuracy:.4f}, Acurácia Direcional Ponderada: {epoch_weighted_directional_accuracy:.4f}')

    plt.figure(figsize=(12, 6))
    plt.plot(epoch_accuracies, label='Precisão')
    plt.plot(epoch_directional_accuracies, label='Acurácia Direcional')
    plt.plot(epoch_weighted_directional_accuracies, label='Acurácia Direcional Ponderada')
    plt.xlabel('Época')
    plt.ylabel('Métrica')
    plt.legend()
    plt.title('Desempenho do Modelo ao Longo das Épocas')
    plt.show()

if __name__ == "__main__":
    main()
```

Este código ajusta a função de recompensa, adiciona camadas de dropout para evitar overfitting e ajusta o buffer de experiência e a taxa de decaimento do epsilon. Experimente e ajuste os hiperparâmetros conforme necessário para melhorar o desempenho do agente.
