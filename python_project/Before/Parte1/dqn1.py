import numpy as np
import pandas as pd
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Classe do agente DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # fator de desconto
        self.epsilon = 1.0  # taxa de exploração inicial
        self.epsilon_decay = 0.995  # decaimento da taxa de exploração
        self.epsilon_min = 0.01  # taxa de exploração mínima
        self.learning_rate = 0.001  # taxa de aprendizado
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Carregamento dos dados do jogo
data_final = pd.read_csv('/home/darkcover/Documentos/Out/dados/data_final.csv')

# Função para definir o estado atual com base na última entrada
def get_state(data_final, index):
    state = data_final.loc[index, ['odd_entrada', 'media5', 'media10', 'media20', 'media40', 'media80', 'media160', 'media320', 'media640']].values
    return state.reshape(1, -1)

# Definição das constantes
state_size = 9  # dimensão do estado
action_size = 2  # número de ações (apostar ou não apostar)
n_episodes = 1000  # número de episódios
batch_size = 32  # tamanho do lote para o replay

# Inicialização do agente DQN
agent = DQNAgent(state_size, action_size)

# Loop de treinamento
for episode in range(n_episodes):
    state = get_state(data_final, 0)  # estado inicial
    done = False
    total_reward = 0
    for index in range(len(data_final)):
        action = agent.act(state)
        next_state = get_state(data_final, index + 1)
        reward = data_final.loc[index, 'acerto']  # recompensa é o acerto (0, 1 ou 2)
        done = index == len(data_final) - 1
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    print(f"Episode: {episode + 1}/{n_episodes}, Total Reward: {total_reward}")

# Após o treinamento, você pode usar o agente para fazer previsões no jogo real
# Por exemplo, você pode usar agent.act(state) para obter a ação do agente com base no estado atual
