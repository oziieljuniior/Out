### Fase 1
* O código começa importando as bibliotecas necessárias: pandas e numpy.
* Em seguida, ele inicializa algumas variáveis. A variável i é usada para controlar as entradas do jogo. A variável lista_entradas é usada para armazenar todas as entradas geradas. As variáveis apostar, contagem e level são usadas para controlar o estado do jogo.
* O código então cria um DataFrame pandas chamado data com algumas informações iniciais.
* O jogo começa com um loop while que continua até que i seja igual a zero. Dentro deste loop, o código solicita ao usuário para inserir a última entrada gerada.
* O código então verifica se a condição de aposta é verdadeira e se a entrada é maior que 1.5. Se for verdade, o código incrementa a contagem e verifica se a contagem chegou a 15. Se a contagem chegar a 15, o nível do jogador aumenta e a contagem é redefinida para zero.
* Se a condição de aposta for verdadeira e a entrada for menor ou igual a 1.5, o código decrementa a contagem em 2. Se a contagem for menor ou igual a -10, o nível do jogador é redefinido para 1 e a contagem é redefinida para zero.
* O código então adiciona a entrada à lista_entradas e atualiza o DataFrame data com as informações atuais do jogo.
* Se o número de entradas for maior ou igual a 5, o código começa a calcular as médias das últimas 5, 10, 20, 40, 80, 160, 320 e 640 entradas. Ele então verifica se qualquer uma dessas médias está abaixo de um certo limite. Se estiver, a condição de aposta é definida como verdadeira.
* Finalmente, o código imprime a lista_entradas e o DataFrame data, e salva o DataFrame em um arquivo CSV.

Espero que isso ajude a entender o que o código está fazendo! Se você tiver mais perguntas, fique à vontade para perguntar.

### Fase 2
* Importação de bibliotecas: As bibliotecas pandas e numpy são importadas. Elas são usadas para manipulação de dados e operações matemáticas, respectivamente.
* Carregamento de dados: O script carrega um arquivo CSV chamado ‘odds_200k.csv’ em um DataFrame do pandas. Em seguida, ele remove uma coluna chamada ‘Unnamed: 0’ e renomeia a coluna ‘Odd_Categoria’ para ‘odd_saida’.
* Inicialização de variáveis: Várias variáveis são inicializadas, incluindo uma lista vazia chamada lista_entradas e várias variáveis de controle do jogo (apostar, contagem, level). Além disso, um DataFrame chamado data_final é criado com uma única linha de dados iniciais.
* Loop principal: O script então entra em um loop for que itera sobre cada linha do DataFrame data_inicial. Para cada linha, ele realiza várias operações, incluindo:
* Atualiza a variável i com o valor da coluna ‘odd_saida’ da linha atual.
* Verifica se a condição de aposta foi atendida e, em caso afirmativo, atualiza as variáveis de controle do jogo (acerto, contagem, level) de acordo.
* Adiciona o valor de i à lista_entradas.
* Se o tamanho da lista_entradas for maior ou igual a 6, ele adiciona uma nova linha ao DataFrame data_final com os valores atuais das variáveis de controle do jogo e as médias das últimas entradas.
* Se o tamanho da lista_entradas for maior ou igual a 5, ele calcula várias médias móveis das últimas entradas e verifica se alguma delas atende a determinadas condições. Se sim, ele define a variável apostar para 1; caso contrário, ele a define para 0.
* Salvando os resultados: Finalmente, o script imprime a lista_entradas e o DataFrame data_final, e salva o data_final em um arquivo CSV.
#### Primeira análise nos dados

[Acesso](https://github.com/oziieljuniior/Out/blob/main/notes/Analise1.pdf)

### Fase 3

O código apresentado é um exemplo de um agente de aprendizado por reforço profundo (Deep Q-Learning Agent - DQN) implementado para tomar decisões baseadas em dados históricos. A estrutura geral do código pode ser dividida em várias partes principais, cada uma com uma funcionalidade específica:

1. **Importação de bibliotecas**:
   - `numpy` e `pandas`: Manipulação de dados.
   - `sklearn.model_selection`: Divisão de dados em conjuntos de treino e teste.
   - `sklearn.preprocessing`: Normalização dos dados.
   - `tensorflow`: Criação e treinamento da rede neural.
   - `collections.deque` e `random`: Manipulação de memória e amostragem aleatória.
   - `matplotlib.pyplot`: Visualização dos resultados.

2. **Definição da classe `DQNAgent`**:
   - **Inicialização (`__init__`)**: Configura os parâmetros do agente, como estados, ações, taxa de aprendizado, fator de desconto (`gamma`), epsilon (para exploração), memória de experiência e construção do modelo de rede neural.
   - **Construção do modelo (`build_model`)**: Cria uma rede neural com camadas densas (fully connected) e dropout para regularização.
   - **Escolha da ação (`get_action`)**: Decide a ação a ser tomada com base nas Q-valores preditos pelo modelo ou escolhe aleatoriamente (exploração).
   - **Memorizar experiência (`remember`)**: Armazena experiências (estado, ação, recompensa, próximo estado, terminal) na memória.
   - **Rejogar experiências (`replay`)**: Treina a rede neural com um minibatch de experiências armazenadas.
   - **Atualização do epsilon (`update_epsilon`)**: Decresce o valor de epsilon para reduzir a exploração ao longo do tempo.

3. **Função de cálculo de recompensa (`calculate_reward`)**:
   - Define a recompensa com base na ação tomada e se a previsão foi correta ou incorreta.

4. **Normalização dos dados (`normalize_data`)**:
   - Normaliza os dados de entrada utilizando `StandardScaler`.

5. **Função principal (`main`)**:
   - **Carregamento e preparação dos dados**: Carrega os dados de um arquivo CSV, seleciona features e variável de saída, e divide os dados em conjuntos de treino e teste.
   - **Configuração de parâmetros**: Define parâmetros de treinamento como número de épocas e tamanho do lote.
   - **Criação do agente DQN**: Inicializa o agente com os parâmetros definidos.
   - **Treinamento do agente**: Loop de treinamento para várias épocas, onde o agente interage com o ambiente, armazena experiências e treina a rede neural.
   - **Avaliação do desempenho**: Calcula e armazena métricas de desempenho para cada época (precisão, acurácia direcional e acurácia direcional ponderada).
   - **Visualização dos resultados**: Plota gráficos das métricas de desempenho ao longo das épocas.

### Objetivo do Código
O objetivo deste código é treinar um agente de aprendizado por reforço que aprende a tomar decisões (apostar ou não apostar) com base em dados históricos de apostas. O agente usa uma rede neural profunda para estimar os Q-valores das ações e toma decisões baseadas nestas estimativas. O treinamento é realizado através da técnica de rejogar experiências armazenadas na memória, o que permite ao agente aprender de suas experiências passadas.

### Pontos Importantes
- **Exploração vs Exploração**: O agente balanceia a exploração de novas ações com a exploração de ações conhecidas utilizando uma estratégia epsilon-greedy.
- **Memória de Experiência**: Utiliza uma memória de experiência para armazenar e amostrar experiências, o que melhora a eficiência do aprendizado.
- **Rede Neural Profunda**: A rede neural utilizada tem duas camadas densas com dropout para evitar overfitting.
- **Normalização dos Dados**: Os dados de entrada são normalizados para melhorar a eficiência do treinamento do modelo.

### Possíveis Melhorias
- **Novas Features**: [Novas Features para estudo](https://github.com/oziieljuniior/Out/blob/main/notes/novas_features.md)
- **Hiperparâmetros**: A escolha dos hiperparâmetros (taxa de aprendizado, gamma, epsilon) pode ser ajustada usando técnicas como otimização bayesiana.
- **Aprimoramento do Modelo**: Testar arquiteturas de rede neural mais complexas ou outras técnicas de aprendizado por reforço, como Dueling DQN, Double DQN, ou Prioritized Experience Replay.
- **Avaliação Detalhada**: Implementar métricas adicionais de avaliação e realizar uma análise mais detalhada do desempenho do agente.
