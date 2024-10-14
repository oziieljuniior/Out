Para melhorar a previsão das novas entradas, você pode considerar algumas abordagens que otimizem a integração das variáveis e padrões observados. Aqui estão algumas sugestões:

### 1. **Melhoria na Tendência (Detecção de Padrões Mais Complexos)**
A função `calcular_tendencia` usa apenas a diferença média (`np.diff`) das últimas 60 entradas, o que pode ser um pouco simples. Para capturar padrões mais complexos, você poderia introduzir uma análise mais robusta, como:

- **Médias móveis ponderadas**: Em vez de usar apenas uma média simples, aplique uma ponderação maior para as entradas mais recentes, o que pode ajudar a capturar mudanças mais significativas em tendências.
  
- **Análise de autocorrelação**: Verifique se há uma autocorrelação nas entradas anteriores para capturar ciclos ou padrões que podem influenciar a previsão.

- **Séries temporais**: Considere implementar técnicas de séries temporais, como ARIMA (AutoRegressive Integrated Moving Average), que podem capturar melhor a dependência entre entradas passadas e futuras.

### 2. **Ajuste da Variância**
A variância pode estar tendo um impacto grande nas previsões de probabilidades. Você pode experimentar suavizar o impacto da variância ou incorporar métricas como a **desvio padrão** para controlar as flutuações extremas:

```python
desvio_padrao = np.std(array)
ajuste_variancia = tendencia * desvio_padrao
```
Isso pode suavizar os ajustes nas probabilidades, evitando mudanças abruptas baseadas na variabilidade.

### 3. **Explorar Modelos Preditivos com Regressão**
Como você já mencionou a utilização de **regressão** em outras partes do modelo, uma ideia seria criar um modelo simples de **regressão logística** ou **regressão linear** para prever a probabilidade de `1` com base nas últimas entradas e no comportamento da tendência. Isso pode ajudar a tornar a predição mais precisa:

```python
from sklearn.linear_model import LogisticRegression

# Treinar um modelo de regressão logística com as entradas anteriores
X = np.array(novas_entradas).reshape(-1, 1)
y = np.array([1 if valor >= 0.5 else 0 for valor in novas_entradas])

modelo = LogisticRegression()
modelo.fit(X, y)

# Usar o modelo para prever a probabilidade de vir um 1 ou 0 nas próximas entradas
probabilidade_de_1 = modelo.predict_proba([[valor_atual]])[0, 1]
```

### 4. **Ajuste da Probabilidade com um Decaimento**
Você pode ajustar a probabilidade de `1` usando uma função de **decaimento** para controlar a influência das tendências antigas, reduzindo a importância delas ao longo do tempo:

```python
# Suavizar a tendência com um decaimento exponencial
fator_decaimento = 0.95
tendencia_suavizada = np.sum([fator_decaimento**i * diff for i, diff in enumerate(diffs[::-1])])
```
Essa abordagem dá mais peso às mudanças recentes e menos às mais antigas, tornando o modelo mais adaptável a flutuações recentes.

### 5. **Considerar Redes Neurais**
Se você estiver aberto a incorporar um modelo mais avançado, redes neurais simples, como **Redes Neurais Recorrentes (RNNs)** ou **LSTMs**, são boas opções para capturar padrões mais sutis e não lineares nas sequências de entradas.

Essas abordagens podem ajudar a detectar padrões de longo prazo que as médias simples não capturam, permitindo que o modelo aprenda automaticamente como diferentes combinações de entradas impactam futuras previsões.

### Implementação com Pequenas Alterações
Aqui está uma versão ajustada da sua função `prever_01s`, integrando a suavização da variância e um decaimento na tendência:

```python
def prever_01s(novas_entradas, array, tamanho_previsao=120, limite_inferior=0.28, limite_superior=0.63):
    previsoes = []
    fator_decaimento = 0.95  # Para suavização
    desvio_padrao = np.std(array)
    
    for i in range(tamanho_previsao):
        valor_atual = novas_entradas[-1] if len(novas_entradas) > 0 else 0.5
        tendencia = calcular_tendencia(novas_entradas)
        
        # Suavizar a tendência usando o fator de decaimento
        tendencia_suavizada = tendencia * fator_decaimento
        
        # Ajustar a probabilidade de 1 com a tendência suavizada e a variância
        probabilidade_de_1 = valor_atual + tendencia_suavizada * desvio_padrao
        probabilidade_de_1 = np.clip(probabilidade_de_1, limite_inferior, limite_superior)
        
        previsao = 1 if np.random.rand() < probabilidade_de_1 else 0
        previsoes.append(previsao)
        
        novas_entradas = np.append(novas_entradas, probabilidade_de_1)
    
    return previsoes
```

Essas melhorias podem tornar as previsões mais robustas e ajustadas às oscilações naturais dos dados.