# Out
## Como o código funciona
Informações como o jogo funciona, acesse:

[Leia sobre o jogo](https://github.com/oziieljuniior/Out/blob/main/notes/sobre_jogo.md)

[Coisas Passadas](https://github.com/oziieljuniior/Out/blob/main/notes/CoisasP.md)

[Últimas Atualizações](https://github.com/oziieljuniior/Out/blob/main/notes/update_27_07.md)

## Resumo da Opera:
No final, tudo se resume em como você lida com a média móvel. Isto é possível apenas por causa de duas caracteristicas.
A primeira, se passa pela geração de número pseudo aleatórios(NPA).

Quando estamos trabalhando com NPA visualizamos as caracteríticas intrisicas das médias móveis. Pois as médias vão ter uma convergência uniforme a partir de uma certa quantidade de entradas.

## Coisas futuras
Dado que os números são pseudo-aleatórios e as abordagens tradicionais de séries temporais, Monte Carlo, e redes neurais não têm sido eficazes, além da ausência de dependência direta do tempo, podemos explorar algumas outras formas de modelagem que talvez capturem melhor as oscilações e padrões observados.

Aqui estão algumas abordagens que podem ser úteis:

### 1. **Modelagem de Oscilações com Funções Sob Medida**
Como você mencionou, há oscilações entre 0,30 e 0,60 com um erro de ±0,05. Podemos tentar modelar diretamente esse comportamento oscilatório sem focar em previsões tradicionais baseadas em tempo, utilizando funções customizadas que se ajustem a esses intervalos. O objetivo seria capturar o comportamento cíclico, sem depender de séries temporais tradicionais ou do tempo diretamente.

#### Estratégia:
   - **Ajuste por Osciladores Não Lineares**: Podemos criar um modelo que simule a oscilação observada entre 0,30 e 0,60, utilizando um oscilador não linear (similar a um sistema dinâmico). Isso pode ser feito construindo uma função que gere oscilações aleatórias entre esses valores, incorporando ruído controlado. Isso levaria em conta a aleatoriedade sem que o modelo esteja dependente de uma série temporal.
   - **Fator de Amortecimento**: Caso o comportamento tenha oscilações que diminuem ou aumentam em amplitude ao longo do tempo, podemos adicionar um fator de amortecimento na função.
   
#### Exemplo (Python):
```python
import numpy as np
import matplotlib.pyplot as plt

# Função que gera oscilações pseudo-aleatórias com erro controlado
def pseudo_oscillation(x, amplitude, base, noise_factor):
    noise = np.random.normal(0, noise_factor, len(x))
    oscillation = base + amplitude * np.sin(2 * np.pi * x / 50) + noise
    return oscillation

x_data = np.linspace(0, 1000, 1000)
y_data = pseudo_oscillation(x_data, 0.15, 0.45, 0.05)  # Oscila entre 0.30 e 0.60 com erro de 0.05

plt.plot(x_data, y_data)
plt.show()
```
Essa abordagem gera oscilações pseudo-aleatórias com amplitude controlada entre 0,30 e 0,60 e um erro de ±0,05.

### 2. **Algoritmo Evolutivo (Genético)**

[Leitura sobre o algoritmo Evolutivo](https://github.com/oziieljuniior/Out/blob/main/notes/Algoritmo_Genetico.md)
Outra abordagem seria usar algoritmos evolutivos, que podem ajustar o comportamento pseudo-aleatório para aprender com as oscilações e padrões. Um algoritmo genético pode iterativamente encontrar um conjunto de parâmetros que melhor se ajusta à oscilação observada.

#### Estratégia:
   - Definir uma **função de fitness** que minimize a diferença entre o valor previsto e os valores reais oscilantes entre 0,30 e 0,60.
   - Usar a variação e mutação para explorar diferentes soluções.
   - Esse tipo de abordagem é bastante eficaz quando os padrões são difíceis de modelar com métodos tradicionais, especialmente em problemas de otimização com dados pseudo-aleatórios.

   Aqui estão as respostas às suas dúvidas:

1) **Se eu aumentar o tamanho da amostra, o modelo melhora seu desempenho?**

   Não necessariamente. O aumento do tamanho da amostra pode oferecer mais dados para o modelo aprender, mas também pode aumentar a complexidade e o tempo de execução do algoritmo. Em um algoritmo genético, a chave é a **qualidade dos dados e da função de fitness**. Se a amostra maior trouxer mais padrões relevantes, o desempenho pode melhorar, mas se o aumento introduzir mais ruído, o desempenho pode até piorar. O ideal é realizar testes com amostras de diferentes tamanhos para ver como o modelo reage.

2) **Melhor solução: [0.0956795788731983, 0.41146099606375847, 0.5298001732932583, 0.019504466237991215]**

   Este resultado refere-se aos parâmetros do melhor indivíduo encontrado pelo algoritmo genético. Cada valor corresponde a um dos parâmetros ajustados para a geração das oscilações. Especificamente:

   - **0.0956795788731983 (amplitude)**: Esse valor indica o tamanho das oscilações geradas. Uma amplitude baixa, como neste caso, significa que as oscilações previstas são pequenas.
   - **0.41146099606375847 (frequência)**: Refere-se à rapidez com que as oscilações ocorrem. Um valor intermediário sugere oscilações de ritmo moderado.
   - **0.5298001732932583 (offset)**: Esse valor desloca as oscilações para cima ou para baixo. Aqui, ele centraliza as oscilações em torno de 0,53.
   - **0.019504466237991215 (ruído)**: Esse parâmetro adiciona um elemento aleatório (ruído) às oscilações. Um valor muito baixo indica que há pouco ruído, ou seja, as previsões são mais suaves e menos perturbadas por flutuações aleatórias.

   Em resumo, esses números representam a melhor combinação de parâmetros que o algoritmo genético encontrou para aproximar as oscilações dos dados reais. Se o resultado final está previsível, você pode experimentar aumentar a amplitude ou o ruído, ou até ajustar a função de fitness para penalizar previsões que não capturam as oscilações de forma suficiente.

### 3. **Simulação de Processos Estocásticos Customizados**
Osciladores pseudo-aleatórios podem ser modelados usando processos estocásticos customizados que simulam oscilações. Ao invés de utilizar Monte Carlo de forma tradicional, você pode customizar o comportamento estocástico com base nas oscilações entre 0,30 e 0,60.

#### Estratégia:
   - **Cadeia de Markov**: Podemos implementar uma cadeia de Markov onde os estados de oscilação (por exemplo, entre 0,30 e 0,60) têm probabilidades de transitar entre si, capturando as variações altas e baixas observadas no gráfico.
   - Em cada ponto, a transição para o próximo valor pode depender de uma matriz de probabilidade construída a partir dos dados observados.

### 4. **Modelos de Funções Spline (Interpolação)**
Se os dados oscilam dentro de um intervalo conhecido com um erro controlado, funções **splines** podem ser úteis para modelar a oscilação. Splines são basicamente curvas suaves ajustadas entre pontos de dados conhecidos e podem ajudar a suavizar os ruídos enquanto capturam o padrão oscilatório de forma flexível.

### 5. **Desenvolvimento de Critérios Heurísticos**
Se os modelos mais formais estão falhando em capturar as oscilações corretamente, talvez seja necessário explorar a criação de um conjunto de **regras heurísticas**. Essas regras podem ser baseadas em observações empíricas sobre quando ocorrem oscilações ou quedas e podem ser mais adaptáveis do que os modelos tradicionais.

#### Exemplo:
   - Se a oscilação atinge um limite superior (por exemplo, 0,60), há uma chance maior de haver uma queda.
   - Se atinge um limite inferior (por exemplo, 0,30), há uma chance maior de subir.
   
   Isso criaria um modelo baseado em comportamento limítrofe, simulando os comportamentos de subida e descida.

---

### Considerações Finais
Parece que uma abordagem baseada em osciladores ajustáveis com adição de ruído ou um algoritmo genético, que pode adaptar-se às oscilações pseudo-aleatórias, seria mais apropriada para o seu caso. Outra opção seria o uso de processos estocásticos customizados, que podem capturar melhor as oscilações entre os valores limites observados.

Qual dessas abordagens faz mais sentido para você ou gostaria de explorar mais a fundo?