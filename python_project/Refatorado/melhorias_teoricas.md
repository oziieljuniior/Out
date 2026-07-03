# Melhorias Teóricas Oportadas do Livro "Prediction, Learning, and Games"

Este documento descreve as propostas de melhoria conceitual para o simulador de previsão sequencial com base no livro de Nicolò Cesa-Bianchi e Gábor Lugosi (2006).

---

## 1. Substituir a Rede Neural por um Algoritmo de Gradiente Online
* **Teoria (Capítulo 11 - Linear Pattern Recognition)**:
  Em aprendizado online puro, treinar redes neurais em lote (*batch*) periodicamente é inadequado devido à complexidade de tempo $O(t \times \text{épocas})$ que cresce a cada passo. Em vez disso, a decisão pode ser representada por um preditor linear:
  $$\hat{y}_t = \sigma(w_{t-1} \cdot x_t)$$
  E a cada passo $t$, atualiza-se o vetor de pesos $w$ usando a regra do **Online Gradient Descent (OGD)** (derivada da regra de Widrow-Hoff para perda quadrática, ou gradiente da entropia cruzada binária):
  $$w_t = w_{t-1} - \eta (\hat{y}_t - y_t) x_t$$
  Onde $\eta$ é a taxa de aprendizado, $x_t$ é o vetor de recursos (Odds e suas binarizações), e $y_t$ é a classe real observada (0 se $\text{Odd} \ge 3$, 1 caso contrário).
* **Vantagens**:
  - Elimina a dependência de bibliotecas pesadas (TensorFlow/Keras).
  - Execução instantânea (complexidade $O(d)$ por passo).
  - Garantias teóricas robustas de arrependimento acumulado (*regret bounds*).

---

## 2. Rastreamento com Fixed Share (Tracking the Best Expert)
* **Teoria (Capítulo 5.2 - Tracking the Best Expert)**:
  Em séries temporais longas com mudanças de comportamento (mudança de dinâmicas no mercado esportivo), o melhor modelo varia ao longo do tempo. O algoritmo **Fixed Share** permite "rastrear" o melhor especialista compartilhando uma fração $\alpha$ da probabilidade de confiança de cada um a cada iteração:
  $$v_{i,t} = w_{i,t-1} e^{-\eta \ell_t(i)}$$
  $$w_{i,t} = (1 - \alpha) v_{i,t} + \frac{\alpha}{N} \sum_{j=1}^N v_{j,t}$$
* **Implementação**:
  Definir especialistas com diferentes parametrizações ou focos de Odd (ex: especialistas focados em Odds baixas vs. Odds altas) e combinar dinamicamente suas previsões.

---

## 3. Ridge Regression Online (Vovk-Azoury-Warmuth)
* **Teoria (Capítulos 11.7 e 11.8 - The Elliptic Potential)**:
  Para previsão de valores contínuos sob perda quadrática, a regressão de cumeeira online (*Online Ridge Regression*) com o potencial elíptico (algoritmo VAW) oferece limites de regret logarítmicos $O(d \log n)$, muito melhores do que os limites sublineares $O(\sqrt{n})$ de algoritmos baseados em gradiente estático.
* **Implementação**:
  Atualização recursiva rápida da matriz de covariância invertida $A_t^{-1}$ a cada iteração usando a identidade de Sherman-Morrison:
  $$A_t^{-1} = A_{t-1}^{-1} - \frac{A_{t-1}^{-1} x_t x_t^\top A_{t-1}^{-1}}{1 + x_t^\top A_{t-1}^{-1} x_t}$$

---

## 4. Agregação por Pesos Exponenciais (Weighted Majority)
* **Teoria (Capítulo 2 - Weighted Average Prediction)**:
  Substituir palpites arbitrários por um comitê ponderado baseado na perda acumulada de cada heurística de entrada (palpites fuzzy, modelos de regressão, tendências direcionais). A confiança em cada palpite é ponderada por $p_{i,t} \propto e^{-\eta L_{i,t-1}}$.
