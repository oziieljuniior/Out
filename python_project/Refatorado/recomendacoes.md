# Recomendações Técnicas para a Refatoração do Simulador

Este documento detalha os problemas identificados nos arquivos originais do simulador de previsão sequencial e as estratégias adotadas para a sua refatoração, com base nas melhores práticas de engenharia de software e nos princípios teóricos de aprendizado online abordados no livro *Prediction, Learning, and Games* (Cesa-Bianchi & Lugosi, 2006).

---

## 1. Problemas Críticos Identificados

### A. O Bug do Fatiamento de Lista em `Placar60.atualizar` (Erro de Tipo)
* **Problema**: No método `atualizar` da classe `Placar60` (dentro de `Modulos/Placares.py`), quando o resultado da predição não era `1` e o tamanho do histórico era $\ge 59$, a variável `medida_pontual` era definida como:
  ```python
  medida_pontual = self.media_parray[-60:]
  ```
  Isso retornava um *slice* (uma lista de até 60 elementos) e a adicionava via `.append()` à lista geral. Isso fazia com que `media_parray` contivesse listas aninhadas em vez de valores decimais contínuos de acurácia. No loop principal do simulador, a chamada `round(m, 4)` quebrava com um erro de tipo (`TypeError: type list doesn't define __round__ method`).
* **Solução**: Alterar o fatiamento de `[-60:]` (slice) para `[-60]` (acesso a um único índice anterior), adicionando uma verificação de limite seguro para evitar `IndexError` se o histórico tiver menos de 60 elementos:
  ```python
  if len(self.media_parray) < 60:
      medida_pontual = 0.0
  else:
      medida_pontual = self.media_parray[-60]
  ```

### B. Gargalo de Desempenho por Retreinamento do Zero (Batch vs. Online Learning)
* **Problema**: O loop principal roda até $210.000$ iterações. A cada 60 passos, o código chamava a função `reden(...)` que compilava e treinava um modelo Keras do zero (`model.fit` por 150 épocas). Treinar uma rede neural profunda a partir do zero $3.500$ vezes com volumes crescentes de dados é computacionalmente proibitivo.
* **Solução**: Implementar o **treinamento incremental**. Se o modelo já foi inicializado anteriormente, o script reutiliza a instância do modelo e seus pesos, realizando apenas um ajuste fino (*fine-tuning*) com poucas épocas em vez de redefinir e treinar tudo do zero por 150 épocas. Isso acelera drasticamente a simulação e preserva o aprendizado online, assemelhando-se ao comportamento dos algoritmos online (como *Online Gradient Descent*) discutidos no livro.

### C. Bug no Cálculo da Métrica `F1Score`
* **Problema**: A classe customizada `F1Score` continha a operação `tf.argmax(..., axis=-1)`. Em uma rede neural de classificação binária com uma única saída sigmoidal, as previsões têm formato `(batch_size, 1)`. O `argmax` ao longo de um eixo de tamanho 1 sempre retorna `0`, tornando a métrica inutilizável.
* **Solução**: Corrigir a métrica para realizar limiarização (thresholding) no tensor de probabilidade ou utilizar o cálculo correto de Verdadeiros Positivos, Falsos Positivos e Falsos Negativos. Como a métrica não era compilada no modelo, podemos também limpá-la ou integrá-la corretamente se necessário.

### D. Código Morto e Poluição de Variáveis Globais
* **Problema**:
  - As funções `lista_predicao` e `ponderar_lista` e a classe de métrica `F1Score` estavam declaradas no arquivo, mas nunca eram utilizadas no fluxo do programa.
  - Várias variáveis globais (como `matrix1s`, `matrix1n`, `acerto2`, etc.) eram inicializadas no topo do script mas nunca eram manipuladas, gerando confusão na leitura.
* **Solução**: Remover as variáveis inativas e limpar as declarações redundantes para manter o código limpo, legível e otimizado.

### E. Falta de Portabilidade (Caminhos Absolutos Rígidos)
* **Problema**: O arquivo lia dados e salvava resultados em caminhos absolutos baseados no usuário da máquina (`/home/darkcover/...`), impossibilitando a execução em qualquer outro ambiente.
* **Solução**: Utilizar caminhos relativos baseados no diretório do script ou aceitar o caminho do arquivo de entrada como argumento dinâmico de execução.

### F. Erros de Grafia (Typos)
* **Problema**: O método `tranforsmar_final_matriz` no arquivo `Modulos/Vetores.py` possuía um erro de grafia ("tranforsmar" com "s" intruso).
* **Solução**: Renomear e corrigir a assinatura e as chamadas para `transformar_final_matriz`.

---

## 2. Estrutura da Subpasta Refatorada

Para garantir um ambiente de análise limpo e isolado, criamos a subpasta `/python_project/Refatorado` contendo:
* `codec1_1.py`: O script principal simulador, otimizado, sem código morto e adaptado com treinamento incremental.
* `Placares.py`: O módulo de pontuação estatística com o bug do fatiamento e possíveis divisões por zero corrigidos.
* `Vetores.py`: O módulo de processamento de Odds corrigido.
* `recomendacoes.md`: Este arquivo guia.

Os arquivos foram desacoplados de pastas externas de módulos, permitindo a importação direta local (`import Placares` e `import Vetores`), tornando o novo implemento totalmente independente e de fácil execução.
