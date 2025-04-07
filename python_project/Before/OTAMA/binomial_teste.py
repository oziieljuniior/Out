import numpy as np
from scipy.stats import binom, binomtest, norm

# Parâmetros gerais
n = 320  # número de entradas na amostra
p = 0.5  # probabilidade de sucesso (0 ou 1) em cada ensaio

# Gera uma sequência binária aleatória simulando uma amostra de 0 e 1 com p = 0.5
np.random.seed(42)  # para reprodutibilidade
amostra = np.random.binomial(1, p, n)

# Predição de eventos binários
# Baseado em sucessos anteriores (número de 1's), prever a probabilidade de sucesso nas próximas entradas
sucessos_anteriores = np.sum(amostra)  # número de 1's na amostra
prob_sucesso = sucessos_anteriores / n  # proporção de 1's na amostra

# Intervalos de confiança para o número de 1's (usando a distribuição binomial)
alpha = 0.05  # nível de significância de 5%
interval_conf = binom.interval(1-alpha, n, p)

# Teste de hipóteses
# Se suspeitarmos que a proporção de 1's não é 0.5, podemos usar um teste binomial
statistic, p_value = binomtest(sucessos_anteriores, n, p, alternative='two-sided'), binomtest(sucessos_anteriores, n, p)

# Resultados
sucessos_anteriores, prob_sucesso, interval_conf, statistic, p_value

print(sucessos_anteriores, prob_sucesso, interval_conf, statistic, p_value)