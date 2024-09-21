import numpy as np
import pandas as pd

array1, array2, array3, array4, array5,  arraygeral = [], [], [], [], [], []
count1, count2, count3, count4, i1, j1 = 108, 52, 160, 320, 1, 0

data = pd.read_csv("/home/darkcover/Documentos/Out/dados/odds_200k.csv")

data = data.drop(columns=['Unnamed: 0'])
data = data.rename(columns={'Odd_Categoria': 'odd_saida'})
print("Data Carregada ...")

for i in range(160):
    if data['odd_saida'][i] >= 5:
        zeroum = 1
    else:
        zeroum = 0
    array1.append(zeroum)


# Parâmetros do problema
num_simulacoes = 100000  # Número de simulações de Monte Carlo
tamanho_array = 160  # Tamanho das sequências iniciais e previstas
media_desejada = 0.67
erro_toleravel = 0.05
num_rodadas = 60  # Número de rodadas para predição (da entrada 161 até 260)

# Função para gerar uma sequência binária com uma média especificada
def gerar_sequencia_binaria(tamanho, media):
    return np.random.binomial(1, media, tamanho)

# Função para realizar a simulação de Monte Carlo e prever a próxima entrada
def previsao_monte_carlo(sequencia_atual, num_simulacoes, media_desejada, erro_toleravel):
    count_1 = 0
    
    for _ in range(num_simulacoes):
        # Geração da sequência simulada com Monte Carlo
        sequencia_prevista = gerar_sequencia_binaria(len(sequencia_atual), media_desejada)
        sequencia_total = np.concatenate((sequencia_atual, sequencia_prevista[:1]))  # Apenas prever a próxima entrada
        
        # Verificar se a média está dentro do intervalo desejado
        media_sequencia = np.mean(sequencia_total)
        if abs(media_sequencia - media_desejada) <= erro_toleravel:
            count_1 += sequencia_prevista[0]  # Contar se a primeira previsão é 1
    
    # Probabilidade da próxima entrada ser 1
    probabilidade_proxima_entrada = count_1 / num_simulacoes
    return probabilidade_proxima_entrada

# Geração da sequência inicial de 160 entradas
sequencia_atual = array1

# Armazenar resultados das previsões
previsoes = []
resultados = []

# Rodar previsões por 60 rodadas
for rodada in range(num_rodadas):
    probabilidade_1 = previsao_monte_carlo(sequencia_atual, num_simulacoes, media_desejada, erro_toleravel)
    previsao = 1 if probabilidade_1 >= 0.67 else 0  # Prever 1 se a probabilidade for >= 0.5, caso contrário 0
    
    # Gerar a entrada real para esta rodada
    entrada_real = gerar_sequencia_binaria(1, media_desejada)[0]
    
    # Armazenar resultados
    previsoes.append(previsao)
    resultados.append(entrada_real)
    
    # Atualizar a sequência com a entrada real
    sequencia_atual = np.append(sequencia_atual[1:], entrada_real)

# Comparar previsões com resultados reais
acertos = sum([1 for p, r in zip(previsoes, resultados) if p == r])
print(f"Número de acertos: {acertos} em {num_rodadas} rodadas.")
print(f"Precisão: {acertos / num_rodadas:.2f}")
