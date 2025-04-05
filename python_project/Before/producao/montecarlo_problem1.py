import pandas as pd
import numpy as np

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
num_simulacoes = 500000  # Número de simulações de Monte Carlo
tamanho_array = 60  # Tamanho das sequências iniciais e previstas
media_desejada = 0.67
erro_toleravel = 0.05

# Função para gerar uma sequência binária com uma média especificada
def gerar_sequencia_binaria(tamanho, media):
    return np.random.binomial(1, media, tamanho)

# Função para realizar a simulação de Monte Carlo
def simulacao_monte_carlo(num_simulacoes, tamanho_array, media_desejada, erro_toleravel):
    sequencias_simuladas = []
    
    for _ in range(num_simulacoes):
        # Geração das primeiras 160 entradas
        sequencia_inicial = array1
        
        # Previsão das próximas 160 entradas com Monte Carlo
        sequencia_prevista = gerar_sequencia_binaria(tamanho_array, media_desejada)
        
        # Concatenar sequências para ter 320 entradas
        sequencia_total = np.concatenate((sequencia_inicial, sequencia_prevista))
        
        # Verificar se a média está dentro do intervalo desejado
        media_sequencia = np.mean(sequencia_total)
        if abs(media_sequencia - media_desejada) <= erro_toleravel:
            sequencias_simuladas.append(sequencia_total)
    
    return sequencias_simuladas

# Adicionando variáveis para armazenar as previsões e resultados reais
previsoes = []
resultados_reais = []

# Iterando sobre as próximas 60 entradas (da 161 até 221)
for i in range(160, 220):
    # Executando a simulação de Monte Carlo
    sequencias_geradas = simulacao_monte_carlo(num_simulacoes, tamanho_array, media_desejada, erro_toleravel)

    # Analisando as sequências geradas
    num_sequencias_validas = len(sequencias_geradas)
    print(f"Número de sequências válidas geradas: {num_sequencias_validas} de {num_simulacoes} simulações.")

    # Calculando a probabilidade da entrada atual ser 1
    entrada = [seq[160] for seq in sequencias_geradas]
    probabilidade_entrada = np.mean(entrada)
    print(f"Probabilidade de a entrada {i+1} ser 1: {probabilidade_entrada}")

    # Decidindo a previsão com base na probabilidade
    previsao = 1 if probabilidade_entrada >= 0.5 else 0
    previsoes.append(previsao)

    # Verificando a entrada real
    if data['odd_saida'][i] >= 5:
        resultado_real = 1
    else:
        resultado_real = 0
    resultados_reais.append(resultado_real)

    # Verificando se a previsão foi correta
    acerto = previsao == resultado_real
    print(f"Previsão para entrada {i+1}: {previsao}, Real: {resultado_real}, Acerto: {acerto}")

    # Atualizando array1 para a próxima iteração
    array1.append(resultado_real)
    array1.pop(0)

# Calculando a taxa de acerto
acertos = sum([1 for p, r in zip(previsoes, resultados_reais) if p == r])
print(f"Taxa de acerto: {acertos / 60:.2f}")