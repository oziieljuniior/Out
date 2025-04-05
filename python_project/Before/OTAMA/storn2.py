import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/DOUBLE - 17_09_s1.csv')

# Função para gerar oscilação limitada a uma variação máxima de ±0,02
def gerar_oscillacao(amplitude, frequencia, offset, ruido, tamanho):
    x_data = np.linspace(0, 1000, tamanho)
    osc = amplitude * np.sin(frequencia * x_data) + offset
    osc_final = osc + np.random.normal(0, abs(ruido), len(x_data))
    return np.array(osc_final)


# Função de fitness (erro médio absoluto)
def fitness_function(individuo, dados_reais):
    amplitude, frequencia, offset, ruido = individuo
    previsoes = gerar_oscillacao(amplitude, frequencia, offset, ruido, len(dados_reais))
    erro = np.mean(np.abs(previsoes - dados_reais))
    return -erro

# Função de crossover
def crossover(pai1, pai2):
    return [(gene1 + gene2) / 2 for gene1, gene2 in zip(pai1, pai2)]

# Função de mutação
def mutacao(individuo, taxa_mutacao=0.01):
    return [gene + np.random.normal(0, taxa_mutacao) if np.random.rand() < 0.1 else gene for gene in individuo]

# Função para rodar o modelo genético
def modelo(data_teste):
    populacao_tamanho = 1000
    geracoes = 1000
    taxa_mutacao = 0.0095
    dados_reais = data_teste

    populacao = [np.random.uniform(0, 1, 4) for _ in range(populacao_tamanho)]

    for geracao in range(geracoes):
        fitness_scores = [fitness_function(individuo, dados_reais) for individuo in populacao]
        sorted_population = [populacao[i] for i in np.argsort(fitness_scores)]
        populacao = sorted_population[-populacao_tamanho//2:]
        nova_populacao = []
        for _ in range(populacao_tamanho // 2):
            idx_pai1, idx_pai2 = np.random.choice(len(populacao), 2, replace=False)
            pai1, pai2 = populacao[idx_pai1], populacao[idx_pai2]
            filho = crossover(pai1, pai2)
            filho = mutacao(filho, taxa_mutacao)
            nova_populacao.append(filho)
        populacao += nova_populacao

    melhor_individuo = populacao[np.argmax(fitness_scores)]
    return melhor_individuo

# Coleta de 120 entradas iniciais
i, j, by_sinal = 0, 0, 0
data_teste, array1, array3, array4, data_teste1, data_teste2, novas_entradas = [], [], [], [], [], [], []

fig, ax = plt.subplots(figsize=(10, 8))  # Criar uma única área de plotagem

novas_entradas = []

while i <= 1280:
    print(24*'*-')
    print(f'Rodada: {i}')
    if i <= 721:
        print(data1['Entrada'][i].replace(",","."))
        odd = data1['Entrada'][i].replace(",", ".")
    else:
        odd = input("Insira o número da odd: ").replace(",", ".")
        print(odd)
        
    # Insere 0, 1 no array1
    if float(odd) >= 2:
        array1.append(1)
    else:
        array1.append(0)

    # Calcula a média das últimas 60 entradas e salva em data_teste
    if i >= 60:
        array2 = array1[i - 60: i]
        media = sum(array2) / 60
        data_teste.append(media)

        desvpad = np.std(array2, ddof=1)  # desvio padrão amostral
        data_teste1.append(desvpad)

        print(f'Media60: {media} \nDesvio Padrão60: {desvpad}')
        #print(len(array2))

    if i >= 122:
        # últimas 60 entradas das médias e desvios padrão
        array3 = data_teste[len(data_teste) - 60: len(data_teste)]
        array4 = data_teste1[len(data_teste1) - 60: len(data_teste1)]
        
        # Calcula a correlação de Pearson
        correlacao, p_valor = pearsonr(array3, array4)
        data_teste2.append(correlacao)

        print(f"Correlação de Pearson: {correlacao}")
        print(f"Valor-p: {p_valor}")

        # Atualiza o gráfico principal (dados gerais)
        ax.clear()
        ax.plot(data_teste2, label='Data Teste 2 (Correlação)', color='blue')

        # Se existirem novas entradas, plota no mesmo gráfico
        if len(novas_entradas) > 0:
            ax.plot(range(len(data_teste2), len(data_teste2) + len(novas_entradas)), novas_entradas[:len(novas_entradas)], label='Novas Entradas', color='green')
        
        ax.set_title('Gráfico Geral e Novas Entradas')
        ax.legend()
        
    if i % 60 == 0 and i >= 360:
        print(f'Executando o modelo após {i} entradas coletadas inicialmente:')
        melhor_individuo = modelo(data_teste2)
        amplitude, frequencia, offset, ruido = melhor_individuo
        novas_entradas = gerar_oscillacao(amplitude, frequencia, offset, abs(ruido), 60)
        j = 0

    if i >= 360 and j <= 58:
        if j >= 2:
            by_sinal = novas_entradas[j + 1] - novas_entradas[j]
        if by_sinal > 0 and j >= 1:
            print("Gráfico deve subir")
            print(f'Segundo a previsão: \nA média atual: {novas_entradas[j]} \nA média prevista: {novas_entradas[j + 1]}')
        elif by_sinal == 0 and j >= 1:
            print("Gráfico deve se manter o mesmo...")
            print(f'Segundo a previsão: \nA média atual: {novas_entradas[j]} \nA média prevista: {novas_entradas[j + 1]}')
        else:
            print('Gráfico deve descer')
            print(f'Segundo a previsão: \nA média atual: {novas_entradas[j]} \nA média prevista: {novas_entradas[j + 1]}')
        j += 1

    plt.pause(0.1)  # Pausa para atualizar os gráficos
    i += 1

plt.show()
