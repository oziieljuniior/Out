import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/DOUBLE - 17_09_s1.csv')

# Função para gerar oscilação controlada com valor fixo de incremento, decremento ou manutenção do valor
def gerar_oscillacao(valor_inicial, incremento, tamanho, limite_inferior=0.28, limite_superior=0.63, media_inicial=None):
    if media_inicial is not None:
        osc_final = [media_inicial]
    else:
        osc_final = [valor_inicial]
    
    tamanho = int(tamanho)

    for i in range(1, tamanho):
        probabilidade = np.random.rand()
        
        if probabilidade < 1/3:
            proximo_valor = osc_final[-1] + incremento
        elif probabilidade < 2/3:
            proximo_valor = osc_final[-1]
        else:
            proximo_valor = osc_final[-1] - incremento
        
        proximo_valor = np.clip(proximo_valor, limite_inferior, limite_superior)
        osc_final.append(proximo_valor)

    return np.array(osc_final)

def fitness_function(individuo, dados_reais):
    amplitude, frequencia, offset, ruido = individuo
    previsoes = gerar_oscillacao(amplitude, frequencia, int(len(dados_reais)), 0.28, 0.63, media_inicial=dados_reais[-1])
    erro = np.mean(np.abs(previsoes - dados_reais))
    return -erro  # Fitness negativo porque queremos minimizar o erro

def crossover(pai1, pai2):
    return [(gene1 + gene2) / 2 for gene1, gene2 in zip(pai1, pai2)]

def mutacao(individuo, taxa_mutacao=0.01):
    return [gene + np.random.normal(0, taxa_mutacao) if np.random.rand() < 0.1 else gene for gene in individuo]

def modelo(data_teste):
    populacao_tamanho = 100
    geracoes = 500
    taxa_mutacao = 0.0015
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
    amplitude, frequencia, offset, ruido = melhor_individuo
    print("Melhor solução:", melhor_individuo)
    return melhor_individuo

# Coleta de 120 entradas iniciais
i, j, by_sinal = 0, 0, 0
data_teste, array1, novas_entradas =  [], [], []

fig, ax = plt.subplots(figsize=(10, 8))

novas_entradas_fixas = None  # Para manter as novas entradas fixas no gráfico

while i <= 1280:
    print(24*'*-')
    print(f'Rodada: {i}')
    if i <= 640:
        print(data1['Entrada'][i].replace(",","."))
        odd = float(data1['Entrada'][i].replace(",","."))
    else:
        odd = input("Insira o número da odd: ").replace(",",".")
        
    if float(odd) >= 2:
        array1.append(1)
    else:
        array1.append(0)

    if i >= 60:
        array2 = array1[i - 60: i]
        media = sum(array2)/60
        data_teste.append(media)
        print(f'Media60: {media}')
        array2 = []

    if i % 120 == 0 and i >= 120:
        print(f'Executando o modelo após {i} entradas coletadas inicialmente:')
        melhor_individuo = modelo(data_teste)
        amplitude, frequencia, offset, ruido = melhor_individuo
        print("Melhor solução:", melhor_individuo)
        
        print("Gerando novas entradas, a partir das últimas entradas:")
        incremento_fixo = 1/60
        novas_entradas = gerar_oscillacao(valor_inicial=media, incremento=incremento_fixo, tamanho=120, limite_inferior=0.28, limite_superior=0.63)
        
        print(f'Entradas criadas: {len(novas_entradas)}')

        # Deslocar as novas entradas para a direita
        x_novas_entradas = np.arange(len(data_teste), len(data_teste) + len(novas_entradas))

        if novas_entradas_fixas is None:
            # Plotar as novas entradas deslocadas (fixas no gráfico)
            novas_entradas_fixas = novas_entradas
            ax.plot(x_novas_entradas, novas_entradas_fixas, label='Novas Entradas (fixas)', color='blue', linestyle='--')
        
        # Mostrar o gráfico atualizado
        plt.legend()
        plt.pause(0.01)
    
        j = 0

    # Gráfico das médias atualizado constantemente
    if i >= 60:
        ax.clear()  # Limpa o gráfico para redesenhar as médias
        if novas_entradas_fixas is not None:
            # Plotar as novas entradas fixas e deslocadas
            ax.plot(x_novas_entradas, novas_entradas_fixas, label='Novas Entradas (fixas)', color='blue', linestyle='--')
        # Plotar as médias atualizadas
        ax.plot(data_teste, label='Médias (atualizadas)', color='red')
        
        plt.legend()
        plt.pause(0.01)

    if i >= 120 and j <= 58:
        if j >= 2:
            by_sinal = novas_entradas[j + 1] - novas_entradas[j]
        if by_sinal > 0 and j >= 1:
            print("Gráfico deve subir")
        elif by_sinal == 0 and j >= 1:
            print("Gráfico deve se manter o mesmo...")
        else:
            print('Gráfico deve descer')
        j += 1
    
    i += 1

plt.show()
