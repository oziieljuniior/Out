import numpy as np
import pandas as pd

data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/DOUBLE - 17_09_s1.csv').replace(",",".", regex=True).astype(float)

# Função para gerar oscilação controlada com valor fixo de incremento, decremento ou manutenção do valor
def gerar_oscillacao(valor_inicial, incremento, tamanho, limite_inferior=0.28, limite_superior=0.63, media_inicial=None):
    if media_inicial is not None:
        osc_final = [media_inicial]
    else:
        osc_final = [valor_inicial]
    
    tamanho = int(tamanho)

    # Percorre a sequência ajustando o valor por incremento, decremento ou mantendo o valor
    for i in range(1, tamanho):
        probabilidade = np.random.rand()
        
        if probabilidade < 1/3:
            # Subir a média
            proximo_valor = osc_final[-1] + incremento
        elif probabilidade < 2/3:
            # Manter a média
            proximo_valor = osc_final[-1]
        else:
            # Descer a média
            proximo_valor = osc_final[-1] - incremento
        
        # Limitar o valor aos limites estabelecidos
        proximo_valor = np.clip(proximo_valor, limite_inferior, limite_superior)
        osc_final.append(proximo_valor)

    return np.array(osc_final)

# Função de fitness (erro médio absoluto)
def fitness_function(individuo, dados_reais):
    amplitude, frequencia, offset, ruido = individuo
    previsoes = gerar_oscillacao(amplitude, frequencia, int(len(dados_reais)), 0.28, 0.63, media_inicial=dados_reais[-1])
    erro = np.mean(np.abs(previsoes - dados_reais))
    return -erro  # Fitness negativo porque queremos minimizar o erro

# Função de crossover
def crossover(pai1, pai2):
    return [(gene1 + gene2) / 2 for gene1, gene2 in zip(pai1, pai2)]

# Função de mutação
def mutacao(individuo, taxa_mutacao=0.01):
    return [gene + np.random.normal(0, taxa_mutacao) if np.random.rand() < 0.1 else gene for gene in individuo]

# Função para rodar o modelo genético
def modelo(data_teste):
    # Parâmetros do algoritmo genético
    populacao_tamanho = 100
    geracoes = 500
    taxa_mutacao = 0.0015
    dados_reais = data_teste  # Dados pseudo-aleatórios iniciais

    # Inicializa uma população de indivíduos
    populacao = [np.random.uniform(0, 1, 4) for _ in range(populacao_tamanho)]  # Amplitude, freq, offset, ruído

    for geracao in range(geracoes):
        fitness_scores = [fitness_function(individuo, dados_reais) for individuo in populacao]
        sorted_population = [populacao[i] for i in np.argsort(fitness_scores)]
        populacao = sorted_population[-populacao_tamanho//2:]  # Elitismo
        nova_populacao = []
        for _ in range(populacao_tamanho // 2):
            idx_pai1, idx_pai2 = np.random.choice(len(populacao), 2, replace=False)
            pai1, pai2 = populacao[idx_pai1], populacao[idx_pai2]
            filho = crossover(pai1, pai2)
            filho = mutacao(filho, taxa_mutacao)
            nova_populacao.append(filho)
        populacao += nova_populacao

    # Melhor solução encontrada
    melhor_individuo = populacao[np.argmax(fitness_scores)]
    amplitude, frequencia, offset, ruido = melhor_individuo
    print("Melhor solução:", melhor_individuo)
    return melhor_individuo


# Coleta de 120 entradas iniciais
i, j, by_sinal = 0, 0, 0
data_teste, array1 =  [], []

while i <= 1280:
    print(24*'*-')
    print(f'Rodada: {i}')
    if i < 119:
        print(data1['Entrada'][i])
        odd = data1['Entrada'][i]
    else:
        odd = input("Insira o número da odd: ").replace(",",".")
        
    if float(odd) >= 2:
        array1.append(1)
    else:
        array1.append(0)

    if i >= 60:
        array2 = array1[i - 60: i]
        print(len(array2))
        media = sum(array2)/60
        data_teste.append(media)
        print(f'Media60: {media}')
        array2 = []

    if i % 60 == 0 and i >= 120:
        print(f'Executando o modelo após {i} entradas coletadas inicialmente:')
        # Executa o modelo com os 120 dados coletados inicialmente
        melhor_individuo = modelo(data_teste)
        amplitude, frequencia, offset, ruido = melhor_individuo
        print("Melhor solução:", melhor_individuo)
        
        print("Gerando novas entradas, a partir das últimas entradas:")
        # Gerando as próximas 60 previsões
        incremento_fixo = 0.016666666667  # Incremento fixo
        novas_entradas = gerar_oscillacao(valor_inicial=media, incremento=incremento_fixo, tamanho=60, limite_inferior=0.28, limite_superior=0.63)
        ##Adicionar um gráfico nessa parte
        print(f'Entradas criadas: {len(novas_entradas)}')  # Deve imprimir 60
        j = 0

    if i >= 120 and j <= 58:
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
    
    i += 1
