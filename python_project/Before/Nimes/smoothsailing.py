#Guardar informações sobre os modelos treinados. Guarda-los como opção de teste. E guardar informações sobre acertos e guardar funções é opção possível.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, binomtest
import time

data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/Saidas/FUNCOES/DOUBLE - 17_09_s1.csv')

###FUNCOES

# Função para gerar oscilação controlada com os parâmetros do melhor indivíduo treinado
def gerar_oscillacao(
        tamanho, 
        individuo = [np.float64(0.5726568559014906), np.float64(0.0020658693527675536), np.float64(0.43684108427099644), np.float64(0.3801867392801998)], 
        valor_inicial=None, 
        limite_inferior=0.28, 
        limite_superior=0.63):
    
    amplitude, frequencia, offset, ruido = individuo

    valor_inicial = valor_inicial if valor_inicial is not None else offset
    osc_final = [valor_inicial]

    for i in range(1, tamanho):
        probabilidade = np.random.rand()

        # Ajuste o valor de oscilação com base na frequência, amplitude e adicione o ruído
        if probabilidade < 1/3:
            proximo_valor = osc_final[-1] + frequencia * amplitude + np.random.normal(0, ruido)
        elif probabilidade < 2/3:
            proximo_valor = osc_final[-1] + np.random.normal(0, ruido)  # Mantém o valor com ruído
        else:
            proximo_valor = osc_final[-1] - frequencia * amplitude + np.random.normal(0, ruido)

        # Limita o valor entre o limite inferior e superior
        proximo_valor = np.clip(proximo_valor, limite_inferior, limite_superior)
        osc_final.append(proximo_valor)

    return np.array(osc_final)

def fitness_function(individuo, dados_reais):
    amplitude, frequencia, offset, ruido = individuo
    previsoes = gerar_oscillacao(
        tamanho=int(len(dados_reais)), 
        individuo=individuo,
        valor_inicial=dados_reais[-1], 
        limite_inferior=0.28, 
        limite_superior=0.65)
    erro = np.mean(np.abs(previsoes - dados_reais))
    return -erro  # Fitness negativo porque queremos minimizar o erro


def crossover(pai1, pai2):
    return [(gene1 + gene2) / 2 for gene1, gene2 in zip(pai1, pai2)]

def mutacao(individuo, taxa_mutacao=0.01):
    return [gene + np.random.normal(0, taxa_mutacao) if np.random.rand() < 0.1 else gene for gene in individuo]

def modelo(data_teste):
    populacao_tamanho = 240
    geracoes = 120
    taxa_mutacao = 1/60
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

# Função para calcular a tendência das últimas entradas
def calcular_tendencia(novas_entradas, janela=60):
    diffs = np.diff(novas_entradas[-janela:])
    tendencia = np.mean(diffs)  # Tendência positiva se média está subindo, negativa se está descendo
    return tendencia

def prever_entradas(novas_entradas, array, tamanho_previsao=120, limite_inferior=0.28, limite_superior=0.63):
    previsoes = []
    for i in range(tamanho_previsao):
        valor_atual = novas_entradas[-1] if len(novas_entradas) > 0 else 0.5
        
        tendencia = calcular_tendencia(novas_entradas)

        variancia = np.var(array)  # Correção: removido o y da função

        probabilidade_de_1 = valor_atual + tendencia * variancia  # Ajuste a influência da tendência
        probabilidade_de_1 = np.clip(probabilidade_de_1, limite_inferior, limite_superior)
        
        previsao = 1 if np.random.rand() < probabilidade_de_1 else 0
        previsoes.append(previsao)
        
        novas_entradas = np.append(novas_entradas, probabilidade_de_1)
    
    return previsoes, variancia


#####DEVELOP
# Coleta de 120 entradas iniciais
i, j, l, k, m, by_sinal = 0, 0, 0, 0, 0, 0
data_teste, array1, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12, data_teste1, data_teste2, data_teste3, novas_entradas, saida1, saida2, saida3, saida4, saida5, saida6, proximas_entradas = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

order, order1 = np.zeros(181), np.zeros(181)

acertos = []

# Figuras para diferentes gráficos
fig, (ax, ax_corr) = plt.subplots(2, 1, figsize=(10, 12))

novas_entradas_fixas, correlacao_fixas = None, None  # Para manter as novas entradas fixas no gráfico

while i <= 1800:
    print(24*'*-')
    print(f'Rodada: {i}')
    
    if i <= 240:
        print(data1['Entrada'][i].replace(",", '.'))
        odd = float(data1['Entrada'][i].replace(",", '.'))
    else:
        odd = input("Insira o número da odd: ").replace(",",".")
    
    if odd == 0:
        break
        
    if float(odd) >= 2:
        array1.append(1)
    else:
        array1.append(0)

    if i >= 60:
        array2 = array1[i - 60: i]
        media = sum(array2)/60
        data_teste.append(media)

        desvpad = np.std(array2, ddof=1)
        data_teste1.append(desvpad)

        binomial_teste = binomtest((sum(array2)),len(array2),0.5,alternative='two-sided')
                
        if k == 119:
            k = k - 1
        else:
            k += 1
        
        print(k)
        print(f'Media60: {media} \nDesvio Padrão60: {desvpad} \nBinomial Estatistica: {binomial_teste} \nProximas entradas: {order[k]} | lenorder: {len(order)}')
        if len(order) != 181 and i > 180:
            m += 1
            if len(order1) >= m:
                print(f'Proximas Entradas da Predição Anterior: {order1[m]} | lenorder1 >> {len(order1)}')
            else:
                print("Calculando novo array")
           

    if i % 60 == 0 and i >= 120:
        
        print(f'Executando o modelo após {i} entradas coletadas inicialmente:')
        melhor_individuo = modelo(data_teste)
        amplitude, frequencia, offset, ruido = melhor_individuo
        print("Melhor solução:", melhor_individuo)
        
        time.sleep(10)
        print("Gerando novas entradas, a partir das últimas entradas:")
        incremento_fixo = 1/60
        novas_entradas = gerar_oscillacao(
            tamanho=120,
            individuo=melhor_individuo, 
            valor_inicial=data_teste[-1], incremento=incremento_fixo, limite_inferior=0.28, limite_superior=0.63)
        
        order1 = proximas_entradas[59:120]
        print(len(order1))
        m = 0
        time.sleep(10)
        
        
        proximas_entradas, variancia = prever_entradas(novas_entradas, array=array1[i-120:i], tamanho_previsao=120)
        
        k = 0
        
        order = proximas_entradas

        print(f'Entradas criadas das medias criada: {len(novas_entradas)} \nEntradas 0 e 1 criada: {proximas_entradas}')

        kil1 = np.concatenate((data_teste[i - 120: i], novas_entradas))
        kil2 = np.concatenate((array1, proximas_entradas))
        array5, array6 = [], []
        for j in range(len(array1) - 61, len(kil2)):
            array5 = kil2[j-60:j]
            desvpad_teste = np.std(array5, ddof=1)
            array6.append(desvpad_teste)

        print(len(kil1), len(kil2), len(array6))

        data_teste3 = []
        for l in range(120, 181):
            array7 = kil1[l - 60: l]
            array8 = array6[l - 60: l]
            correlacao_teste, p_value_teste = pearsonr(array7, array8)
            data_teste3.append(correlacao_teste)


        #time.sleep(10)

        # Deslocar as novas entradas para a direita
        x_novas_entradas = np.arange(len(data_teste), len(data_teste) + len(novas_entradas))
        xx_novas_entradas = np.arange(len(data_teste2), len(data_teste2) + len(data_teste3))

        if novas_entradas_fixas is None:
            novas_entradas_fixas = novas_entradas
            ax.plot(x_novas_entradas, novas_entradas_fixas, label='Novas Entradas (fixas)', color='blue', linestyle='--')
        if correlacao_fixas is None:
            correlacao_fixas = data_teste3
            ax_corr.plot(xx_novas_entradas, data_teste3, color = 'orange', linestyle = '--')

        plt.legend()
        plt.pause(0.01)

        j = 0
    if l == 1:
        break
        
    # Gráfico das médias atualizado constantemente
    if i >= 60:
        ax.clear()
        if novas_entradas_fixas is not None:
            ax.plot(x_novas_entradas, novas_entradas_fixas, label='Novas Entradas (fixas)', color='blue', linestyle='--')

        ax.plot(data_teste, label='Médias (atualizadas)', color='red')
        
        plt.legend()
        plt.pause(0.01)

    if i >= 120:
        array3 = data_teste[len(data_teste) - 60: len(data_teste)]
        array4 = data_teste1[len(data_teste1) - 60: len(data_teste1)]

        correlacao, p_valor = pearsonr(array3, array4)
        data_teste2.append(correlacao)

        print(f'Correlação de Pearson: {correlacao}')
        print(f'Valor-p: {p_valor}')

        ax_corr.clear()
               
        if correlacao_fixas is not None:
            ax_corr.plot(xx_novas_entradas, data_teste3, label = 'Correlação (Predição)', color = 'orange', linestyle = '--')

        # Atualizar gráfico de correlação
        ax_corr.plot(data_teste2, label='Correlação (Histórica)', color='green')
        ax_corr.set_title('Correlação ao longo do tempo')
        ax_corr.legend()
        plt.pause(0.01)

    if i >= 121 and j <= 58:
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
