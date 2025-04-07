import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, binomtest
import time
import os
import pickle
from datetime import datetime
import CarregarData as carregando

class ModelIt:
    def __init__(self, individuo=[np.float64(0.5726568559014906), np.float64(0.0020658693527675536), np.float64(0.43684108427099644), np.float64(0.3801867392801998)]):
        self.individuo = individuo

    # Função para gerar oscilação controlada com os parâmetros do melhor indivíduo treinado
    def gerar_oscillacao(self, tamanho, valor_inicial=None, limite_inferior=0.28, limite_superior=0.63):
        amplitude, frequencia, offset, ruido = self.individuo

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

    # Função de fitness para avaliar cada indivíduo
    def fitness_function(self, individuo, dados_reais):
        amplitude, frequencia, offset, ruido = individuo
        previsoes = self.gerar_oscillacao(tamanho=int(len(dados_reais)), valor_inicial=dados_reais[-1], limite_inferior=0.28, limite_superior=0.65)
        erro = np.mean(np.abs(previsoes - dados_reais))
        return -erro  # Fitness negativo porque queremos minimizar o erro

    # Função de crossover entre dois indivíduos
    def crossover(self, pai1, pai2):
        return [(gene1 + gene2) / 2 for gene1, gene2 in zip(pai1, pai2)]

    # Função de mutação para variar os genes do indivíduo
    def mutacao(self, individuo, taxa_mutacao=0.01):
        return [gene + np.random.normal(0, taxa_mutacao) if np.random.rand() < 0.1 else gene for gene in individuo]

    # Modelo evolutivo para encontrar o melhor indivíduo
    def modelo(self, data_teste):
        populacao_tamanho = 240
        geracoes = 120
        taxa_mutacao = 1/60
        dados_reais = data_teste
        
        populacao = [np.random.uniform(0, 1, 4) for _ in range(populacao_tamanho)]

        for geracao in range(geracoes):
            fitness_scores = [self.fitness_function(individuo, dados_reais) for individuo in populacao]
            sorted_population = [populacao[i] for i in np.argsort(fitness_scores)]
            populacao = sorted_population[-populacao_tamanho//2:]
            nova_populacao = []
            for _ in range(populacao_tamanho // 2):
                idx_pai1, idx_pai2 = np.random.choice(len(populacao), 2, replace=False)
                pai1, pai2 = populacao[idx_pai1], populacao[idx_pai2]
                filho = self.crossover(pai1, pai2)
                filho = self.mutacao(filho, taxa_mutacao)
                nova_populacao.append(filho)
            populacao += nova_populacao

        melhor_individuo = populacao[np.argmax(fitness_scores)]
        self.individuo = melhor_individuo  # Atualiza o indivíduo com o melhor encontrado
        print("Melhor solução:", melhor_individuo)
        return melhor_individuo

    # Função para calcular a tendência das últimas entradas
    def calcular_tendencia(self, novas_entradas, janela=60):
        diffs = np.diff(novas_entradas[-janela:])
        tendencia = np.mean(diffs)  # Tendência positiva se a média está subindo, negativa se está descendo
        return tendencia

    # Função para prever as próximas entradas com base em novas entradas e tendências
    def prever_entradas(self, novas_entradas, array, tamanho_previsao=120, limite_inferior=0.28, limite_superior=0.63):
        previsoes = []
        for i in range(tamanho_previsao):
            valor_atual = novas_entradas[-1] if len(novas_entradas) > 0 else 0.5
            
            tendencia = self.calcular_tendencia(novas_entradas)
            variancia = np.var(array)

            probabilidade_de_1 = valor_atual + tendencia * variancia  # Ajuste a influência da tendência
            probabilidade_de_1 = np.clip(probabilidade_de_1, limite_inferior, limite_superior)
            
            previsao = 1 if np.random.rand() < probabilidade_de_1 else 0
            previsoes.append(previsao)
            
            novas_entradas = np.append(novas_entradas, probabilidade_de_1)
        
        return previsoes, variancia
