#Guardar informações sobre os modelos treinados. Guarda-los como opção de teste. E guardar informações sobre acertos e guardar funções é opção possível.

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

    def __init__(self, valor_inicial, incremento, tamanho, individuo, dados_reais, pai1, pai2, data_teste, novas_entradas, array, janela=60, tamanho_previsao=120, limite_inferior=0.28, limite_superior=0.63,taxa_mutacao=0.01):
        self.valor_inicial = valor_inicial
        self.incremento = incremento
        self.tamanho = tamanho
        self.limite_inferior = limite_inferior
        self.limite_superior = limite_superior
        self.individuo = individuo
        self.dados_reais = dados_reais
        self.pail1 = pai1
        self.pail2 = pai2
        self.taxa_mutacao = taxa_mutacao
        self.data_teste = data_teste
        self.novas_entradas = novas_entradas 
        self.array = array
        self.tamanho_previsao = tamanho_previsao
        self.janela = janela

    # Função para gerar oscilação controlada com valor fixo de incremento, decremento ou manutenção do valor
    def gerar_oscillacao(self, valor_inicial, incremento, tamanho, limite_inferior=0.28, limite_superior=0.63):
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

    def fitness_function(self, individuo, dados_reais):
        amplitude, frequencia, offset, ruido = individuo
        previsoes = ModelIt.gerar_oscillacao(valor_inicial=dados_reais[-1], incremento=frequencia, tamanho=int(len(dados_reais)), limite_inferior=0.28, limite_superior=0.65)
        erro = np.mean(np.abs(previsoes - dados_reais))
        return -erro  # Fitness negativo porque queremos minimizar o erro


    def crossover(self, pai1, pai2):
        return [(gene1 + gene2) / 2 for gene1, gene2 in zip(pai1, pai2)]

    def mutacao(self,individuo, taxa_mutacao=0.01):
        return [gene + np.random.normal(0, taxa_mutacao) if np.random.rand() < 0.1 else gene for gene in individuo]

    def modelo(self, data_teste):
        populacao_tamanho = 240
        geracoes = 120
        taxa_mutacao = 1/60
        dados_reais = data_teste

        populacao = [np.random.uniform(0, 1, 4) for _ in range(populacao_tamanho)]

        for geracao in range(geracoes):
            fitness_scores = [ModelIt.fitness_function(individuo, dados_reais) for individuo in populacao]
            sorted_population = [populacao[i] for i in np.argsort(fitness_scores)]
            populacao = sorted_population[-populacao_tamanho//2:]
            nova_populacao = []
            for _ in range(populacao_tamanho // 2):
                idx_pai1, idx_pai2 = np.random.choice(len(populacao), 2, replace=False)
                pai1, pai2 = populacao[idx_pai1], populacao[idx_pai2]
                filho = ModelIt.crossover(pai1, pai2)
                filho = ModelIt.mutacao(filho, taxa_mutacao)
                nova_populacao.append(filho)
            populacao += nova_populacao

        melhor_individuo = populacao[np.argmax(fitness_scores)]
        amplitude, frequencia, offset, ruido = melhor_individuo
        print("Melhor solução:", melhor_individuo)
        return melhor_individuo

    # Função para calcular a tendência das últimas entradas
    def calcular_tendencia(self, novas_entradas, janela=60):
        diffs = np.diff(novas_entradas[-janela:])
        tendencia = np.mean(diffs)  # Tendência positiva se média está subindo, negativa se está descendo
        return tendencia

    def prever_entradas(sef, novas_entradas, array, tamanho_previsao=120, limite_inferior=0.28, limite_superior=0.63):
        previsoes = []
        for i in range(tamanho_previsao):
            valor_atual = novas_entradas[-1] if len(novas_entradas) > 0 else 0.5
            
            tendencia = ModelIt.calcular_tendencia(novas_entradas)

            variancia = np.var(array)  # Correção: removido o y da função

            probabilidade_de_1 = valor_atual + tendencia * variancia  # Ajuste a influência da tendência
            probabilidade_de_1 = np.clip(probabilidade_de_1, limite_inferior, limite_superior)
            
            previsao = 1 if np.random.rand() < probabilidade_de_1 else 0
            previsoes.append(previsao)
            
            novas_entradas = np.append(novas_entradas, probabilidade_de_1)
        
        return previsoes, variancia