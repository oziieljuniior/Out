import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, binom
import datetime
import os
import shutil
import time
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller


###Carregar Paths e Dados
data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/Saidas/FUNCOES/DOUBLE - 17_09_s1.csv')

path_data_modelos = '/home/darkcover/Documentos/Out/dados/Saidas/Metallica/Modelos_Listados/Novos/'
path_data_modelos_salvos = '/home/darkcover/Documentos/Out/dados/Saidas/Metallica/Modelos_Listados/Salvos/'

path_data_modelos_gerais = '/home/darkcover/Documentos/Out/dados/Saidas/Metallica/Modelo_Geral/ModelosGerais.csv'
###

###FUNCOES
# Função para gerar oscilação controlada, agora com referência ao modelo AR

def gerar_oscillacao(casas, valor_inicial, incremento, previsao_ar, limite_inferior=0.28, limite_superior=0.64):
    osc_final = [valor_inicial]
    valor = valor_inicial

    for i in range(0, len(previsao_ar)):
        if i == 0:
            probabilidade1 = valor_inicial
        else:
            probabilidade1 = round(previsao_ar[i-1],casas)
            
        probabilidade2 = round(previsao_ar[i],casas)
# Se a prob1 > prob2 entao a media decaiu, se a prob1 < prob2 entao a media subiu, se prob1 = prob2 entao a media se manteu
        
        if probabilidade1 < probabilidade2:
            proximo_valor = valor + incremento
        elif probabilidade1 > probabilidade2:
            proximo_valor = valor - incremento
        else:
            proximo_valor = valor
        
        # Aplicar os limites e o controle da previsão AR
        proximo_valor = np.clip(proximo_valor, limite_inferior, limite_superior)
        osc_final.append(proximo_valor)
        valor = proximo_valor  
    print(probabilidade1, probabilidade2)
    return osc_final

# Função para calcular a tendência
def calcular_tendencia(novas_entradas, janela=61):
    diffs = np.diff(novas_entradas[-janela:])
    tendencia = np.mean(diffs)
    return tendencia

def calcular_distribuicao_binomial(array):
    # Tamanho do array e número de 1s (sucessos)
    n = len(array)
    num_sucessos = sum(array)
    
    # Estimativa da probabilidade de sucesso (média dos 1s no array)
    prob_sucesso = num_sucessos / n
    
    # Função de distribuição binomial
    # pmf calcula a probabilidade de termos exatamente 'num_sucessos' sucessos em 'n' tentativas
    probabilidade_binomial = binom.pmf(num_sucessos, n, prob_sucesso)
    
    return probabilidade_binomial

# Função para prever com base nas funções armazenadas no DataFrame
def prever_01s(erro, novas_entradas, array, casas, tamanho_previsao, limite_inferior, limite_superior):
    previsoes = []
    novas_entradas_fixas1 = novas_entradas[1:len(novas_entradas)]
    fator_decaimento = fator_decaimento = 1 - min(0.05, np.var(novas_entradas[-60:]) / 10) # Para suavização
    desvio_padrao = np.std(array)
    ini = 0
    count = 0
    while ini == 0:
        count += 1

        for i in range(tamanho_previsao):
            valor_atual = novas_entradas[-1]
            tendencia = calcular_tendencia(novas_entradas)
            
            # Suavizar a tendência usando o fator de decaimento
            tendencia_suavizada = tendencia * fator_decaimento
            
            # Ajustar a probabilidade de 1 com a tendência suavizada e a variância
            probabilidade_de_1 = valor_atual + tendencia_suavizada * desvio_padrao
            probabilidade_de_1 = np.clip(probabilidade_de_1, limite_inferior, limite_superior)
            
            previsao = 1 if np.random.rand() < probabilidade_de_1 else 0
            previsoes.append(previsao)
            
            novas_entradas = np.append(novas_entradas, probabilidade_de_1)

        binomial1 = round(calcular_distribuicao_binomial(previsoes), casas)
        binomial2 = round(calcular_distribuicao_binomial(array), casas)
        mediass = sum(previsoes)/len(previsoes)

        error = abs(binomial1 - binomial2)

        if error <= erro  and mediass >= limite_inferior and mediass <= limite_superior:
            print(binomial1, binomial2)
            print(mediass)

            return previsoes, count
        
        elif count >= 1000:
            print(binomial1, binomial2)
            print(mediass)
            print("Previsoes Aleatorias vão aparecer ...")
            return previsoes, count

        previsoes = []
        novas_entradas = novas_entradas_fixas1

def salvar_resumo_ar(model_ar_fit, nome_arquivo="resumo_modelo_ar.txt"):
    # Obter o resumo do modelo
    resumo = model_ar_fit.summary()
    
    # Obter a data e hora atual
    data_atual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Abrir o arquivo no modo append ("a"), para adicionar conteúdo ao final
    with open(nome_arquivo, "a") as arquivo:
        # Adicionar separador para identificar cada execução
        arquivo.write("\n" + "="*40 + "\n")
        
        # Escrever a data no início do arquivo
        arquivo.write(f"Data das entradas: {data_atual}\n\n")
        
        # Escrever o resumo do modelo no arquivo
        arquivo.write(str(resumo))
        arquivo.write("\n" + "="*40 + "\n")  # Adicionar separador no final

    print(f"Resumo do modelo AR adicionado ao arquivo {nome_arquivo}")


###Testar o tipo de array que o model_fit gera e a quantidade
i0, i1, i2, i6, i7 = 0, 0, 0, 0, 0
i3 = 5
i4 = 15
i5 = 1
guitar = 60
TP, TN, FP, FN = 0, 0, 0, 0
array1, array2, array3, data_teste, data_teste1 = [], [], [], [], []

matriz_resultante1, matriz_resultante2 = None, None

while i0 <= 1800:
    print(24*'*-')
    print(f'Rodada: {i0}')

    #Rotacionar entradas. Ela pode ser realizada de duas maneiras, através de um banco de dados, ou através de entradas inseridas manualmente.
    while True:
        try:
            if i0 <= 275:
                print(data1['Entrada'][i0].replace(",",'.'))
                odd = float(data1['Entrada'][i0].replace(",",'.'))
                if odd == 0:
                    odd = 1
                    break
                else:
                    odd = odd
                    break
            else:
                odd = input("Insira o número da odd: ").replace(",",".")
                break
        except ValueError:
            print("Entrada inválida. Por favor, insira um número válido.")

    # Condição para salvar e sair ao digitar 0
    if float(odd) == 0 and i0 > 120:
        print("Encerrando a execução...")
        break
        
    if float(odd) >= 2:
        array1.append(1)
    else:
        array1.append(0)

    print('Entrada: ', array1[-1])

    if i0 >= 60:
        array2 = array1[i0 - 60: i0]
        media = sum(array2)/60
        data_teste.append(media)

        desvpad = np.std(array2, ddof=1)
        data_teste1.append(desvpad)
        print(f'Tamanho data_teste: {len(data_teste)} \nTamanho data_teste1: {len(data_teste1)}')
        
        binomial_teste = calcular_distribuicao_binomial(array2)
        
        print(f'Media60: {media} \nDesvio Padrão60: {desvpad} \nBinomial Estatistica: {binomial_teste}')

        #time.sleep(0.5)
    if i0 >= 240:
        i5 = 0
        while i2 == 0:
            # Ajustar o modelo AR ao array2
            guitar = 61
            i1 = 0
            i2 = 1
            while i1 == 0:
                i7 += 1
                model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
                model_ar_fit = model_ar.fit()

                # Exibir sumário do modelo AR
                #print(model_ar_fit.summary())

                previsao_ar = model_ar_fit.predict(start=len(data_teste), end=len(data_teste) + 60)
                #print(previsao_ar)
                #


                # Verificar se há valores maiores que 10
                valor_predeterminado1 = 0.28
                valor_predeterminado2 = 0.64
                
                resultado1 = np.any(previsao_ar <= valor_predeterminado1)
                resultado2 = np.any(previsao_ar >= valor_predeterminado2)

                if resultado1 == False and resultado2 == False:
                    print("Previsao das medias realizado com sucesso!")
                    break
                if i7 >= 5000:
                    break
            i1, i7 = 0,0

            # Exemplo de uso da função
            salvar_resumo_ar(model_ar_fit, "resumo_modelo_ar.txt")
            print(len(previsao_ar), type(previsao_ar))
            #print(previsao_ar[0], data_teste[-1])
            
            novas_entradas = gerar_oscillacao(
                casas = i3,
                valor_inicial=data_teste[-1], 
                incremento=1/60,
                previsao_ar=previsao_ar,
                limite_inferior=0.28, 
                limite_superior=0.63)
            
            print('*-*-*- NOVAS ENTRADAS MEDIAS')
            #print(novas_entradas)
            print(len(novas_entradas), type(novas_entradas))

            proximas_entradas, count = prever_01s(
                erro= i5,
                novas_entradas=novas_entradas[1: len(novas_entradas)], 
                array=array1[i0-60:i0], 
                casas = i4,
                tamanho_previsao=61,
                limite_inferior=0.28, 
                limite_superior=0.63)
            
            print(count, i5)
            if count >= 1000:
                i2 = 0
                i3 = 5
                i4 = 15
                i5 = 1
                
            i3 = i3 - 1
            i4 = i4 - 3
            i5 = i5 + 0.2
            print(i3, i4)

        i2 = 0
        print("*-*-*- PROXIMAS ENTRADAS 01's")
        #print(proximas_entradas)
        print(len(proximas_entradas), type(proximas_entradas))

        # Acessando arrays do sumário
        coeficientes = model_ar_fit.params          # Coeficientes estimados
        erros_padrao = model_ar_fit.bse             # Erros padrão dos coeficientes
        valores_p = model_ar_fit.pvalues            # Valores p dos coeficientes
        intervalos_conf = model_ar_fit.conf_int()   # Intervalos de confiança

        # Exibindo os resultados
        print("Coeficientes:", len(coeficientes))
        print("Erros padrão:", len(erros_padrao))
        print("Valores p:", len(valores_p))
        print("Intervalos de confiança:", len(intervalos_conf))

        print(24*'*-')

        matriz2 = np.array([proximas_entradas])
        
        #m1
        matriz_coef = np.array([coeficientes[1:len(coeficientes)]])
        #m2
        matriz_pvalue = 1 - np.array([valores_p[1: len(valores_p)]])

        m1dotm2 = np.dot(matriz_coef.T, matriz_pvalue)

        # Pegando os elementos da diagonal principal
        diagonal1 = np.array([np.diagonal(m1dotm2)])
        
        #indices
        indices = np.abs(diagonal1)
        
        #m1dotm2dotmatriz2
        vce = np.dot(matriz2.T, indices)
        
        #Entradas futuras
        diagonal2 = np.array(([np.diagonal(vce)]))

        
        if matriz_resultante1 is None:
            matriz_resultante1 = diagonal2
        else:
            matriz_resultante1 = np.vstack((matriz_resultante1, diagonal2))

        #print(matriz_resultante1)
        print(f'Matriz entradas ok: {matriz_resultante1.shape}')

        if matriz_resultante2 is None:
            matriz_resultante2 = diagonal1
        else:
            matriz_resultante2 = np.vstack((matriz_resultante2, diagonal1))
        
        print(f'Matriz indices ok: {matriz_resultante2.shape}')
        
    if i0 > 300:
        print("*-"*24)
        print(f'Previsão da entrada: {i0 + 1}')
    
        ghost_array1, ghost_array2 = [], []
        m = 60
        for name in range(matriz_resultante1.shape[0]-60, matriz_resultante1.shape[0]):
            #print(name, m)
            #print(matriz_resultante1[name, 1 + matriz_resultante1.shape[0] - name])
            ghost_array1.append(matriz_resultante1[name, m])
            ghost_array2.append(matriz_resultante2[name, m])
            m -= 1
        #print(ghost_array1)
        #print(len(ghost_array1))
        under = sum(ghost_array1)
        final = under / len(ghost_array1)
        array3.append(final)
        if i6 == 0:
            print(f'Probabilidade da previsão: {final}')
        else:
            print(f'Probabilidade da previsão: {final}')
            
            if array3[-2] >= 0.0225:
                predito = 1
            else:
                predito = 0
            real = array1[-1]
            if real == 1 and predito == 1:
                TP += 1  # True Positive
            elif real == 0 and predito == 0:
                TN += 1  # True Negative
            elif real == 0 and predito == 1:
                FP += 1  # False Positive
            elif real == 1 and predito == 0:
                FN += 1  # False Negative

            # Exibir a matriz de confusão
            print(24*'*')
            print("Matriz de Confusão:")
            print(f"TP (True Positive): {TP}")
            print(f"TN (True Negative): {TN}")
            print(f"FP (False Positive): {FP}")
            print(f"FN (False Negative): {FN}")
            print(24*'*')
            
            # A partir da matriz de confusão, você pode calcular outras métricas de performance, como:
            acuracia = (TP + TN) / (TP + TN + FP + FN)
            precisao = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) != 0 else 0

            print(f"Acurácia: {acuracia}")
            print(f"Precisão: {precisao}")
            print(f"Recall: {recall}")
            print(f"F1-Score: {f1_score}")
        
        #time.sleep(2)
        
        i6 += 1
    i0 += 1