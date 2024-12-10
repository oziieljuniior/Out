import pandas as pd
import numpy as np
from scipy.stats import pearsonr, binom

from statsmodels.tsa.ar_model import AutoReg

import datetime
import time

import seaborn as sns
import matplotlib.pyplot as plt

## Carregar data
data1 = pd.read_csv('/home/darkcover/Documentos/Data/Out/Entrada.csv')

## Funcoes
### Função para gerar oscilação controlada, agora com referência ao modelo AR
def gerar_oscillacao(valor_inicial, incremento, previsao_ar, limite_inferior=0.28, limite_superior=0.72):
    osc_final = [valor_inicial]
    
    for i in range(1, len(previsao_ar)):
        probabilidade = np.random.rand()
        referencia_ar = previsao_ar[i]
                
        if probabilidade < 1/3:
            proximo_valor = osc_final[-1] + incremento
        elif probabilidade < 2/3:
            proximo_valor = osc_final[-1]
        else:
            proximo_valor = osc_final[-1] - incremento
        
        # Aplicar os limites e o controle da previsão AR
        proximo_valor = np.clip(proximo_valor, max(limite_inferior, referencia_ar - incremento), 
                                min(limite_superior, referencia_ar + incremento))
        
        osc_final.append(proximo_valor)
        
    return osc_final

# Função para calcular a tendência
def calcular_tendencia(novas_entradas, janela=60):
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
def prever_01s(novas_entradas, array, tamanho_previsao=120, limite_inferior=0.28, limite_superior=0.72):
    previsoes = []
    novas_entradas_fixas1 = novas_entradas
    fator_decaimento = fator_decaimento = 1 - min(0.05, np.var(novas_entradas[-60:]) / 10) # Para suavização
    desvio_padrao = np.std(array)
    ini = 0
    count = 0
    epsilon_ajustado = 0
    outro_ajuste = 0
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

        #print('*.')
        binomial1 = round(calcular_distribuicao_binomial(previsoes), 25)
        binomial2 = round(calcular_distribuicao_binomial(array), 25)
        mediass = sum(previsoes)/len(previsoes)
        #print(f'Binomial1(Previsões) -> {binomial1:.25f} \nBinomial2(Array_Entrada) -> {binomial2:.25f} \nMedia_Previsao -> {mediass}')
                
        error = binomial2 - binomial1
        diferenca = np.abs(error)
        
        epsilon_ajustado = 10**(-(25 - outro_ajuste)) 
        #print(f'error -> {error} \nDirença -> {diferenca} \nEpsilon --> {epsilon_ajustado}')
        if diferenca < epsilon_ajustado and mediass >= limite_inferior and mediass <= limite_superior:
            #print("Right ...")
            return previsoes
        elif count >= 1000:
            outro_ajuste = outro_ajuste + 2
            count = 0
            
        previsoes = []
        novas_entradas = novas_entradas_fixas1

### Salva os relatorios da serie temporal
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

## Producao
i, i0, i1, i2, i3, i4, i5, i6, i7, j2 = 0,0,0,0,0,0,0,0,0,0
array1 = []
tilt, logica1 = False, False
inteiro = int(input("Insira a posição da data ---> "))

while i <= 10000:
    print(24*'***')
    
    print(f'Posição data - {i}')
    if i <= inteiro:
        odd = data1['Entrada'][i].replace(",",".")
        print(f'Entrada -> {odd}')
    else:
        odd = input("Entrada -> ").replace(",",".")
    odd = float(odd)
    if odd == 0:
        break
    
    #Sequencia de if:
    count1 = 1.75
    array = []
    j1 = 1
    k1 = 1
    while count1 <= 3:
        #print('*-'*24, count1)
        if odd > count1:
            att1 = 1
        else:
            att1 = 0
        array.append(att1)
        
        if count1 == 2:
            k1 = j1
    
        j1 += 1
        count1 = count1 + 0.01
        count1 = np.round(count1, 2)
    
    
    print(f'Order: {k1} | Len(array): {len(array)}')
    
    if i == 0:
        matrix1 = np.array([array])
    else:
        matrix1 = np.vstack([matrix1, array])
    #print(matrix1)

    ## Visualização
    if i >= 60:
        matrix2 = matrix1[-60:, :]
        print(matrix2.shape)
        media = np.array([1/60 * np.sum(matrix2, axis=0)])
        
        if i == 60:
            matrix3 = media
        else:
            matrix3 = np.vstack([matrix3, media])
        ####Issue1.0 -> Confirmar se a variavel matrix3[i-60,19] corresponde aos dados reais
        print(f'Media2: {matrix3[i-60, (k1-2)]} | Tamanho amostra: {matrix3.shape}')

    if tilt is not False:
        print('o-o'*12)
        if logica1 is False:
            print(f'Rodada da predição: {j2 + 1}')
            logica1 = False
        else:
            if odd >= 2:
                i3 = i3 + 1
                i5 = i5 + 1
                i4 = i4 + 1
                i6 = i6 + 1
            else:   
                i4 = i4 + 1
                i6 = i6 + 1
        
            acuracia_geral = i5 / i6
            acuracia_local = i3 / i4

            print(f'Rodada da predição: {j2 + 1} \nAcuracia Local: {acuracia_local} | Acuracia Global: {acuracia_geral}')
            print('o-o'*12)

    if i >= 180 and (i % 60) == 0:
        i3, i4 = 0,0

        # Determinando o número de colunas
        num_colunas = matrix1.shape[1]
        print("Número de colunas:", num_colunas)

        # Acessando e exibindo cada coluna
        for m in range(num_colunas):
            coluna = matrix1[:, m]
            #print(coluna)
            # Ajustar o modelo AR ao array2
            if i <= 330:
                guitar = [1,20]
                model_ar = AutoReg(coluna, lags=guitar)  # Ajuste o lag conforme 
            if i > 330 and i <= 630:
                guitar = [1,60,61]
                model_ar = AutoReg(coluna, lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
            if i > 630 and i <= 930:
                guitar = [1,60,61,120,121]
                model_ar = AutoReg(coluna, lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
            if i > 930 and i <= 1230:
                guitar = [1,60,61,120,121,180,181]
                model_ar = AutoReg(coluna, lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
            if i > 1230 and i <= 1530:
                guitar = [1,60,61,120,121,180,181,240,241]
                model_ar = AutoReg(coluna, lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
            if i > 1530 and i <= 1830:
                guitar = [1,60,61,120,121,180,181,240,241,300,301]
                model_ar = AutoReg(coluna, lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
            if i > 1830 and i <= 2130:
                guitar = [1,60,61,120,121,180,181,240,241,300,301,360,361]
                model_ar = AutoReg(coluna, lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
            if i > 2130 and i <= 2430:
                guitar = [1,60,61,120,121,180,181,240,241,300,301,360,361,420,421]
                model_ar = AutoReg(coluna, lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
            if i > 2430:
                guitar = [1,60,61,120,121,180,181,240,241,300,301,360,361,420,421,480,481]
                model_ar = AutoReg(coluna, lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
            
            model_ar_fit = model_ar.fit()

            previsao_ar = model_ar_fit.predict(start = len(coluna), end = len(coluna) + 60)
            #print(f'Tamanho do previsa_ar: {len(previsao_ar)}')
            #print('.')
            # Exibir sumário do modelo AR
            #print(model_ar_fit.summary())

            # Coeficientes
            #print("Coeficientes:", model_ar_fit.params)

            # AIC
            #print("AIC:", model_ar_fit.aic)

            # BIC
            #print("BIC:", model_ar_fit.bic)

            # Exemplo de uso da função
            #salvar_resumo_ar(model_ar_fit, "resumo_modelo_ar.txt")
            max1 = np.max(matrix1[:, m])
            min1 = np.min(matrix1[:, m])

            novas_entradas = gerar_oscillacao(
                            valor_inicial=coluna[-1], 
                            incremento=1/60,
                            previsao_ar=previsao_ar,
                            limite_inferior=min1, 
                            limite_superior=max1)
            if m == 0:
                matrix4 = np.array([novas_entradas]).T
                #print(matrix4.shape)
                #time.sleep(2)
            else:
                array = np.array([novas_entradas]).T
                matrix4 = np.hstack([matrix4, array])
            #print('..')
            #print(f'Tamanho da matriz 4: {matrix4.shape}')
            #time.sleep(2)
            #print(f'Tamanho novas entradas: {len(novas_entradas)}')
            #print('...')
            proximas_entradas = prever_01s(
                                        novas_entradas[1:len(novas_entradas)], 
                                        array=matrix1[-60:, m], 
                                        tamanho_previsao=60,
                                        limite_inferior=min1, 
                                        limite_superior=max1 
                                        )
            #print(f'Tamanho proximas entradas: {len(proximas_entradas)}')
            #print('....')
            if m == 0:
                matrix5 = np.array([proximas_entradas]).T

            else:
                array = np.array([proximas_entradas]).T
                matrix5 = np.hstack([matrix5, array])
            
            predicao = np.sum(matrix5, axis=1)
            #print('.....')
        
    if i >= 180:
        print(len(predicao))
        m = i % 60
        j2 = m

        tilt = round((m * 0.01) + 1,2)
        array1.append(tilt)
        print(15*'---')
        
        if i7 <= 60:
            order = 1.5
        else:
            order = (np.sum(array1)) / len(array1)

        order = np.round(order,2)

        if tilt >= order:
            print("APOSTAR")
            logica1 = True
        else:
            logica1 = False


        print(f'Predição: {tilt} | Coeficiente de Corte: {order}')
        
        print(15*'---')

    i += 1
    print(24*'***')
    print('-')
    