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


def gerar_oscillacao(valor_inicial, incremento, previsao_ar, limite_inferior=0.28, limite_superior=0.63):
    osc_final = [valor_inicial]
    valor = valor_inicial
    for i in range(0, len(previsao_ar)):
        if i == 0:
            probabilidade1 = valor_inicial
        else:
            probabilidade1 = round(previsao_ar[i - 1],4)
        
        probabilidade2 = round(previsao_ar[i],4)

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
def prever_01s(novas_entradas, array, tamanho_previsao=120, limite_inferior=0.28, limite_superior=0.63):
    previsoes = []
    novas_entradas_fixas1 = novas_entradas
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

        binomial1 = round(calcular_distribuicao_binomial(previsoes), 15)
        binomial2 = round(calcular_distribuicao_binomial(array), 15)
        print(binomial1, binomial2)
        mediass = sum(previsoes)/len(previsoes)
        print(mediass)

        if binomial1 == binomial2 and mediass >= limite_inferior and mediass <= limite_superior:
            
            return previsoes
        
        elif count >= 5000:
            print("Previsoes Aleatorias vão aparecer ...")
            return previsoes

        previsoes = []
        novas_entradas = novas_entradas_fixas1

def consultar_modelos_listados():
    # Verifica se a pasta de origem existe
    if not os.path.exists(path_data_modelos):
        print(f"O diretório {path_data_modelos} não existe.")
        return
    
    # Lista todos os arquivos .csv no diretório de origem
    arquivos_csv = [f for f in os.listdir(path_data_modelos) if f.endswith('.csv')]
    
    if len(arquivos_csv) == 0:
        print(f"A pasta {path_data_modelos} está vazia.")
        return
    
    # Verifica se a pasta de destino existe; se não, cria-a
    if not os.path.exists(path_data_modelos_salvos):
        print(f"O diretório {path_data_modelos} não existe.")
        return
    
    # Loop para consultar e salvar os arquivos .csv
    for arquivo in arquivos_csv:
        caminho_origem = os.path.join(path_data_modelos, arquivo)
        caminho_destino = os.path.join(path_data_modelos_salvos, arquivo)
        # Lê o arquivo de origem (df)
        df1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/Saidas/Metallica/Modelo_Geral/ModelosGerais.csv')
        
        # Lê o conteúdo do arquivo .csv
        try:

            # Lê o arquivo de origem (df)
            df = pd.read_csv(caminho_origem)
            # Seleciona apenas as colunas correspondentes de df para garantir compatibilidade
            
            df = df[['historico_01', 'historico_medias', 'historico_desvio_padrao', 'Predicao_Medias', 
                    'Predicao_01', 'Predicao_desvpad', 'Predicao_correlacao', 'Acuracia', 'Precisao', 
                    'Recall', 'F1-Score']]
            

            # Adiciona as novas linhas ao df1
            df1 = pd.concat([df1, df], ignore_index=True)

            # Salva o novo df1 com as linhas combinadas
            df1.to_csv('/home/darkcover/Documentos/Out/dados/Saidas/Metallica/Modelo_Geral/ModelosGerais.csv', index=False)

            print("Dados foram adicionados ao df1 e salvos com sucesso!")

            # Copia o arquivo para o diretório de destino
            shutil.move(caminho_origem, caminho_destino)
            print(f"Arquivo {arquivo} movido para {caminho_destino}.")
        
        
        except Exception as e:
            print(f"Erro ao processar o arquivo {arquivo}: {e}")
    return

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

###DEVELOP
# Coleta de 120 entradas iniciais
i, j, l, k, m, m1, n, n1, n2, by_sinal = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
data_teste, array1, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12, array13, data_teste1, data_teste2, data_teste3, novas_entradas, saida1, saida2, saida3, saida4, saida5, saida6, proximas_entradas = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

order = np.zeros(162)

# Figuras para diferentes gráficos
fig, (ax, ax_corr) = plt.subplots(2, 1, figsize=(10, 12))

novas_entradas_fixas, correlacao_fixas = None, None  # Para manter as novas entradas fixas no gráfico

while i <= 5000:
    print(24*'*-')
    print(f'Rodada: {i}')
    
    consultar_modelos_listados()
    
    #Rotacionar entradas. Ela pode ser realizada de duas maneiras, através de um banco de dados, ou através de entradas inseridas manualmente.
    while True:
        try:
            if i <= 240:
                print(data1['Entrada'][i].replace(",",'.'))
                odd = float(data1['Entrada'][i].replace(",",'.'))
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
    if float(odd) == 0 and i > 120:
        print("Encerrando a execução...")
        break
        
    if float(odd) >= 2:
        array1.append(1)
    else:
        array1.append(0)
    
    print('Entrada: ', array1[-1])

    if i >= 60:
        array2 = array1[i - 60: i]
        media = sum(array2)/60
        data_teste.append(media)

        desvpad = np.std(array2, ddof=1)
        data_teste1.append(desvpad)
        print(f'Tamanho data_teste: {len(data_teste)} \nTamanho data_teste1: {len(data_teste1)}')
        
        binomial_teste = calcular_distribuicao_binomial(array2)
        
        print(k)
        print(f'Media60: {media} \nDesvio Padrão60: {desvpad} \nBinomial Estatistica: {binomial_teste}')
        
        k += 1

#1.0    
    if i % 30 == 0 and i >= 180:
        n7 = 0
        while True:
            try:
                if len(novas_entradas) == 0:
                    break
                else:
                    while True:
                        try:
                            pergunta5 = int(input("Desejas salvar este modelo ? (0N e 1S)->"))
                            print(pergunta5, type(pergunta5))
                        
                            if pergunta5 == 1:
                                print("Salvando modelo ...")
                                data_e_hora = datetime.datetime.now()
                                timestamp = data_e_hora.strftime("%Y-%m-%d_%H-%M-%S")
                                nome1 = path_data_modelos + str(n2)+ '_' + timestamp + '.csv' 
                                n2 += 1
                                
                                data_saida = pd.DataFrame({
                                    'historico_01':[array2], 
                                    'historico_medias': [data_teste[len(data_teste) - 60: len(data_teste)]],
                                    'historico_desvio_padrao': [data_teste1[len(data_teste1) - 60: len(data_teste1)]],
                                    'Predicao_Medias': [novas_entradas],
                                    'Predicao_01': [proximas_entradas], 
                                    'Predicao_desvpad': [array6],
                                    'Predicao_correlacao': [data_teste3],
                                    'Acuracia': acuracia, 
                                    'Precisao': precisao, 
                                    'Recall': recall, 
                                    'F1-Score': f1_score
                                    })
                                
                                # Salva o DataFrame em um arquivo CSV
                                data_saida.to_csv(str(nome1))
                                print(f"Modelo salvo com sucesso no arquivo {nome1}")

                                print("Saindo do loop após salvar o modelo.")  # Adiciona uma verificação antes do break
                                
                                n7 = 1
                                break  # Encerra o loop após salvar o modelo
                
                            ########################################
                            # 1.2.5 : Atualizar o método de continuação de loop
                            if pergunta5 == 0:
                                # Se a entrada for 0, interrompe o loop sem salvar
                                print("Modelo não será salvo. Continuando o loop.")
                                
                                n7 = 1
                                break
                            ########################################
                            
                        except ValueError:
                            print("Entrada invalida, apenas 0 e 1 permitidos ...")
                        
                    if n7 == 1:
                        break

            except ValueError:
                print('Entrada incorreta ...')
        
        while True:
            try:
                pergunta6 = int(input("Queres parar de executar o modelo ?(0N e 1S) -->"))
                if pergunta6 == 0:
                    m = 0
                    break
                if pergunta6 == 1:
                    m = 1
                    m1 = 1
                    break
            except ValueError:
                print("Entrada invalida ...")

        if m1 == 1:
            break
#############################################################################################
# 1.5 : Atulizar método como se produz as novas_entradas e proximas_entradas
        while m == 0:
            print("Gerando novas entradas, a partir das últimas entradas:")
## Verificação das ultimas entradas com modelos já salvos;
            while True:
                try:
                    pergunta1 = int(input("Consultar banco de funções (0N e 1S) --> "))
                    pergunta4 = int(input("Além disso, desejas visualizar os acertos direto no banco de dados ? Ou desejas atualizar em tempo real ? (0Banco 1Real) -> "))
                    
                    if pergunta1 == 0:
                        kil1, kil2 = [], []
                        # Ajustar o modelo AR ao array2
                        if i <= 400:
                            guitar = 20
                            model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme 
                        if i > 400 and i <= 640:
                            guitar = 60
                            model_ar = AutoReg(data_teste[len(data_teste) - 360: len(data_teste)], lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF
                        if i > 640:
                                guitar = 60
                                model_ar = AutoReg(data_teste[len(data_teste) - 600: len(data_teste)], lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF 

                        model_ar_fit = model_ar.fit()

                        # Exibir sumário do modelo AR
                        print(model_ar_fit.summary())

                        # Exemplo de uso da função
                        salvar_resumo_ar(model_ar_fit, "resumo_modelo_ar.txt")

                        previsao_ar = model_ar_fit.predict(start=len(data_teste), end=len(data_teste) + 30)
                        novas_entradas = gerar_oscillacao(
                            valor_inicial=data_teste[-1], 
                            incremento=1/60,
                            previsao_ar=previsao_ar,
                            limite_inferior=0.28, 
                            limite_superior=0.63)
    
                        proximas_entradas = prever_01s(novas_entradas[1:len(novas_entradas)], 
                                                    array=array1[i-30:i], 
                                                    tamanho_previsao=30)

                        k = 0
                        m = 1
                        break

                                        
                    if pergunta1 == 1:
                        print(12*'*-')
                        n2 = 0
                        kil1, kil2 = [], []

                        # Leitura do dataframe
                        df = pd.read_csv(path_data_modelos_gerais)
                        mute1 = np.array(array2)

                        # Garantindo que 'data_saida2' seja inicializado corretamente
                        data_saida2 = pd.DataFrame(columns=['n_i', 'historico_01', 'historico_medias', 'historico_desvio_padrao', 'Predicao_Medias', 'Predicao_01', 'Predicao_desvpad', 'Predicao_correlacao', 'Acuracia', 'Precisao', 'Recall', 'F1-Score'])

                        # Itera sobre os valores da coluna 'historico_01'
                        for name in range(0, len(df['historico_01'])):
                            # Supondo que 'mute1' seja o array com o qual você está comparando
                            #print("Array1:")
                            #print(mute1)
                            
                            # Tentativa de converter a string da coluna 'historico_01' em um numpy array
                            try:
                                mute2 = np.array(eval(df['historico_01'][name]))  # Converte string para array
                            except:
                                print(f"Erro ao converter {df['historico_01'][name]}")
                                continue

                            #print("Array2:")
                            #print(mute2)

                            # Comparação entre os arrays
                            if np.array_equal(mute1, mute2):
                                print("Encontramos alguém com o mesmo array")

                                # Recupera os dados da linha correspondente
                                historico_01 = np.array(eval(df['historico_01'][name]))
                                historico_medias = np.array(eval(df['historico_medias'][name]))
                                historico_desvio_padrao = np.array(eval(df['historico_desvio_padrao'][name]))
                                novas_entradas = np.array(eval(df['Predicao_Medias'][name]))
                                proximas_entradas = np.array(eval(df['Predicao_01'][name]))
                                Predicao_desvpad = np.array(eval(df['Predicao_desvpad'][name]))
                                Predicao_correlacao = np.array(eval(df['Predicao_correlacao'][name]))
                                acuracia = df['Acuracia'][name]
                                precisao = df['Precisao'][name]
                                recall = df['Recall'][name]
                                f1_score = df['F1-Score'][name]

                                # Cria o dataframe 'data_saida1'
                                data_saida1 = pd.DataFrame({
                                    'n_i': n2,
                                    'historico_01': [historico_01], 
                                    'historico_medias': [historico_medias],
                                    'historico_desvio_padrao': [historico_desvio_padrao],
                                    'Predicao_Medias': [novas_entradas],
                                    'Predicao_01': [proximas_entradas], 
                                    'Predicao_desvpad': [Predicao_desvpad],
                                    'Predicao_correlacao': [Predicao_correlacao],
                                    'Acuracia': acuracia, 
                                    'Precisao': precisao, 
                                    'Recall': recall, 
                                    'F1-Score': f1_score
                                })

                                #print(data_saida1)

                                # Concatena 'data_saida1' com 'data_saida2'
                                data_saida2 = pd.concat([data_saida2, data_saida1], ignore_index=True)

                                
                                n2 += 1
                                n1 = 1

                        print(f'Quantidade de historico encontrados: {n2} \nn1: {n1}')
                        
                        if n1 == 1:
                            
                            for o in range(0,len(data_saida2)):
                                print(110*'_')
                                print(f'| n_i: {data_saida2['n_i'][o]} | Acuracia: {data_saida2['Acuracia'][o]} | Precisao: {data_saida2['Precisao'][o]} | Recall: {data_saida2['Recall'][o]} | F1-Score: {data_saida2['F1-Score'][o]} |')
                                print(110*'_')
                            print(110*'_')
                            n4 = 0
                            while n4 == 0:
                                n3 = int(input("Escolha sua entrada n_i: "))
                                print(n3, type(n3))

                                if n3 == -1:
                                    n1 = 0
                                    n4 = 1
                                    break

                                for o in range(0, len(data_saida2)):
                                    if int(n3) == o:
                                        print('Entrada selecionada: ', o)
                                        print("Avaliando historicos ...")
                                        print('Fase 1 - Analise de historico de 01 - Já Aprovado')
                                        
                                        historico_medias = data_saida2['historico_medias'][int(n3)]
                                        
                                        real_medias = np.array(data_teste[len(data_teste) - 60: len(data_teste)])
                                        
                                        print('Fase 2 - Analise de medias: ')
                                        if np.array_equal(historico_medias, real_medias):
                                            print("Aprovado ...")
                                            print("Fase 3 - Analise de desvio padrao")
                                            historico_desvio_padrao = np.array(data_saida2['historico_desvio_padrao'][int(n3)])

                                            real_desvio_padrao = np.array(data_teste1[len(data_teste1) - 60: len(data_teste1)])

                                            if np.array_equal(historico_desvio_padrao, real_desvio_padrao):
                                                print(f'Aprovado... \nNovas entradas e proximas entratadas foram herdadas ...')
                                                novas_entradas = np.array(data_saida2['Predicao_Medias'][int(n3)])
                                                proximas_entradas = np.array(data_saida2['Predicao_01'][int(n3)])
                                                n4 = 1
                                                n1 = 1
                                                k = 0

                                            else:
                                                print('Reprovado na fase 3.')
                                                print("Gerando uma nova oscilação ...")
                                                n4 = 1
                                                n1 = 0
                                                k = 0    
                                            
                                        else:
                                            print('Reprovado na fase 2.')
                                            print("Gerando uma nova oscilação ...")
                                            n4 = 1
                                            n1 = 0
                                            k = 0
                        
                        if n1 == 0:
                            kil1, kil2 = [], []
                            # Ajustar o modelo AR ao array2
                            if i <= 400:
                                guitar = 20
                                model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme 
                            if i > 400 and i <= 640:
                                guitar = 60
                                model_ar = AutoReg(data_teste[len(data_teste) - 360: len(data_teste)], lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF 
                            if i > 640:
                                guitar = 60
                                model_ar = AutoReg(data_teste[len(data_teste) - 600: len(data_teste)], lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF 

                            model_ar_fit = model_ar.fit()

                            # Exibir sumário do modelo AR
                            print(model_ar_fit.summary())

                            # Exemplo de uso da função
                            salvar_resumo_ar(model_ar_fit, "resumo_modelo_ar.txt")

                            previsao_ar = model_ar_fit.predict(start=len(data_teste), end=len(data_teste) + 30)
                            novas_entradas = gerar_oscillacao(
                                valor_inicial=data_teste[-1], 
                                incremento=1/60,
                                previsao_ar=previsao_ar,
                                limite_inferior=0.28, 
                                limite_superior=0.63)
                            
                            proximas_entradas = prever_01s(novas_entradas[1:len(novas_entradas)], 
                                                        array=array1[i-30:i], 
                                                        tamanho_previsao=30)

                            k = 0
                            n1 = 0
                                                    
                        break
                except ValueError:
                    print("Entrada Invalida. Algum erro ocorreu ...")

#############################################################################################



########################################
# 1.2   
            print(24*'*-')
            print(f'Entradas das medias criada: {len(novas_entradas)} \nEntradas 0 e 1 criada: {(proximas_entradas)}')

            kil1 = np.concatenate((data_teste[len(data_teste) - 60: len(data_teste)], novas_entradas))
            kil2 = np.concatenate((array1[len(array1) - 120: len(array1)], proximas_entradas))
            array5, array6 = [], []
            for j in range(60, len(kil2)):
                array5 = kil2[j - 60:j]
                desvpad_teste = np.std(array5, ddof=1)
                array6.append(desvpad_teste)

            print(len(kil1), len(kil2), len(array6))

            data_teste3 = []
            for l in range(60, 91):
                array7 = kil1[l - 60: l]
                array8 = array6[l - 60: l]
                #print(len(array7), len(array8))
                correlacao_teste, p_value_teste = pearsonr(array7, array8)
                data_teste3.append(correlacao_teste)

            if pergunta4 == 1:
                TP, TN, FP, FN = 0, 0, 0, 0
                order1 = kil1
                order2 = array6
                order3 = data_teste3
                n6 = 0
                m = 1
                break
    ## 1.2.2 : Adicionar a função de perguntar se quero realizar a predição a partir de um banco de dados, ou para coleta normal.
            
    ### Função que gera uma matriz de confusão a partir de uma nova entrada, proximas entradas e banco de dados
            
            

            if pergunta4 == 0:
                n5 = 0
                while n5 == 0:                    
                    #Determina as próximas 60 entradas
                    for i2 in range(i, i + 60):
                        if float(data1['Entrada'][i].replace(",",'.')) >= 2:
                            array10.append(1)
                        else:
                            array10.append(0)
                        
                    # Inicializar variáveis da matriz de confusão
                    TP, TN, FP, FN = 0, 0, 0, 0

                    # Comparar as entradas reais com as preditas
                    for i1 in range(len(proximas_entradas)):
                        real = array10[i1]
                        predito = proximas_entradas[i1]
                        
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

                    # Limpar os arrays para a próxima iteração
                    array9, array10, array11, array13 = [], [], [], []

                ########################################
                # 1.2.4 : Atualiar método de salvamento dos arrays de historico da medias e historico das predicoes 0 e 1s.
                    while True:
                        try:
                            pergunta2 = int(input("Desejas salvar este modelo ? (0N e 1S)->"))
                            print(pergunta2, type(pergunta2))
                        
                            if pergunta2 == 1:
                                print("Salvando modelo ...")
                                data_e_hora = datetime.datetime.now()
                                timestamp = data_e_hora.strftime("%Y-%m-%d_%H-%M-%S")
                                nome1 = path_data_modelos + str(n2)+ '_' + timestamp + '.csv' 
                                n2 += 1
                                #print(nome1)
                                
                                #Erro de array
                                #print(data_teste[len(data_teste) - 60: len(data_teste)])
                                #print(len(data_teste[len(data_teste) - 60: len(data_teste)]))
                                #print(data_teste1[len(data_teste1) - 60: len(data_teste1)])
                                #print(len(data_teste1[len(data_teste1) - 60: len(data_teste1)]))

                                #time.sleep(60)

                                ## Padronizado para historico medias e historico desvio padrao das últimas 60 entradas.

                                data_saida = pd.DataFrame({
                                    'historico_01':[array2], 
                                    'historico_medias': [data_teste[len(data_teste) - 60: len(data_teste)]],
                                    'historico_desvio_padrao': [data_teste1[len(data_teste1) - 60: len(data_teste1)]],
                                    'Predicao_Medias': [novas_entradas],
                                    'Predicao_01': [proximas_entradas], 
                                    'Predicao_desvpad': [array6],
                                    'Predicao_correlacao': [data_teste3],
                                    'Acuracia': acuracia, 
                                    'Precisao': precisao, 
                                    'Recall': recall, 
                                    'F1-Score': f1_score
                                    })
                                
                                # Salva o DataFrame em um arquivo CSV
                                data_saida.to_csv(str(nome1))
                                print(f"Modelo salvo com sucesso no arquivo {nome1}")

                                print("Saindo do loop após salvar o modelo.")  # Adiciona uma verificação antes do break
                                m = 1
                                n5 = 1
                                break  # Encerra o loop após salvar o modelo
                
                            ########################################
                            # 1.2.5 : Atualizar o método de continuação de loop
                            if pergunta2 == 0:
                                # Se a entrada for 0, interrompe o loop sem salvar
                                print("Modelo não será salvo. Continuando o loop.")
                                # Ajustar o modelo AR ao array2
                                if i <= 400:
                                    guitar = 20
                                    model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme 
                                if i > 400 and i <= 640:
                                    guitar = 60
                                    model_ar = AutoReg(data_teste[len(data_teste) - 360: len(data_teste)], lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF 
                                if i > 640:
                                    guitar = 60
                                    model_ar = AutoReg(data_teste[len(data_teste) - 600: len(data_teste)], lags=guitar)  # Ajuste o lag conforme model_ar = AutoReg(data_teste, lags=guitar)  # Ajuste o lag conforme sua análise de ACF 

                                model_ar_fit = model_ar.fit()

                                # Exibir sumário do modelo AR
                                print(model_ar_fit.summary())

                                # Exemplo de uso da função
                                salvar_resumo_ar(model_ar_fit, "resumo_modelo_ar.txt")

                                # Exibir sumário do modelo AR
                                print(model_ar_fit.summary())

                                previsao_ar = model_ar_fit.predict(start=len(data_teste), end=len(data_teste) + 30)
                                novas_entradas = gerar_oscillacao(
                                    valor_inicial=data_teste[-1], 
                                    incremento=1/60,
                                    previsao_ar=previsao_ar,
                                    limite_inferior=0.28, 
                                    limite_superior=0.63)
                                
                                proximas_entradas = prever_01s(novas_entradas[1:len(novas_entradas)], 
                                                            array=array1[i-30:i], 
                                                            tamanho_previsao=30)

                                k = 0
                                
                                #m = 1
                                n5 = 0
                                break
                            ########################################
                            
                        except ValueError:
                            print("Entrada invalida, apenas 0 e 1 permitidos ...")

            ########################################
                

        if m == 1:
            while True:
                try:
                    pergunta3 = int(input("Deseja continuar a executar o código ? (0S e 1N) -> "))
                    if pergunta3 == 0:
                        n = 0
                        m = 0
                        break
                    elif pergunta3 == 1:
                        n = 1
                        m = 0
                        break
                    else:
                        print("Entrada não é válida, tente novamente ...")


                except ValueError:
                    print("Entrada não lista, tente novamente ...")
        
        if n == 1:
            break

########################################

########################################
#1.4 : Aqui obtem-se o data_teste3. Além de analisar o gráfico de correlação entre o histórico e a predição.
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

########################################

########################################################## AJUSTANDO ESSA PARTE
    if i >= 180 and pergunta4 == 1:
        # Inicializar variáveis da matriz de confusão
        if n6 == 0:
            predito_futura = proximas_entradas[n6] 
            print(12*'*-')
            print(f'Predição: {predito_futura}')
            print(12*'*-')   

        else:
            predito_futura = proximas_entradas[n6] 
            print(12*'*-')
            print(f'Predição: {predito_futura}')
            print(12*'*-')   
            
            real = array1[-1]
            predito = proximas_entradas[n6-1]
            
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

        n6 += 1 
    
         
    
########################################

    if l == 1:
        break
        
    # Gráfico das médias atualizado constantemente
    if i >= 60:
        ax.clear()
        if novas_entradas_fixas is not None:
            ax.plot(x_novas_entradas, novas_entradas_fixas, label='Novas Entradas (fixas)', color='blue', linestyle='--')

        ax.plot(data_teste, label='Médias (atualizadas)', color='red')
        ax.set_title('Médias ao longo tempo')
        ax.legend()
        
        plt.legend()
        plt.pause(0.01)

    if i >= 180:
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

    i += 1