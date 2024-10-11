import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, binomtest
import time

###FUNCOES
data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k_1.csv')

# Criar DataFrame para salvar variáveis das funções bem-sucedidas
banco_funcoes_df = pd.DataFrame(columns=["valor_inicial", "incremento", "tamanho", "taxa_acerto", "harray"])

# Função para gerar oscilação controlada
def gerar_oscillacao(valor_inicial, incremento, tamanho, limite_inferior=0.28, limite_superior=0.63):
    osc_final = [valor_inicial]
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
    return osc_final

# Função para calcular a tendência
def calcular_tendencia(novas_entradas, janela=60):
    diffs = np.diff(novas_entradas[-janela:])
    tendencia = np.mean(diffs)
    return tendencia

# Função para prever com base nas funções armazenadas no DataFrame
def prever_com_banco(novas_entradas, array, tamanho_previsao=120, limite_inferior=0.28, limite_superior=0.63):
    previsoes = []
    
    # Se houver funções no banco, usar elas
    if not banco_funcoes_df.empty:
        print(f'{len(banco_funcoes_df)} funções carregadas do banco para comparação...')
        for index, row in banco_funcoes_df.iterrows():
            previsao = gerar_oscillacao(
                valor_inicial=row['valor_inicial'],
                incremento=row['incremento'],
                tamanho=tamanho_previsao,
                limite_inferior=limite_inferior,
                limite_superior=limite_superior
            )
            previsoes.append(previsao)
            
        # Selecionar a melhor previsão, por exemplo com a média mais alta
        melhor_previsao = max(previsoes, key=lambda x: np.mean(x))
        return melhor_previsao
    else:
        # Se não houver funções no banco, gerar novas previsões
        previsoes = []
        for i in range(tamanho_previsao):
            valor_atual = novas_entradas[-1] if len(novas_entradas) > 0 else 0.5
            tendencia = calcular_tendencia(novas_entradas)
            variancia = np.var(array)

            probabilidade_de_1 = valor_atual + tendencia * variancia
            probabilidade_de_1 = np.clip(probabilidade_de_1, limite_inferior, limite_superior)
            
            previsao = 1 if np.random.rand() < probabilidade_de_1 else 0
            previsoes.append(previsao)
            
            novas_entradas = np.append(novas_entradas, probabilidade_de_1)
        
        return previsoes

# Função para adicionar uma função de oscilação no banco de funções
def adicionar_funcao_ao_banco(valor_inicial, incremento, tamanho, taxa_acerto):
    global banco_funcoes_df
    nova_funcao = pd.DataFrame({
        "valor_inicial": [valor_inicial],
        "incremento": [incremento],
        "tamanho": [tamanho],
        "taxa_acerto": [taxa_acerto]
    })
    banco_funcoes_df = pd.concat([banco_funcoes_df, nova_funcao], ignore_index=True)
    print(f'Função com taxa de acerto {taxa_acerto} adicionada ao banco.')


#####DEVELOP
# Coleta de 120 entradas iniciais
i, j, l, k, by_sinal = 0, 0, 0, 0, 0
data_teste, array1, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12, array13, data_teste1, data_teste2, data_teste3, novas_entradas, saida1, saida2, saida3, saida4, saida5, saida6, proximas_entradas = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

order = np.zeros(162)

# Figuras para diferentes gráficos
fig, (ax, ax_corr) = plt.subplots(2, 1, figsize=(10, 12))

novas_entradas_fixas, correlacao_fixas = None, None  # Para manter as novas entradas fixas no gráfico

while i <= 1800:
    print(24*'*-')
    print(f'Rodada: {i}')
    
    while True:
        try:
            if i <= 5000:
                print(data1['Odd'][i])
                odd = float(data1['Odd'][i])
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
    if float(odd) == 0:
        print("Salvando os dados e encerrando a execução...")
        data1 = pd.DataFrame({"Entrada":array1})
        try:
            banco_funcoes_df = pd.read_csv('banco_funcoes.csv')
            print(f'{len(banco_funcoes_df)} funções carregadas com sucesso.')
            break
        except FileNotFoundError:
            print('Nenhum banco de funções encontrado. Criando um novo.')
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
        if len(proximas_entradas) != 0:
            order = proximas_entradas
            if k == 59:
                k = k - 1
        print(k)
        print(f'Media60: {media} \nDesvio Padrão60: {desvpad} \nBinomial Estatistica: {binomial_teste} \nProximas entradas: {order[k+1]} | lenorder: {len(order)}')
        
        k += 1
    if i % 60 == 0 and i >= 120:
        if i > 120:
            if i < 360:
                pergunta = 0
                print("Continuando a execução do loop...")  # Apenas continue normalmente
            elif i >=  360:    
                pergunta = int(input("Você ainda quer continuar modelando? 0S ou 1N: "))
                
            if pergunta == 0:
                print("Continuando a execução do loop...")  # Apenas continue normalmente
            else:
                print("Salvando os dados organizados ... ")
                # Salvando os dados de saída e saindo do loop
                data_saida = pd.DataFrame({
                    'Acertos_Reais_CTG': saida1,
                    'Acertos_Reais_PCT': saida2,
                    'Acertos_Predicao_CTG': saida3,
                    'Acertos_Predicao_PCT': saida4,
                    'Acertos1_Predicao': saida5,
                    'Acerto1_Predicao_PCT': saida6,
                    'Variancia': array12
                }).to_csv('/home/darkcover/Documentos/Out/dados/Saidas/Data_Saida.csv')
                l = 1
                break  # Sai do loop principal

        print("Gerando novas entradas, a partir das últimas entradas:")
        
        novas_entradas = gerar_oscillacao(
            valor_inicial=data_teste[-1], 
            incremento=1/60,
            tamanho=60,
            limite_inferior=0.28, 
            limite_superior=0.63)
        
        proximas_entradas = prever_com_banco(novas_entradas, array=array1[i-60:i], tamanho_previsao=60)

        k = 0

        print(f'Entradas criadas das medias criada: {len(novas_entradas)} \nEntradas 0 e 1 criada: {(proximas_entradas)}')
        
        for i2 in range(i, i + 60):
            if float(data1['Odd'][i2]) >= 2:
                array10.append(1)
            else:
                array10.append(0)
        

        for i1 in range(0,len(proximas_entradas)):
            if array10[i1] == proximas_entradas[i1]:
                array9.append(1)
                if array10[i1] == 1:
                    array13.append(1)
                else:
                    array13.append(0)
            else:
                array9.append(0)
        for i3 in range(0, len(array9)):
            if array9[i3] == 1:
                array11.append(1)
            else:
                array11.append(0)
        print(24*'*-')
        print(f'Qt. Real> {sum(array10)} | Porcentagem: {sum(array10) / len(array10)} \nQt. Variancia> {sum(array9)} | Porcentagem: {sum(array9) / len(array9)} \nQt. Acerto 1 Predicao> {sum(array11)} | Acerto 1 Predicao PCT: {sum(array11) / len(array11)} \nAcertos: {sum(array13)} | Acertos(%) >> {sum(array13)/len(array13)}')

        taxa_acerto1 = sum(array13)/len(array13)
        if taxa_acerto1 >= 0.5:
            adicionar_funcao_ao_banco(valor_inicial=array10[0], incremento=1/60,tamanho=60, taxa_acerto = taxa_acerto1)

        saida1.append(sum(array10)), saida2.append(sum(array10) / len(array10)), saida3.append(sum(array9)), saida4.append(sum(array9) / len(array9)), saida5.append(sum(array11)), saida6.append(sum(array11) / len(array11))

        array9, array10, array11, array13 = [], [], [], []

        time.sleep(30)

        kil1 = np.concatenate((data_teste[i - 120: i], novas_entradas))
        kil2 = np.concatenate((array1, proximas_entradas))
        array5, array6 = [], []
        for j in range(len(array1) - 61, len(kil2)):
            array5 = kil2[j-60:j]
            desvpad_teste = np.std(array5, ddof=1)
            array6.append(desvpad_teste)

        print(len(kil1), len(kil2), len(array6))

        data_teste3 = []
        for l in range(60, 121):
            array7 = kil1[l - 60: l]
            array8 = array6[l - 60: l]
            #print(len(array7), len(array8))
            correlacao_teste, p_value_teste = pearsonr(array7, array8)
            data_teste3.append(correlacao_teste)


        time.sleep(10)

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