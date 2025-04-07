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
import Models



#####DEVELOP

# Função para exibir o menu de opções com tratamento de exceções
def mostrar_menu():
    while True:
        print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        print("| 1. Listar modelos salvos                     |")
        print("| 2. Executar um modelo existente              |")
        print("| 3. Executar um novo modelo                   |")
        print("| 4. Visualizar dados                          |")
        print("| 5. Sair                                      |")
        print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        
        escolha = input("Escolha uma opção: ")

        try:
            if escolha == '1':
                listar_modelos_salvos()
            elif escolha == '2':
                modelo = ()
                executar_modelo_existente(escolha, modelo)
            elif escolha == '3':
                modelo = ()
                executar_novo_modelo(escolha,modelo)
            elif escolha == '4':
                visualizar_dados()
            elif escolha == '5':
                print("Saindo...")
                break
            else:
                print("Escolha inválida. Tente novamente.")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            continue


# Função para listar modelos salvos
def listar_modelos_salvos():
    caminho_pasta = '/home/darkcover/Documentos/Out/python_project/OTAMA/Funcoes'
    arquivos = [f for f in os.listdir(caminho_pasta) if f.endswith('.pkl')]
    
    if len(arquivos) == 0:
        print("Nenhum modelo salvo encontrado.")
    else:
        print("\nModelos disponíveis:")
        for i, arquivo in enumerate(arquivos):
            print(f"{i+1}. {arquivo}")
        input("Pressione Enter para continuar...")

# Função para carregar e executar um modelo existente
def executar_modelo_existente(escolha, modelo):
    caminho_pasta = '/home/darkcover/Documentos/Out/python_project/OTAMA/Funcoes'
    arquivos = [f for f in os.listdir(caminho_pasta) if f.endswith('.pkl')]

    if len(arquivos) == 0:
        print("Nenhum modelo salvo encontrado.")
        return

    print("\nModelos disponíveis:")
    for i, arquivo in enumerate(arquivos):
        print(f"{i+1}. {arquivo}")
    
    escolha = int(input("Escolha o número do modelo que deseja executar: "))

    if escolha < 1 or escolha > len(arquivos):
        print("Escolha inválida.")
        return

    caminho_arquivo = os.path.join(caminho_pasta, arquivos[escolha - 1])
    with open(caminho_arquivo, 'rb') as arquivo:
        modelo = pickle.load(arquivo)
    
    print(f"Executando o modelo {arquivos[escolha - 1]}...")
    # Aqui você pode chamar a função para executar o modelo carregado
    # exemplo: modelo.executar()
    #carregando.ModelManager().listar_modelos_salvos()
    print(modelo)
    executar_novo_modelo(escolha, modelo)

    input("Pressione Enter para continuar...")

# Função para visualizar dados
## Aqui posso impulsionar resumos, gráficos, tabelas e estatítiscas
###DEVEENTRAREMOBRA###
def visualizar_dados():
    caminho_pasta = '/home/darkcover/Documentos/Out/dados/Saidas/FUNCOES'
    arquivos = [f for f in os.listdir(caminho_pasta) if f.endswith('.csv')]
    
    if len(arquivos) == 0:
        print("Nenhum dado encontrado.")
    else:
        print("\nDados disponíveis:")
        for i, arquivo in enumerate(arquivos):
            print(f"{i+1}. {arquivo}")
    
    escolha = int(input("Escolha o número do dado que deseja visualizar: "))
    
    if escolha < 1 or escolha > len(arquivos):
        print("Escolha inválida.")
        return

    caminho_arquivo = os.path.join(caminho_pasta, arquivos[escolha - 1])
    print(f"Exibindo o conteúdo do arquivo {arquivos[escolha - 1]}...")
    with open(caminho_arquivo, 'r') as arquivo:
        conteudo = arquivo.read()
        print(conteudo)
    
    input("Pressione Enter para continuar...")


# Função para executar um novo modelo
def executar_novo_modelo(escolha, modelo):
    print("\nExecutando um modelo...")
    print(len(modelo), print(type(modelo)))
    print(escolha, type(escolha))
    DataGeral = carregando.ModelManager()

    # Coleta de 120 entradas iniciais
    i, j, l, k, m, by_sinal = 0, 0, 0, 0, 0, 0
    data_teste, array1, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12, data_teste1, data_teste2, data_teste3, novas_entradas, saida1, saida2, saida3, saida4, saida5, saida6, proximas_entradas = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    order, order1 = np.zeros(181), np.zeros(181)

    acertos = []

    # Figuras para diferentes gráficos
    fig, (ax, ax_corr) = plt.subplots(2, 1, figsize=(10, 12))

    novas_entradas_fixas, correlacao_fixas = None, None  # Para manter as novas entradas fixas no gráfico
    
    if int(escolha) == 1:
        # Carrega os dados referenciados
        data1 = DataGeral.carregar_data() 
        i = len(data1)  # Inicia a execução a partir da última entrada
        print(f"Iniciando a execução a partir da entrada {i} da data carregada.")

      
    while int(escolha) == 1:
        # Se houver mais entradas no data1, usar a entrada correspondente
        for i in range(0, i):
            print(24*'*-')
            print(f'Rodada: {i}')
            
            print(data1['Entrada'][i].replace(",", '.'))
            odd = float(data1['Entrada'][i].replace(",", '.'))

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
                
                print(f'Executando o modelo após {i} entradas coletadas inicialmente: \nModelo>> {modelo}')
                escolha, melhor_individuo = modelo
                amplitude, frequencia, offset, ruido = melhor_individuo
                print("Melhor solução:", melhor_individuo)
                
                print("Gerando novas entradas, a partir das últimas entradas:")
                incremento_fixo = 1/60
                novas_entradas = gerar_oscillacao(valor_inicial=data_teste[-1], incremento=incremento_fixo, tamanho=120, limite_inferior=0.28, limite_superior=0.63)
                
                DataGeral.salvar_modelo(novas_entradas, 'Ironic')

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
            print("Tamanho data = ", len(data1), '-----------',i)
            if i == len(data1):
                escolha = 3

    # Solicitar entrada manual quando iniciar do zero (escolha == 1) ou quando a entrada no arquivo não existir mais
    while int(escolha) == 3:
        print(24*'*-')
        print(f'Rodada: {i}')
        
        while True:
            try:
                odd = float(input("Entrada: ").replace(",", '.'))
                break
            except ValueError:
                print("Entrada inválida. Por favor, insira um número válido.")

        # Condição para salvar e sair ao digitar 0
        if float(odd) == 0:
            print("Salvando os dados e encerrando a execução...")
            data1 = pd.DataFrame({"Entrada":array1})
            DataGeral.salvar_data(data1 if escolha == 2 else array1, "IDIFY")
            return 
            
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
            
            while True:
                try:
                    pergunta1 = int(input("Queres utilizar o último modelo ? (0s e 1n): "))
                    print(pergunta1)
                    break
                except:
                    print("Entrada incorreta...")
            #Caso queira carregar o ultimo modelo é necessario listar modelos disponiveis, alem de configurar o modelo a partir dai
            if pergunta1 == 0:
                melhor_individuo = Models.ModelIt().modelo(data_teste=data_teste)
                amplitude, frequencia, offset, ruido = melhor_individuo
                print("Melhor solução:", melhor_individuo)

            #Caso nao queira carregar um modelo, é necessario continuar com o treinamento para executar um modelo para predicao
            elif pergunta1 == 1:
                melhor_individuo = modelo
                if melhor_individuo is None:
                    while True:
                        try:
                            print("Necessário carregar modelo, para isto: ")
                            print()
                            break
                        except:
                            print("Opção Invalida...")
                    

            print("Gerando novas entradas, a partir das últimas entradas:")
            incremento_fixo = 1/60
            #print(data_teste[-1], type(data_teste1[-1]))
            
            novas_entradas = Models.ModelIt(melhor_individuo=melhor_individuo).gerar_oscillacao(tamanho=120, valor_inicial=data_teste[-1], limite_inferior=0.28, limite_superior=0.63)
            
            DataGeral.salvar_modelo(novas_entradas, 'Ironic')

            order1 = proximas_entradas[59:120]
            print(len(order1))
            m = 0
            time.sleep(10)
            
            
            proximas_entradas, variancia = Models.ModelIt().prever_entradas(novas_entradas, array=array1[i-120:i], tamanho_previsao=120)
            
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
            return
            
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
    input("Pressione Enter para continuar...")

# Executar o menu
mostrar_menu()
