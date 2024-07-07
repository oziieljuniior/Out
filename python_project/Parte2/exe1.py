import pandas as pd
import numpy as np
import time

# Carrega os dados
data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
data1 = data1.drop(columns=['Unnamed: 0'])
data1 = data1.rename(columns={'Odd_Categoria': 'odd_saida'})
print("Data Carregada ...")

array_count = []
matrix_count = np.zeros((48, 10))  # Inicializa a matriz com a forma correta
apostar_count = [0]
media_array = []
rodada_aposta = []
apostar_matrix = []
sense, count, media_apostas, order = 0, 161, 0, 21
register1, register2 = 0, 0

# Itera sobre o range especificado
for i in range(len(data1)):
    print(f'Saida: {i} \nMedia Geral: {sum(array_count)/(i+1):.4f}')
    
    # Adiciona 1 ou 0 à lista array_count com base no valor de odd_saida
    if data1['odd_saida'][i] >= 5:
        array_count.append(1)
    else:
        array_count.append(0)
    
    # Verifica se array_count tem pelo menos 640 elementos
    if len(array_count) >= 640:
        rows = 48  # Número de linhas necessárias, de 160 a 640 com intervalo de 10
        cols = 10  # Número de colunas necessárias, representando os intervalos de soma

        # Calcula as médias para a matriz matrix_count
        for row in range(rows):
            start = 160 + row * 10  # Determina o ponto inicial para a soma
            for col in range(cols):
                interval = start + col  # Determina o intervalo de soma
                matrix_count[row, col] = sum(array_count[-interval:]) / interval  # Calcula a média
                if matrix_count[row, col] < 0.60 or matrix_count[row, col] > 0.69:
                    sense = 1
                    count = 0
                    register1, register2 = row, col

        # Cria um DataFrame pandas a partir da matriz para impressão mais legível
        df_matrix = pd.DataFrame(matrix_count, columns=[f'Interval {160 + row * 10 + col}' for col in range(cols)])
        print(df_matrix)
        
        if count <= 320 and sense == 1:
            if order <= 20:
                order += 1
                
            else:
                if data1['odd_saida'][i] >= 5:
                    apostar_count.append(1)
                else:
                    apostar_count.append(0)
                
                media_apostas = sum(apostar_count) / len(apostar_count)
                media_array.append(media_apostas)
                
                if media_apostas >= 0.73 and len(apostar_count) >= 20:
                    apostar_matrix.append(apostar_count)
                    apostar_count = [0]
                    order = 0
                    count = 321
                    sense = 0
                if len(apostar_count) >= 640 and media_apostas <= 0.69:
                    apostar_matrix.append(apostar_count)
                    apostar_count = [0]
                    count = 321
                    sense = 0
                    order = 0

                rodada_aposta.append(i)
                print(f'Quantidade de Apostas: {len(apostar_count)} \nMedia Apostas: {media_apostas} \nÚltima entrada: {array_count[-1]}')
                print(order)
                count += 1
                order += 1
                time.sleep(0.5)
            

            
                    
df = pd.DataFrame({'Media_Apostas': media_array, 'Rodada': rodada_aposta})
df.to_csv("/home/darkcover/Documentos/Out/dados/estudo6.csv")
df.to_excel("/home/darkcover/Documentos/Out/dados/estudo6.xlsx")

