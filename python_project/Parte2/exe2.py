import pandas as pd
import numpy as np

array_count, array_geral = [], []
matrix_count = np.zeros((48, 10))  # Inicializa a matriz com a forma correta
apostar_count = [0]
media_array = []
rodada_aposta = []
apostar_matrix = []
sense, count, media_apostas, order = 0, 161, 0, 21
register1, register2 = 0, 0
bag = 50
i = 1

# Itera sobre o range especificado
#Começamos o jogo aqui, com a primeira entrada gerada.
while i != 0:
    print("Entrada carregada ...")
    print(len(array_geral))
    print(24*'*-')
             
    i = input("Insira a última entrada determinada: ")
    i = float(i.replace(',','.'))
    if i == '':
        i = input("Insira a última entrada determinada: ") 
    if i == 0:
        break
    
    if i < 10:
        if i < 5:
            if i < 3.5:
                if i < 2.6:
                    if i < 2.1:
                        if i < 1.7:
                            if i < 1.45:
                                if i < 1.3:
                                    if i < 1.15:
                                        if i < 1.05:
                                            odd_saida = 1
                                        else:
                                            odd_saida = 2
                                    else:
                                        odd_saida = 3
                                else:
                                    odd_saida = 4
                            else:
                                odd_saida = 5
                        else:
                            odd_saida = 6
                    else:
                        odd_saida = 7
                else:
                    odd_saida = 8
            else:
                odd_saida = 9
        else:
            odd_saida = 10
    else:
        odd_saida = 11

    if odd_saida >= 5:
        array_count.append(1)
    else:
        array_count.append(0)
    
    array_geral.append(odd_saida)
    
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
                if odd_saida >= 5:
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
            
            
                    
df = pd.DataFrame({'Media_Apostas': media_array, 'Rodada': rodada_aposta})
df.to_csv("/home/darkcover/Documentos/Out/dados/estudo6.csv")
df.to_excel("/home/darkcover/Documentos/Out/dados/estudo6.xlsx")

df1 = pd.DataFrame({'Odd_Saida': array_geral}).to_excel("/home/darkcover/Documentos/Out/dados/estudo7.xlsx")

