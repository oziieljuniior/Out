import pandas as pd
import numpy as np

# Função para converter uma odd em categoria
def converter_odd_em_categoria(i):
    if i < 1.05: return 1
    elif i < 1.15: return 2
    elif i < 1.3: return 3
    elif i < 1.45: return 4
    elif i < 1.7: return 5
    elif i < 2.1: return 6
    elif i < 2.6: return 7
    elif i < 3.5: return 8
    elif i < 5: return 9
    elif i < 10: return 10
    else: return 11

# Inicialização das variáveis
array_count, array_geral = [], []
matrix_count = np.zeros((48, 10))
apostar_count = [0]
media_array = []
rodada_aposta = []
apostar_matrix = []
sense, media_apostas, order = 0, 0, 21
register1, register2 = 0, 0
bag = 50
i = 1

# Função para calcular a média das últimas 640 jogadas
def calcular_media(array, interval):
    return sum(array[-interval:]) / interval

while i != 0:
    print("Entrada carregada ...")
    print(len(array_geral))
    print(24 * '*-')
             
    i = input("Insira a última entrada determinada: ")
    while i == '':
        i = input("Insira a última entrada determinada: ") 
    i = float(i.replace(',', '.'))
    if i == 0:
        break
    
    odd_saida = converter_odd_em_categoria(i)

    # Atualiza os arrays
    if odd_saida >= 5:
        array_count.append(1)
    else:
        array_count.append(0)
    
    array_geral.append(odd_saida)
    
    # Verifica se array_count tem pelo menos 640 elementos
    if len(array_count) >= 640:
        for row in range(48):
            start = 160 + row * 10
            for col in range(10):
                interval = start + col
                matrix_count[row, col] = calcular_media(array_count, interval)
                if matrix_count[row, col] < 0.60 or matrix_count[row, col] > 0.69:
                    sense = 1
                    register1, register2 = row, col

        # Cria um DataFrame pandas a partir da matriz para impressão mais legível
        df_matrix = pd.DataFrame(matrix_count, columns=[f'Interval {160 + row * 10 + col}' for col in range(10)])
        print(df_matrix)
        
        if sense == 1:
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
                    sense = 0
                if len(apostar_count) >= 640 and media_apostas <= 0.69:
                    apostar_matrix.append(apostar_count)
                    apostar_count = [0]
                    sense = 0
                    order = 0

                rodada_aposta.append(i)
                print(f'Quantidade de Apostas: {len(apostar_count)} \nMedia Apostas: {media_apostas} \nÚltima entrada: {array_count[-1]}')
                order += 1

# Salvar os resultados em arquivos
df = pd.DataFrame({'Media_Apostas': media_array, 'Rodada': rodada_aposta})
df.to_csv("/home/darkcover/Documentos/Out/dados/Parte2/estudo6.csv")
df.to_excel("/home/darkcover/Documentos/Out/dados/Parte2/estudo6.xlsx")

df1 = pd.DataFrame({'Odd_Saida': array_geral})
df1.to_excel("/home/darkcover/Documentos/Out/dados/Parte2/estudo7.xlsx")
