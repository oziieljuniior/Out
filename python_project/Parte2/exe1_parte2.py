import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Use o backend TkAgg para permitir gráficos interativos no terminal
plt.switch_backend('TkAgg')

# Carrega os dados
data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
data1 = data1.drop(columns=['Unnamed: 0'])
data1 = data1.rename(columns={'Odd_Categoria': 'odd_saida'})
print("Data Carregada ...")

# Inicialização das variáveis
array_count, array_geral, media_array, rodada_aposta, apostar_matrix = [], [], [], [], []
matrix_count = np.zeros((48, 10))
apostar_count, media_array1, x  = [0], [0], [0]
sense, media_apostas, order, register1, register2, i, p_value = 0, 0, 321, 0, 0, 1, 0.12

# Função para calcular a média das últimas jogadas
def calcular_media(array, interval):
    if len(array) < interval:
        return sum(array) / len(array)
    return sum(array[-interval:]) / interval

def plotar_grafico(x, y, media_geral):
    # Visualizar os resultados
    plt.clf()  # Limpa o gráfico anterior
    plt.plot(x, y, label='Médias')
    plt.axhline(y=media_geral, color='r', linestyle='--', label='Média Móvel')
    plt.text(len(x) - 1, media_geral, f'{media_geral:.5f}', color='r', ha='right', va='bottom')
    plt.xlabel('Número de Jogadas')
    plt.ylabel('Tendência')
    plt.title('Médias de Jogadas')
    plt.legend()
    plt.pause(0.01)  # Pausa para atualizar o gráfico

def calcular_pz(n, p_obs):
    # Teste de proporções Z manual
    p_esperado = 0.66  # proporção esperada (0.69 para 69%)
    z_score = (p_obs - p_esperado) / np.sqrt(p_esperado * (1 - p_esperado) / n)
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))  # p-valor

    return z_score, p_value

# Itera sobre o range especificado
for i in range(len(data1)):
    print(24*'*-')
    print(f'Saida: {i} \nMedia Geral: {sum(array_count)/(i+1)}')
             
    odd_saida = data1['odd_saida'][i]

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
                if matrix_count[row, col] >= 0.67:
                    sense = 1
                    register1, register2 = row, col

        # Cria um DataFrame pandas a partir da matriz para impressão mais legível
        df_matrix = pd.DataFrame(matrix_count, columns=[f'Interval {160 + row * 10 + col}' for col in range(10)])
        print(df_matrix)
        
        if sense == 1:
            if order <= 80:
                order += 1
            else:
                if odd_saida >= 5:
                    apostar_count.append(1)
                else:
                    apostar_count.append(0)
                
                media_apostas = sum(apostar_count) / len(apostar_count)
                x.append(len(apostar_count))
                media_array1.append(media_apostas)
                media_array.append(media_apostas)
                
                if media_apostas >= 0.73 and len(apostar_count) >= 7:
                    apostar_matrix.append(apostar_count)
                    apostar_count = [0]
                    media_array1 = [0]
                    x = [0]
                    order = 0
                    sense = 0
                if len(apostar_count) >= 480 and media_apostas <= 0.69:
                    apostar_matrix.append(apostar_count)
                    apostar_count = [0]
                    media_array1 = [0]
                    x = [0]
                    sense = 0
                    order = 0

                rodada_aposta.append(i)
                plotar_grafico(x, media_array1, media_apostas)
                sucessos, n = media_apostas, len(apostar_count)
                z_stat, p_value = calcular_pz(n, sucessos)
                
                print(f'Quantidade de Apostas: {len(apostar_count)} \nMedia Apostas: {media_apostas:.5f} \nÚltima entrada: {array_count[-1]} \nEstatística Z:{z_stat} \nValor p: {p_value}')
                order += 1

# Salvar os resultados em arquivos
df = pd.DataFrame({'Media_Apostas': media_array, 'Rodada': rodada_aposta})
df.to_csv("/home/darkcover/Documentos/Out/dados/Parte2/estudo6.csv")
df.to_excel("/home/darkcover/Documentos/Out/dados/Parte2/estudo6.xlsx")

df1 = pd.DataFrame({'Odd_Saida': array_geral})
df1.to_excel("/home/darkcover/Documentos/Out/dados/Parte2/estudo7.xlsx")
