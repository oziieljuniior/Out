import pandas as pd
import time

# Carrega os dados
data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
data1 = data1.drop(columns=['Unnamed: 0'])
data1 = data1.rename(columns={'Odd_Categoria': 'odd_saida'})
print("Data Carregada ...")

array_count, array_geral = [], []

media = 0
# Itera sobre o range especificado
for i in range(len(data1)):
    print(24*'*-')
    print(f'Saida: {i} \nMedia Geral: {media}')
    time.sleep(0.25)
             
    odd_saida = data1['odd_saida'][i]

    # Atualiza os arrays
    if odd_saida >= 5:
        array_count.append(1)
    else:
        array_count.append(0)
    
    if len(array_count) == 160:
        media = sum(array_count) / 160
        array_geral.append(array_count)
        array_count = []

        print(media)

print(array_geral)

array_geral = pd.DataFrame({'Listas': array_geral})

array_geral.to_csv('/home/darkcover/Documentos/Out/dados/Parte2/matrix.csv')
array_geral.to_excel('/home/darkcover/Documentos/Out/dados/Parte2/matrix.xlsx')