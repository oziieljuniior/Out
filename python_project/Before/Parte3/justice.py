import pandas as pd
import numpy as np
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import zlib
import time

# Use o backend TkAgg para permitir gráficos interativos no terminal
plt.switch_backend('TkAgg')


# Carrega os dados
data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
data1 = data1.drop(columns=['Unnamed: 0'])
data1 = data1.rename(columns={'Odd_Categoria': 'odd_saida'})

data2 = pd.read_csv('/home/darkcover/Documentos/Out/dados/Parte2/universe1.csv')
data2 = data2.drop(columns=['Unnamed: 0'])

print("Data Carregada ...")

# Carregar Funções
def calcular_pz(n, p_obs):
    # Teste de proporções Z manual
    p_esperado = 0.66  # proporção esperada (0.66 para 66%)
    z_score = (p_obs - p_esperado) / np.sqrt(p_esperado * (1 - p_esperado) / n)
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))  # p-valor

    return z_score, p_value

def testar_incompressibilidade(sequence):
    # Verifica a incompressibilidade usando a complexidade de Kolmogorov
    compressed_length = len(zlib.compress(sequence))
    if len(sequence) == 0:
        return 0
    else:
        return compressed_length / len(sequence)

def testar_martin_lof(sequence):
    # Implementação simplificada de um teste de Martin-Löf para aleatoriedade
    n = len(sequence)
    k = int(np.log2(n))
    subsets = [sequence[i::k] for i in range(k)]
    subset_means = [np.mean(subset) for subset in subsets]
    chi_squared = sum(((mean - 0.5) ** 2) / 0.5 for mean in subset_means)
    p_value = 1 - norm.cdf(np.sqrt(chi_squared))
    return chi_squared, p_value
print("Funções Carregadas ...")

array_count1, array_count2, array_geral, array_provavel = [], [], [], []

media, i1 = 0, 1
# Itera sobre o range especificado
for i in range(len(data1)):
    print(24*'*-')
    print(f'Saida: {i} \nMedia Geral: {media}')
    #time.sleep(0.25)
             
    odd_saida = data1['odd_saida'][i]

    # Atualiza os arrays
    if odd_saida >= 5:
        array_count1.append(1)
        array_count2.append(1)
        array_geral.append(1)
    else:
        array_count1.append(0)
        array_count2.append(1)
        array_geral.append(0)
    t2 = len(array_count1)
    t3 = len(array_count2)
    t4 = len(array_geral)
    if t2 == 160 or t2 == 185:
        media = sum(array_count1) / 185
        
        if t2 == 160: 
            name1 = array_count1
            t1 = len(data2)
            # Iterando sobre cada string em data['Listas']
            for i in range(t1):
                # Avaliando a string como uma lista
                lista = ast.literal_eval(data2['Listas'][i])
                # Convertendo a lista para um array numpprint("A média do array é:", medy de floats
                array = np.array(lista, dtype=float)
                    
                # Concatenando os arrays
                array_unido = np.concatenate((name1, array))
                array_provavel.append(array_unido)
                array_unido = []
                print(f'Tamanho: 3.605.250 \nReal: {i / 3605250}')
                #time.sleep(10)
            
        if t2 == 185:
            array_real = array_count1
            data_array_provavel = pd.DataFrame({"Arrays": array_provavel})
            t5 = len(data_array_provavel)
            for i in range(t5):    
                print(f'Tamanho: 3.605.250 \nReal: {i}')
                name = data_array_provavel['Arrays'][i]
                
                if np.array_equal(name, array_real) == True:
                    print(name, type(name), len(name))
                #    sucessos, n = media, len(array_real)
                #    z_stat, p_value = calcular_pz(n, sucessos)

                    # Teste de incompressibilidade
                #    sequence_bytes = bytes(array_real)
                #    incompressibility_ratio = testar_incompressibilidade(sequence_bytes)
                    
                    # Teste de Martin-Löf
                #    chi_squared, martin_lof_p_value = testar_martin_lof(array_real)
                                    
                #    print(f'Quantidade de Apostas: {len(name)} \nMedia Apostas: {media} \nÚltima entrada: {name[-1]} \nEstatística Z:{z_stat} \nValor p: {p_value} \nRazão de incompressibilidade: {incompressibility_ratio} \nTeste de Martin-Löf - Chi-squared: {chi_squared} \nValor p: {martin_lof_p_value}')
                    
                #    time.sleep(600)
            array_count1 = [] 
            
            # Calculando a média do array
            
            #media = np.mean(array)
            
            #if media <= 0.62 or media >= 0.73:
            #    i += 1
            #    print("A média do array é:", media)
        

        print(i)

        print(media)

print(array_geral)

array_geral = pd.DataFrame({'Listas': array_geral})

array_geral.to_csv('/home/darkcover/Documentos/Out/dados/Parte2/matrix.csv')
array_geral.to_excel('/home/darkcover/Documentos/Out/dados/Parte2/matrix.xlsx')
