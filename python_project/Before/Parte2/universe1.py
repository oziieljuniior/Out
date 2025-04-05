import itertools
import numpy as np
import pandas as pd

# Definindo o tamanho da sequência e o valor da média desejada
n = 25
target_mean = 0.67
error_margin = 0.05

# Gerando e verificando as combinações uma a uma
valid_sequences = []

# Função para calcular a média de uma sequência
def calculate_mean(sequence):
    return np.mean(sequence)

# Iterando sobre todas as combinações possíveis de sequências binárias
for seq in itertools.product([0, 1], repeat=n):
    print(seq)
    mean = calculate_mean(seq)
    valid_sequences.append(seq)
    
    # Se quiser limitar o número de sequências válidas, pode adicionar um break aqui
    # e definir um limite máximo.

    # Exemplo: limitando a 1000 sequências válidas
    # if len(valid_sequences) >= 1000:
    #     break

# Exemplo de impressão das primeiras 10 sequências válidas
for seq in valid_sequences[:10]:
    print(seq)


data = pd.DataFrame({'Listas': valid_sequences})

data.to_csv('/home/darkcover/Documentos/Out/dados/Parte2/universe2.csv')
#data.to_excel('/home/darkcover/Documentos/Out/dados/Parte2/universe1.xlsx')