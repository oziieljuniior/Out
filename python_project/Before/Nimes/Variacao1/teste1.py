import pandas as pd
import numpy as np
import time

data = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k_1.csv')

array1 = []

print(data.columns)

for i in range(0, len(data)):
    if data['Odd'][i] >= 2:
        array1.append(1)
    else:
        array1.append(0)
array2 = []
array3 = []

for j in range(60, len(array1)):
    order = array1[j - 60: j]
    media = sum(order)/60
    array2.append(media)

for k in range(0, len(array2), 60):
    print(array2[k])
    print(k)
    
    desvpad = np.des
    soma = (array2[k] * array2[k + 60] * array2[k + 0])
    media = soma**(1/3)
    print(48*'*-')
    print('media', media)
    print(48*'*-')
    time.sleep(2)