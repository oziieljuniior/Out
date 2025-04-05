import numpy as np



## Funções

def matriz(i0, array1, array2):
    lista = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
    info = []
    final1, final2 = [],[]
    for name in lista:
        order = i0 % name
        if order == 0:
            info.append(name)
        
    print(f'Novas colunas para: {info} ...')
    
    for name in info:
        m0, m1 = len(array1), len(array2)
        order1 = m1 // name
        
        print(name, order1, m0, m1)
        
        if order1 >= 5:
            matriz1 = np.array(array1).reshape(-1, name).T
            matriz2 = np.array(array2).reshape(-1, name).T
            print(f'Order3: {i} | MatrixS: {[matriz1.shape, matriz2.shape]}')
            final1.append(matriz1), final2.append(matriz2)
    return final1, final2

array1, array2 = [], []
i = 0
while i <= 1200:
    print(4*'*-')
    print(f'i: {i}')
    if i % 60 == 0:
        matriz11, matriz12 = matriz(i, array1, array2)
    
    
    array1.append(i), array2.append(i)
    i += 1
print(np.array(matriz11[0]).shape)