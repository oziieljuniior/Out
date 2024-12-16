import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import time


# Libs
import warnings

# Configs
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

import time 


## Carregar data
data = pd.read_csv('/home/darkcover/Documentos/Data/Out/Entrada.csv')

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j = 0,0,0

media_parray, acerto01 = [], []

# Inicializar classes
lautgh1 = np.zeros(60, dtype = int)
lautgh2 = np.zeros(60, dtype = int)

acerto, core = 0,0

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

while i <= 210000:
    print(24*'---')
    
    if len(media_parray) < 59:
        m = 0
    else:
        m = media_parray[len(media_parray) - 60]

    print(f'Número da Entrada - {i} | Acuracia_{core + 1}: {round(m,4)}')
    if i <= inteiro:
        odd = float(data['Entrada'][i].replace(",",'.'))
        #odd = float(data['Entrada'][i])
        if odd == 0:
            odd = 1
        print(f'Entrada: {odd}')
    else:
        odd = float(input("Entrada -> ").replace(",",'.'))

    if odd == 0:
        break
    
    array2s.append(odd)
    if odd >= 2:
        corte2 = 1
    else:
        corte2 = 0

    array2n.append(corte2)

    if i <= inteiro:
        print('**'*20)
        if i >= 60:
            track = i - 60
            corte1 = array2s[track]
            corte3 = array2n[track]
            
            array3s.append(corte1)
            array3n.append(corte3)
        
            print(f'Order1: {corte1} | Order2: {corte3} | LArrayI: {[len(array3n), len(array3s)]}')
        
            if (i % 60) == 0 and i > 60:
                print('*-'*16)
                print(f'LArrayII: {[len(array3n[i - 120: i - 60]), len(array3s[i - 120: i - 60])]}')
                if i == 120:
                    matrix1s = np.array(array3s[i - 120: i - 60]).reshape(-1,1)
                    matrix1n = np.array(array3n[i - 120: i - 60]).reshape(-1,1)
            
                else:
                    array3ss = np.array(array3s[i - 120: i - 60]).reshape(-1,1)
                    array3ns = np.array(array3n[i - 120: i - 60]).reshape(-1,1)

                    matrix1s = np.hstack([matrix1s,array3ss])
                    matrix1n = np.hstack([matrix1n,array3ns]) 
                    
                #print(matrix1n)
                print(f'Order3: {i} | LArrayIII: {[len(array3n), len(array3s)]} | MatrixS: {[matrix1n.shape, matrix1s.shape]}')
                print('*-'*16)
        print('**'*20)
        
        if i >= 301:
            print(24*"-'-")
            for name in y_pred:
                print(f'Predição Modelo R.: {name}')
                if name == 1:
                    if odd >= 2:
                        count = 1
                        if count == name:
                            acerto = acerto + 1
                            j = j + 1
                    else:
                        j = j + 1
            if j == 0:
                acuracia = 0
            else:
                acuracia = (acerto / j) * 100

            if np.sum(y_pred[0]) == 1:
                if core == 59:
                    lautgh1[0] = lautgh1[0] + 1
                    lautgh2[0] = lautgh2[0] + 1
                else:
                    lautgh1[core + 1] = lautgh1[core + 1] + 1
                    lautgh2[core + 1] = lautgh2[core + 1] + 1
            else:
                if core == 59:
                    lautgh1[0] = lautgh1[0] + 1
                else:
                    lautgh1[core + 1] = lautgh1[core + 1] + 1
            if core == 59:
                medida_pontual = lautgh2[0] / lautgh1[0]
            else:    
                medida_pontual = lautgh2[core + 1] / lautgh1[core + 1]

            media_parray.append(medida_pontual)
            print(f'Acuracia modelo Geral: {round(acuracia,4)} | Acuracia_{core + 1}: {round(medida_pontual,4)}')
            print(24*"-'-")

        if i >= 300 and i % 60 == 0:
            print('/-/'*16)
            print(f'Treinamento necessario ...')
            print(f'MatrixS: {[matrix1n.shape, matrix1s.shape]} | Indice: {matrix1n.shape[1]}')
            m,n = matrix1n.shape
            modelos = []
            for j1 in range(0,m):
                X, y = [],[]
                X = np.array(matrix1n[j1,:-1]).reshape(1,-1).astype("float32")
                y = np.array(matrix1n[j1,-1]).reshape(1,-1).astype("int64")
                # 3. Treinar o modelo de regressão logística
                modelo = DecisionTreeClassifier(random_state=101)
                modelo.fit(X, y)  # Treinamento individual para cada linha
                modelos.append(modelo)
            print(f'Treinamento Realizado com Sucesso ...')
            print('/-/'*16)

        if i >= 300:
            core = i % 60
            if core == 59:
                x_pred = np.array(matrix1n[0,1:matrix1n.shape[1]]).reshape(1,-1).astype("float32")
                y_pred = modelos[0].predict(x_pred)
            else:
                x_pred = np.array(matrix1n[core+1,1:matrix1n.shape[1]]).reshape(1,-1).astype("float32")
                y_pred = modelos[core + 1].predict(x_pred)
            
            
            #print(x_pred, y_pred)
    else:
        print('**'*20)
        if i >= 60:
            track = i - 60
            corte1 = array2s[track]
            corte3 = array2n[track]
            
            array3s.append(corte1)
            array3n.append(corte3)
        
            print(f'Order1: {corte1} | Order2: {corte3} | LArrayI: {[len(array3n), len(array3s)]}')
        
            if (i % 60) == 0 and i > 60:
                print('*-'*16)
                print(f'LArrayII: {[len(array3n[i - 120: i - 60]), len(array3s[i - 120: i - 60])]}')
                if i == 120:
                    matrix1s = np.array(array3s[i - 120: i - 60]).reshape(-1,1)
                    matrix1n = np.array(array3n[i - 120: i - 60]).reshape(-1,1)
            
                else:
                    array3ss = np.array(array3s[i - 120: i - 60]).reshape(-1,1)
                    array3ns = np.array(array3n[i - 120: i - 60]).reshape(-1,1)

                    matrix1s = np.hstack([matrix1s,array3ss])
                    matrix1n = np.hstack([matrix1n,array3ns]) 
                    
                #print(matrix1n)
                print(f'Order3: {i} | LArrayIII: {[len(array3n), len(array3s)]} | MatrixS: {[matrix1n.shape, matrix1s.shape]}')
                print('*-'*20)
        
        if i >= 301:
            print(24*"-'-")
            for name in y_pred:
                if name == 1:
                    if odd >= 2:
                        count = 1
                        if count == name:
                            acerto = acerto + 1
                            j = j + 1
                    else:
                        j = j + 1
            if j == 0:
                acuracia = 0
            else:
                acuracia = (acerto / j) * 100

            if np.sum(y_pred) == 1:
                if core == 59:
                    lautgh1[0] = lautgh1[0] + 1
                    lautgh2[0] = lautgh2[0] + 1
                else:
                    lautgh1[core + 1] = lautgh1[core + 1] + 1
                    lautgh2[core + 1] = lautgh2[core + 1] + 1
            else:
                if core == 59:
                    lautgh1[0] = lautgh1[0] + 1
                else:
                    lautgh1[core + 1] = lautgh1[core + 1] + 1
            if core == 59:
                medida_pontual = lautgh2[0] / lautgh1[0]
            else:    
                medida_pontual = lautgh2[core + 1] / lautgh1[core + 1]

            media_parray.append(medida_pontual)
            print(f'Acuracia modelo Geral: {round(acuracia,4)} | Acuracia_{core + 1}: {round(medida_pontual,4)}')
            print(24*"-'-")

        if i >= 300 and i % 60 == 0:
            print('/-/'*16)
            print(f'Treinamento necessario ...')
            print(f'MatrixS: {[matrix1n.shape, matrix1s.shape]} | Indice: {matrix1n.shape[1]}')
            m,n = matrix1n.shape
            modelos = []
            for j1 in range(0,m):
                X, y = [],[]
                X = np.array(matrix1n[j1,:-1]).reshape(1,-1).astype("float32")
                y = np.array(matrix1n[j1,-1]).reshape(1,-1).astype("int64")
                # 3. Treinar o modelo de regressão logística
                modelo = DecisionTreeClassifier(random_state=101)
                modelo.fit(X, y)  # Treinamento individual para cada linha
                modelos.append(modelo)
            print(f'Treinamento Realizado com Sucesso ...')
            print('/-/'*16)
            

        if i >= 300:
            core = i % 60
            if core == 59:
                x_pred = np.array(matrix1n[0,1:matrix1n.shape[1]]).reshape(1,-1).astype("float32")
                y_pred = modelos[0].predict(x_pred)
                print(f'Proxima Entrada:{y_pred}')
            else:
                x_pred = np.array(matrix1n[core+1,1:matrix1n.shape[1]]).reshape(1,-1).astype("float32")
                y_pred = modelos[core + 1].predict(x_pred)
                print(f'Proxima Entrada:{y_pred}')

    
    i += 1            
            
