import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import time


# Libs
import warnings

# Configs
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

import time 


## Carregar data
data = pd.read_csv('/home/darkcover1/Documentos/Data/Out/odds_200k.csv')

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j = 0,0,0

media_parray = []

# Inicializar classes
lautgh1 = np.zeros(240, dtype = int)
lautgh2 = np.zeros(240, dtype = int)

acerto, core = 0,0

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

while i <= 210000:
    print(24*'---')
    print(len(media_parray))
    if len(media_parray) < 239:
        m = 0
    else:
        m = media_parray[len(media_parray) - 240]

    print(f'Número da Entrada - {i} | Acuracia_{core + 1}: {round(m,4)}')
    if i <= inteiro:
        odd = float(data['Odd'][i])
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
        if i >= 240:
            track = i - 240
            corte1 = array2s[track]
            corte3 = array2n[track]
            
            array3s.append(corte1)
            array3n.append(corte3)
        
            print(f'Order1: {corte1} | Order2: {corte3} | LArrayI: {[len(array3n), len(array3s)]}')
        
            if (i % 240) == 0 and i > 240:
                print('*-'*16)
                print(f'LArrayII: {[len(array3n[i - 480: i - 240]), len(array3s[i - 480: i - 240])]}')
                if i == 480:
                    matrix1s = np.array(array3s[i - 480: i - 240]).reshape(-1,1)
                    matrix1n = np.array(array3n[i - 480: i - 240]).reshape(-1,1)
            
                else:
                    array3ss = np.array(array3s[i - 480: i - 240]).reshape(-1,1)
                    array3ns = np.array(array3n[i - 480: i - 240]).reshape(-1,1)

                    matrix1s = np.hstack([matrix1s,array3ss])
                    matrix1n = np.hstack([matrix1n,array3ns]) 
                    
                #print(matrix1n)
                print(f'Order3: {i} | LArrayIII: {[len(array3n), len(array3s)]} | MatrixS: {[matrix1n.shape, matrix1s.shape]}')
                print('*-'*16)
        print('**'*20)
        
        if i >= 961:
            print(24*"-'-")
            for name in y_pred:
                if name[0] == 1:
                    if odd >= 2:
                        count = 1
                        if count == name[0]:
                            acerto = acerto + 1
                            j = j + 1
                    else:
                        j = j + 1
            if j == 0:
                acuracia = 0
            else:
                acuracia = (acerto / j) * 100

            if np.sum(y_pred[0]) == 1:
                if odd >= 2:
                    lautgh1[core] = lautgh1[core] + 1
                    lautgh2[core] = lautgh2[core] + 1
                    medida_pontual = lautgh2[core] / lautgh1[core]
                else:
                    lautgh1[core] = lautgh1[core] + 1
                    lautgh2[core] = lautgh2[core]
                    medida_pontual = lautgh2[core] / lautgh1[core]
            else:
                if len(media_parray) < 239:
                    medida_pontual = 0
                else:
                    medida_pontual = media_parray[len(media_parray) - 240]

            media_parray.append(medida_pontual)
            print(f'Acuracia modelo Geral: {round(acuracia,4)} | Acuracia_{core + 1}: {round(medida_pontual,4)}')
            print(24*"-'-")

        if i >= 960 and i % 240 == 0:
            print('/-/'*16)
            print(f'Treinamento necessario ...')
            print(f'MatrixS: {[matrix1n.shape, matrix1s.shape]} | Indice: {matrix1n.shape[1]}')
            m,n = matrix1n.shape
            modelos = []
            for j1 in range(0,m):
                X, y = [],[]
                X = np.array(matrix1n[j1,0:n-2]).reshape(1,-1).astype("float32")
                y = np.array(matrix1n[j1,n-1]).reshape(1,-1).astype("int64")
                modelo = LinearRegression()
                #print(X.shape, y.shape)
                modelo.fit(X,y)
                modelos.append(modelo)
            print(f'Treinamento Realizado com Sucesso ...')
            print('/-/'*16)

        if i >= 960:
            core = i % 240
            if core == 239:
                x_pred = np.array(matrix1n[0,1:(n-1)]).reshape(1,-1).astype("float32")
                y_pred = modelos[0].predict(x_pred)
                print(f'Predição Modelo R.: {y_pred[0]}')
            else:
                x_pred = np.array(matrix1n[core+1,1:(n-1)]).reshape(1,-1).astype("float32")
                y_pred = modelos[core+1].predict(x_pred)
                print(f'Predição Modelo R.: {y_pred[0]}')
            
            
            #print(x_pred, y_pred)
    else:
        print('**'*20)
        if i >= 240:
            track = i - 240
            corte1 = array2s[track]
            corte3 = array2n[track]
            
            array3s.append(corte1)
            array3n.append(corte3)
        
            print(f'Order1: {corte1} | Order2: {corte3} | LArrayI: {[len(array3n), len(array3s)]}')
        
            if (i % 240) == 0 and i > 240:
                print('*-'*16)
                print(f'LArrayII: {[len(array3n[i - 480: i - 240]), len(array3s[i - 480: i - 240])]}')
                if i == 480:
                    matrix1s = np.array(array3s[i - 480: i - 240]).reshape(-1,1)
                    matrix1n = np.array(array3n[i - 480: i - 240]).reshape(-1,1)
            
                else:
                    array3ss = np.array(array3s[i - 480: i - 240]).reshape(-1,1)
                    array3ns = np.array(array3n[i - 480: i - 240]).reshape(-1,1)

                    matrix1s = np.hstack([matrix1s,array3ss])
                    matrix1n = np.hstack([matrix1n,array3ns]) 
                    
                #print(matrix1n)
                print(f'Order3: {i} | LArrayIII: {[len(array3n), len(array3s)]} | MatrixS: {[matrix1n.shape, matrix1s.shape]}')
                print('*-'*20)
        
        if i >= 961:
            print(24*"-'-")
            for name in y_pred:
                if name[0] == 1:
                    if odd >= 2:
                        count = 1
                        if count == name[0]:
                            acerto = acerto + 1
                            j = j + 1
                    else:
                        j = j + 1
            if j == 0:
                acuracia = 0
            else:
                acuracia = (acerto / j) * 100

            if np.sum(y_pred[0]) == 1:
                if odd >= 2:
                    lautgh1[core] = lautgh1[core] + 1
                    lautgh2[core] = lautgh2[core] + 1
                    medida_pontual = lautgh2[core] / lautgh1[core]
                else:
                    lautgh1[core] = lautgh1[core] + 1
                    lautgh2[core] = lautgh2[core]
                    medida_pontual = lautgh2[core] / lautgh1[core]
            else:
                if len(media_parray) < 239:
                    medida_pontual = 0
                else:
                    medida_pontual = media_parray[len(media_parray) - 240]

            media_parray.append(medida_pontual)
            print(f'Acuracia modelo Geral: {round(acuracia,4)} | Acuracia_{core + 1}: {round(medida_pontual,4)}')
            print(24*"-'-")

        if i >= 960 and i % 240 == 0:
            print('/-/'*16)
            print(f'Treinamento necessario ...')
            print(f'MatrixS: {[matrix1n.shape, matrix1s.shape]} | Indice: {matrix1n.shape[1]}')
            m,n = matrix1n.shape
            modelos = []
            for j1 in range(0,m):
                X, y = [],[]
                X = np.array(matrix1n[j1,0:n-2]).reshape(1,-1).astype("float32")
                y = np.array(matrix1n[j1,n-1]).reshape(1,-1).astype("int64")
                modelo = LinearRegression()
                #print(X.shape, y.shape)
                modelo.fit(X,y)
                modelos.append(modelo)
            print(f'Treinamento Realizado com Sucesso ...')
            print('/-/'*16)
            

        if i >= 960:
            core = i % 240
            if core == 239:
                x_pred = np.array(matrix1n[0,1:(n-1)]).reshape(1,-1).astype("float32")
                y_pred = modelos[0].predict(x_pred)
                print(f'Proxima Entrada:{y_pred[0]}')
            else:
                x_pred = np.array(matrix1n[core+1,1:(n-1)]).reshape(1,-1).astype("float32")
                y_pred = modelos[core+1].predict(x_pred)
                print(f'Proxima Entrada:{y_pred[0]}')

    
    i += 1            
            
