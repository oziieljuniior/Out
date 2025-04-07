import sys
sys.path.append("/home/ozielramos/Documentos/Out/python_project/Atual/cake/Oz/Modulos")
from Arquivos  import FileSelector# Importa a classe FileSelector do módulo Arquivos # type: ignore
import MathMo # type: ignore
import Odds # type: ignore
import Predicao # type: ignore
import Placar # type: ignore
import Matrizes # type: ignore

import pandas as pd
import numpy as np

## Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
selector1 = FileSelector()
file_path1 = selector1.open_file_dialog()

data = pd.read_csv(file_path1)

array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

a1, i, j2, j3 = 0,0,0,0

media_parray, acerto01 = [], []

acerto2, acerto3, core1 = 0,0,0

modelos = [None]*5000
data_matrizes = [None]*5000
recurso1, recurso2 = [None]*5000, [None]*5000

array_geral = np.zeros(6, dtype=float)
df1 = pd.DataFrame({'lautgh1': np.zeros(60, dtype = int), 'lautgh2': np.zeros(60, dtype = int)})

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

data_matriz_float = []
data_matriz_int = []
data_array_float = []
data_array_int = []
array_geral_float = []
data_acuracia_geral = []
data_precisao_geral = []
pred_acumulada = []

processor1 = Odds.FuzzyOddsProcessor()
processor2 = Predicao.ModelPredictionHandler()
processor3 = Placar.ScoreManager()
processor4 = Matrizes.MatrixTransformer()

while i <= 210000:
    print(24*'---')
    #print(len(media_parray))
    if len(media_parray) < 59:
        m = 0
        core1 = 0
    else:
        m = media_parray[len(media_parray) - 60]

    print(f'Número da Entrada - {i} | Acuracia_{core1 + 1}: {round(m,4)}')
    
    array2s, array2n, odd = processor1.coletar_odd(i, inteiro, data, array2s, array2n)
    array_geral_float.append(float)
    if odd == 0:
        break

    if i >= 361:
        print(24*"-'-")
        
        array_geral = processor3.placargeral(resultado, odd, array_geral)
        media_parray = processor3.placar60(df1, i, media_parray, resultado, odd)
        
        if i % 60 == 0:
            core11 = 60
        else:
            core11 = core1
        print(f'Acuracia modelo Geral: {round(array_geral[0],4)} | Acuracia_{core11}: {round(media_parray[-1],4)} \nPrecisao modelo Geral: {round(array_geral[1],4)}')
        data_acuracia_geral.append(array_geral[0]), data_precisao_geral.append(array_geral[1])
        print(24*"-'-")

    if i >= 360 and (i % 60) == 0:
        print('***'*20)
        print(f'Carregando dados ...')
        info = []
        cronor = (i + 300) // 5
        lista = [name for name in range(60, cronor, 60)]
        tier = True

        for click in lista:
            k0 = i % click
            k1 = (i - 60) % click
            if k0 == 0 and k1 == 0:
                info.append(click)
        print(f'{12*"*-"} \nPosições que devem ser carregadas: {info} \n{12*"*-"}')

        for click in info:
            print(f'Treinamento para {click}')
            matriz_final_float, matriz_final_int, posicao0 = processor4.tranforsmar_final_matriz(click, array2s, array2n)
            print(f'Matrix_{click}: {[matriz_final_float.shape, matriz_final_int.shape]} | Posicao: {posicao0}')
            data_matriz_float.append(matriz_final_float), data_matriz_int.append(matriz_final_int)
            n = matriz_final_float.shape[1]
            array1, array2 = matriz_final_float[:,:(n - 3)], matriz_final_int[:,-1]
            #chamada2
            mk = MathMo.RedeNeuralXGBoost(learning_rate=0.05, n_estimators=500)
            models = mk.treinar(array1, array2)
            mk.avaliar()
            modelos[posicao0] = models
            data_matrizes[posicao0] = matriz_final_float
            print(f'Treinamento {click} realizado com sucesso ...  \n')
        print('***'*20)

    if i >= 360:
        core2 = i % 60
        y_pred1 = processor2.lista_predicao(i,len(modelos), modelos, data_matrizes)
        resultado = processor2.ponderar_lista(y_pred1)
        pred_acumulada.append(resultado)
        print(24*'*-')
        print(f'Proxima Entrada: {resultado} | Geral: {sum(pred_acumulada)}')
        print(24*'*-')


    i += 1

print(f'Ordenandos os dados ...')

data_array_float = np.array(array2s)
data_array_int = np.array(array2n)

file_path2 = FileSelector().open_file_dialog()
file_path3 = FileSelector().open_file_dialog()

order1 = pd.DataFrame({'Acuracia': data_acuracia_geral, 'Precisao': data_precisao_geral})
order1.to_csv(file_path2, index=False)

data_final = pd.DataFrame({'Entrada': data_array_float, 'Resultado': data_array_int})
data_final.to_csv(file_path3, index=False)