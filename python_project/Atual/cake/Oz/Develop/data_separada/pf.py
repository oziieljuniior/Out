import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

import skfuzzy as fuzz

# Libs
import time
import warnings

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Configs
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

## Funções
def calculate_means(array4):
    """
    Calcula a média dos elementos de array4 em janelas deslizantes de 59 elementos.

    Args:
        array4 (list): Lista de inteiros (0 ou 1).

    Returns:
        list: Lista com a média dos elementos em janelas de 59 elementos.
    """
    array6 = []
    array7 = []
    for i in range(len(array4) - 1):
        array6.append(array4[i])
        if i >= 59:
            order = float(np.mean(array6))
            array7.append(order)
    
    return array7

def calculate_orders(array4):
        """
        Calcula a soma dos elementos de array4 em janelas deslizantes de 59 elementos.

        Args:
            array4 (list): Lista de inteiros (0 ou 1).

        Returns:
            list: Lista com a soma dos elementos em janelas de 59 elementos.
        """
        # Calcular a soma dos elementos em janelas de 59 elementos
        array5 = []
        for i in range(len(array4) - 1):
            if i >= 59:
                order = sum(array4[i - 59: i])
                array5.append(order)
        return array5

def placar60(df1, i, media_parray, resultado, odd):
    """
    Função que organizar o placar das 60 entradas.
    Args:
        df1 (pd.DataFrame): DataFrame responsavel por armazenar as somas de cada entrada.
        i (int): Valor inteiro não-negativo crescente. Responsável por controlar a posição dos dados.
        media_parray (array): Array responsavel por armazenar a acuracia de cada entrada.
        resultado (int): Valor interio(0 ou 1) gerado pela função ponderar lista com a entrada anterior.
        odd (float): Valor real com duas casas decimais. Ele é determinado pela entrada dos dados, ou usuário.
    Returns:
        array: Array contendo as variaveis df1 e mediap_array atualizadas, também retorna a acuracia atualizada.
    """
    core1 = i % 60
    if resultado == 1:
        if odd >= 3:
            df1.iloc[core1,:] += 1
            medida_pontual = df1.iloc[core1, 0] / df1.iloc[core1, 1]
        else:
            df1.iloc[core1,1] += 1
            medida_pontual = df1.iloc[core1, 0] / df1.iloc[core1, 1]
    else:
        if len(media_parray)<59:
            medida_pontual = 0
        else:
            medida_pontual = media_parray[len(media_parray) - 60]

    media_parray.append(medida_pontual)

    return media_parray

def fuzzy_classification(odd):
    """
    Implementação da lógica fuzzy para classificar as odds no intervalo de 1 a 6.
    """
    odd_range = np.arange(1, 6.1, 0.1)
    
    # Conjuntos fuzzy ajustados para cobrir todo o intervalo de 1 a 6
    baixo = fuzz.trimf(odd_range, [1, 1, 2])
    medio = fuzz.trimf(odd_range, [1.5, 3, 4.5])
    alto = fuzz.trimf(odd_range, [3.5, 5, 6])
    muito_alto = fuzz.trimf(odd_range, [4.5, 6, 6])
    
    # Graus de pertinência
    pert_baixo = fuzz.interp_membership(odd_range, baixo, odd)
    pert_medio = fuzz.interp_membership(odd_range, medio, odd)
    pert_alto = fuzz.interp_membership(odd_range, alto, odd)
    pert_muito_alto = fuzz.interp_membership(odd_range, muito_alto, odd)
    
    # Classificação baseada nos graus de pertinência
    max_pert = max(pert_baixo, pert_medio, pert_alto, pert_muito_alto)
    
    if max_pert == 0:
        return 0  # Nenhuma confiança
    
    if max_pert == pert_muito_alto:
        return 1  # Alta confiança na subida
    elif max_pert == pert_alto:
        return 0.75  # Confiança moderada-alta
    elif max_pert == pert_medio:
        return 0.5  # Confiança média
    else:
        return 0.25  # Baixa confiança

def coletarodd(i, j, data, array2s, array2n):
    """
    Função que coleta e organiza as entradas iniciais do banco de dados.
    Args:
        i (int): Valor inteiro não-negativo. Entrada que controla o loop principal. É um valor cumulativo.
        j (int): Valor inteiro não-negativo. Entrada que determina em qual colunas os dados estão.
        data (pd.DataFrame): Variável carregada inicialmente para treinamento/desenvolvimento. Do tipo data frame.   #FIXWARNING2
        array2s (np.array): Array cumulativo que carrega as entradas reais com duas casas decimais.
        array2n (np.array): Array cumulativo que carrega as entredas inteiras(0 ou 1).
        alanvanca (bool): Variável booleana que determina se a entrada é automática ou manual.   #FIXWARNING1
    Returns:
        np.array: Array cumulativo que carrega as entradas reais com duas casas decimais.
        np.array: Array cumulativo que carrega as entredas inteiras(0 ou 1).
        float: Valor real com duas casas decimais. Ele é determinado pela entrada dos dados, ou usuário.
    """

#FIXWARNING1: O formato da data de entrada pode ser mudado? Atualmente está em .csv

    odd = data[i,j]

    if odd == 0:
        odd = 1

    if odd >= 6:
        odd = 6
    
    corte1 = fuzzy_classification(odd)  # Aplicando lógica fuzzy
    array2s.append(corte1)
    if odd >= 3:
        corte2 = 1
    else:
        corte2 = 0    
    array2n.append(corte2)

    return array2s, array2n, odd

def matriz(num_linhas, array):
    """
    Transforma um array unidimensional em uma matriz organizada por colunas.
    
    Args:
        array (list ou np.array): Lista de números a serem organizados.
        num_linhas (int): Número de linhas desejadas na matriz.

    Returns:
        np.array: Matriz ordenada.
    """
    t = len(array)
    if t % num_linhas != 0:
        raise ValueError("O tamanho do array deve ser múltiplo do número de linhas.")
    
    # Reshape para matriz (por linha) e depois transpõe para organizar por colunas
    matriz = np.array(array).reshape(-1, num_linhas).T
    
    return matriz 

def placargeral(resultado, odd, array_geral):
    """
    Função que realiza o gerenciamento da precisão e acurácia gerais.
    Args:
        resultado (int): Valor interio(0 ou 1) gerado pela função ponderar lista com a entrada anterior.
        odd (float): Valor real com duas casas decimais. Ele é determinado pela entrada dos dados, ou usuário.
        array_geral (np.array): O array contém algumas informações essenciais para o prosseguimento das entradas. -->["acuracia" (float): Valor real não-negativo. Responsável por acompanhar acerto 0,"precisao" (float): Valor real não-negativo. Responsavel por acompanhar acertos 0 e 1, "acerto"(int): Valor inteiro não-negativo. Essa variável é cumulativa com valor inicial igual a zero, ela é determinada a partir da entrada 301. Além disso, ela é responsável pela contagem de acertos da acurácia.   #FIXWARNING1, "acerto1" (int): Valor inteiro não-negativo. Essa variável é cumulativa com valor inicial igual a zero, ela é determinada a partir da entrada 301. . Além disso, ela é responsável pela contagem de acertos da precisão.   #FIXWARNING1, "j" (int): Valor inteiro não-negativo. Essa variável é cumulativa com valor inicial igual a zero, ela é determinada a partir da entrada 301. . Além disso, ela é responsável pela contagem geral da acurácia.   #FIXWARNING1, "j1" (int): j (int): Valor inteiro não-negativo. Essa variável é cumulativa com valor inicial igual a zero, ela é determinada a partir da entrada 301. . Além disso, ela é responsável pela contagem geral da precisão.   #FIXWARNING1 #FIXWARNING3
    Returns:
        np.array: Array contendo acurácia e precisão. Alem de conter acerto, acerto1, j e j1.
    """

#FIXWARNING1: Essa entrada é fixa ?
#FIXWARNING3: Ajustar essa documentação.
#array_geral = [acuracia, precisao, acerto, acerto1, j, j1]

    name = resultado

    if name == 1:
        if odd >= 3:
            count = 1
            if count == name:
                array_geral[2] += 1
                array_geral[3] += 1
                array_geral[4] += 1
                array_geral[5] += 1
        else:
            array_geral[4] += 1
            array_geral[5] += 1
    else:
        if odd < 3:
            count = 0
            if count == name:
                array_geral[3] += 1
                array_geral[5] += 1
        else:
            array_geral[5] += 1

    if array_geral[4] != 0 and array_geral[5] != 0:
        array_geral[0] = (array_geral[2] / array_geral[4]) * 100
        array_geral[1] = (array_geral[3] / array_geral[5]) * 100
    else:
        array_geral[0], array_geral[1] = 0,0

    return array_geral

def lista_predicao(i, t, modelos, array1):
    """
    Gera uma lista com possíveis predições.
    Args:
        t (int): Quantidade de modelos contidos na lista original.
        modelos (np.array): Array que contém modelos treinados.
        array1 (np.array): Lista contendo os últimas matrizes.
    Returns:
        np.array: Array que contém a predição de cada modelo da lista original.
    """
    y_pred1 = []
    for sk in range(0,t):
        if modelos[sk] is not None:
            posicao = 60*sk + 60
            print(sk, posicao)
            matriz1s = array1[sk]
            trick2 = i % 60
            if trick2 == 59:
                order1 = 0
            else:
                order1 = trick2 + 1

            x_new = np.array(matriz1s[order1,3:])
            x_new = x_new.astype("float32")
            x_new = x_new.reshape(1, -1)
            print(x_new.shape)
            predictions = modelos[sk].predict(x_new)

            y_pred = np.argmax(predictions) if predictions.ndim > 1 else predictions
            y_pred1.append(y_pred[0])
    print(y_pred1)
    return y_pred1

def salvar_resumo_ar(text, nome_arquivo="resumo_modelo_ar.txt"):    
    # Obter a data e hora atual
    data_atual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Abrir o arquivo no modo append ("a"), para adicionar conteúdo ao final
    with open(nome_arquivo, "a") as arquivo:
        # Adicionar separador para identificar cada execução
        arquivo.write("\n" + "="*40 + "\n")
        
        # Escrever a data no início do arquivo
        arquivo.write(f"Data das entradas: {data_atual}\n\n")
        
        # Escrever o resumo do modelo no arquivo
        arquivo.write(str(text))
        arquivo.write("\n" + "="*40 + "\n")  # Adicionar separador no final

    print(f"Resumo do modelo AR adicionado ao arquivo {nome_arquivo}")

def reden(i, j,array1, array2):
    """
    Função para treinar uma rede neural usando as entradas e saídas fornecidas.

    Args:
        array1 (numpy.array): Saídas (rótulos) binárias (0 ou 1).
        array3 (numpy.array): Entradas preditoras.
        m (int): Número de amostras.
        n (int): Número de características por amostra.

    Returns:
        keras.Model: Modelo treinado.
    """
    X = array1
    y = array2
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=2,
    eval_metric='mlogloss',
    learning_rate=0.005,  # Reduzir para aprendizado mais estável
    n_estimators=500,  # Mais árvores com taxa de aprendizado menor
    max_depth=4,  # Evitar overfitting
    subsample=0.7,  # Menos amostras por árvore para generalizar melhor
    colsample_bytree=0.7,  # Seleciona menos features por árvore
    gamma=0.2,  # Penaliza divisões irrelevantes
    min_child_weight=5,  # Evita overfitting
    reg_alpha=0.1,  # Regularização L1
    reg_lambda=1.0,  # Regularização L2
    random_state=42 
    )
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Acurácia do modelo: {accuracy:.4f}')
    print(f'F1-Score do modelo: {f1:.4f}')

    xgb.plot_importance(model)
    #plt.show()
    name1 = '/home/darkcover/Documentos/Out/dados/DoIt/imagens/modelo' + str(i) + '_' + str(j) + '.png'
    plt.savefig(name1)  # Salva no formato PNG

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    #plt.show()
    name2 = '/home/darkcover/Documentos/Out/dados/DoIt/imagens/cmatrix' + str(i) + '_' + str(j) + '.png'
    plt.savefig(name2)  # Salva no formato PNG

    print(classification_report(y_test, y_pred))
    salvar_resumo_ar(classification_report(y_test, y_pred))
    
    # Calcular as probabilidades das classes
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calcular a curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plotar a curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    #plt.show()
    name3 = '/home/darkcover/Documentos/Out/dados/DoIt/imagens/roc' + str(i) + '_' + str(j) + '.png'
    plt.savefig(name3)  # Salva no formato PNG
    
    return model

def ponderar_lista(lista, base=1.10):
    """
    Realiza uma ponderação dos elementos da lista com pesos exponenciais crescentes.

    Args:
        lista (list): Lista de inteiros contendo apenas 0 e 1.
        base (float): Base da função exponencial. Deve ser maior que 1.

    Returns:
        int: Resultado ponderado, 0 ou 1.
    """
    n = len(lista)
    if n == 0:
        raise ValueError("A lista não pode estar vazia.")

    # Calcular pesos exponenciais
    pesos = [base ** i for i in range(n)]

    # Calcular soma ponderada e total de pesos
    soma_ponderada = sum(elemento * peso for elemento, peso in zip(lista, pesos))
    total_pesos = sum(pesos)
    
    qrange = 1 / n

    # Retornar 1 se média ponderada >= 0.5, senão 0
    return 1 if soma_ponderada / total_pesos >= qrange else 0

def tranforsmar_final_matriz(click, array1s, array1n):
    """
        Reponsavel por carregar matriz final. Idealmente elaborado
        para comportar outras variáveis de entrada.
        Args:
            click (int): Valor inteiro não-negativo. Entrada 
                que controla o loop principal. É um valor cumulativo.
            array1s (np.array): Array com entradas vetorizadas float.
            array1n (np.array): Array com entradas vetorizadas int.
        Returns:
            np.array: Matriz final.
    """
    n1 = len(array1n) - 61
    print(n1)
    if n1 % click != 0:
        while n1 != 0:
            print(len(array1n) - 61, click)
            # Gerar um array de 60 elementos com distribuição 30% de 1 e 70% de 0
            novo_array = np.random.choice([1, 0], size=60, p=[0.3, 0.7])
            # Concatenar os arrays
            array1n = np.concatenate((novo_array, array1n))

            if (len(array1n) - 61) % click == 0:
                n1 == 1
    arrayacertos60 = calculate_orders(array1n)
    matrizacertos60 = matriz(click, arrayacertos60[1:])
    arraymediamovel = calculate_means(array1n)
    matrizmediamovel = matriz(click, arraymediamovel[1:])
    print(len(array1s[1:]), len(array1n[1:]))
    matrix1s, matrix1n = matriz(click, array1s[1:]), matriz(click, array1n[1:])
    matrix1s, matrix1n = matrix1s[:,1:], matrix1n[:,1:]

    print(matrix1n.shape, matrix1s.shape, matrizacertos60.shape, matrizmediamovel.shape)
    posicao0 = int((click // 60) - 1)

    # Empilhar as matrizes para ter um eixo extra (60, 8, 3)
    X_stack = np.stack([matrix1s, matrizacertos60, matrizmediamovel], axis=2)  # Formato (60, 8, 3)
    # Reorganizar para intercalar coluna por coluna
    matrix1s = X_stack.reshape(60, -1)  # Agora está no formato (60, 24)
    print(matrix1s.shape, matrix1n.shape)  # Saída: (60, 24)

    return matrix1s, matrix1n, posicao0

def particionar_dados(array, num_entradas_por_coluna=1000):
    """
    Particiona um array unidimensional em uma matriz onde cada coluna contém num_entradas_por_coluna entradas.

    Args:
    array (list ou np.array): Lista de números a serem organizados.
    num_entradas_por_coluna (int): Número de entradas desejadas por coluna.

    Returns:
    np.array: Matriz ordenada.
    """
    t = len(array)
    if t % num_entradas_por_coluna != 0:
        raise ValueError("O tamanho do array deve ser múltiplo do número de entradas por coluna.")

    # Reshape para matriz (por coluna)
    matriz = np.array(array).reshape(-1, num_entradas_por_coluna).T

    return matriz


## Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
data1 = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
data1 = data1.iloc[:-999].reset_index(drop=True)
print(data1.shape)
data = particionar_dados(data1['Odd'],1000)
print(data.shape)
for j in range(data.shape[1]):
    print(j)
    array1, array2s, array2n, array3n, array3s, matrix1s, matrix1n = [], [], [], [], [], [], []

    a1, i, j2, j3 = 0,0,0,0

    media_parray, acerto01 = [], []

    acerto2, acerto3, core1 = 0,0,0

    modelos = [None]*5000
    data_matrizes = [None]*5000
    recurso1, recurso2 = [None]*5000, [None]*5000

    array_geral = np.zeros(6, dtype=float)
    df1 = pd.DataFrame({'lautgh1': np.zeros(60, dtype = int), 'lautgh2': np.zeros(60, dtype = int)})

    data_matriz_float = []
    data_matriz_int = []
    data_array_float = []
    data_array_int = []
    array_geral_float = []
    data_acuracia_geral = []
    data_precisao_geral = []

    while i <= 999:
        print(24*'---')
        #print(len(media_parray))
        if len(media_parray) < 59:
            m = 0
            core1 = 0
        else:
            m = media_parray[len(media_parray) - 60]

        print(f'Número da Entrada - {i} | Acuracia_{core1 + 1}: {round(m,4)}')
        
        array2s, array2n, odd = coletarodd(i, j, data, array2s, array2n)
        array_geral_float.append(float)
        if odd == 0:
            break

        if i >= 361:
            print(24*"-'-")
            
            array_geral = placargeral(resultado, odd, array_geral)
            media_parray = placar60(df1, i, media_parray, resultado, odd)
            
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
                matriz_final_float, matriz_final_int, posicao0 = tranforsmar_final_matriz(click, array2s, array2n)
                print(f'Matrix_{click}: {[matriz_final_float.shape, matriz_final_int.shape]} | Posicao: {posicao0}')
                data_matriz_float.append(matriz_final_float), data_matriz_int.append(matriz_final_int)
                n = matriz_final_float.shape[1]
                array1, array2 = matriz_final_float[:,:(n - 3)], matriz_final_int[:,-1]
                models = reden(i, j, array1, array2)
                modelos[posicao0] = models
                data_matrizes[posicao0] = matriz_final_float
                print(f'Treinamento {click} realizado com sucesso ...  \n')
            print('***'*20)

        if i >= 360:
            core2 = i % 60
            y_pred1 = lista_predicao(i,len(modelos), modelos, data_matrizes)
            resultado = ponderar_lista(y_pred1)
            print(24*'*-')
            print(f'Proxima Entrada: {resultado}')
            print(24*'*-')


        i += 1
    if j >= 10:
        break

    print(f'Ordenandos os dados ...')

    data_array_float = np.array(array2s)
    data_array_int = np.array(array2n)

    name_acu = 'OrderAC' + str(j) + '.csv'
    name_res = 'OrderRE' + str(j) + '.csv'
    name_final_acu = '/home/darkcover/Documentos/Out/dados/DoIt/' + name_acu
    name_final_resultados = '/home/darkcover/Documentos/Out/dados/DoIt/' + name_res
    order1 = pd.DataFrame({'Acuracia': data_acuracia_geral, 'Precisao': data_precisao_geral})
    order1.to_csv(name_final_acu, index=False)

    data_final = pd.DataFrame({'Entrada': data_array_float, 'Resultado': data_array_int})
    data_final.to_csv(name_final_resultados, index=False)



