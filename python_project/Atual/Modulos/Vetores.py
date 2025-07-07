import numpy as np
import pandas as pd
import skfuzzy as fuzz
from scipy.stats import entropy,skew, kurtosis
           
            



class AjustesOdds:
    def __init__(self, array1):
        self.array1 = array1

    def coletarodd(self, i, inteiro, data, alavanca=True):
        """
        Função que coleta e organiza as entradas iniciais do banco de dados.
        Args:
            i (int): Valor inteiro não-negativo. Entrada que controla o loop principal. É um valor cumulativo.
            inteiro (int): Valor inteiro não-negativo. Entrada que determina até aonde os dados devem ser carregados automaticamente, através de um banco de dados.
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

        if i <= inteiro:
            if alavanca == True:
                odd = float(data['Entrada'][i].replace(",",'.'))
            else:
                odd = data['Entrada'][i] 

            if odd == 0:
                odd = 1
            print(f'Entrada: {odd}')
        else:
            odd = float(input("Entrada -> ").replace(",",'.'))

        if odd == 0:
            return self.array1, odd

        self.array1.append(odd)
        return self.array1, odd

    def fuzzy_classification(self, odd):
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
    
    def matriz(self, num_colunas, array1):
        """
        Gera uma matriz sequencial a partir de um array, com o número de colunas especificado.

        Args:
            array (list ou np.ndarray): Array de entrada.
            num_colunas (int): Número de colunas desejado na matriz.

        Returns:
            np.ndarray: Matriz sequencial.
        """
        if num_colunas > len(array1):
            raise ValueError("O número de colunas não pode ser maior que o tamanho do array.")

        # Número de linhas na matriz
        num_linhas = len(array1) - num_colunas + 1

        # Criando a matriz sequencial
        matriz = np.array([array1[i:i + num_colunas] for i in range(num_linhas)])
        return matriz
    
    def tranforsmar_final_matriz(self, array1):
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

        #array1normal
        array1 = np.clip(np.array(array1, dtype=np.float32), 1.0, 6.0).tolist()
        matriznormal = self.matriz(120, array1)
        ##array1mediamovel, array1desviopadrao, array1entropia, array1assimetria, array1curtose
        arraymnormal, arraydpnormal, arrayanormal, arraycnormal = [], [], [], []
        for i in range(matriznormal.shape[0]):
            media = np.mean(matriznormal[i,:-1])
            desvio = np.std(matriznormal[i,:-1], ddof=1)  # ddof=1 para amostra
            skewness = skew(matriznormal[i,:-1])
            curtose = kurtosis(matriznormal[i,:-1])

            arraycnormal.append(curtose)
            arrayanormal.append(skewness)   
            arraydpnormal.append(desvio)
            arraymnormal.append(media)
        matrizmnormal = np.array(arraymnormal).reshape(-1,1) #Matriz Media valores 
        matrizdpnormal = np.array(arraydpnormal).reshape(-1,1) #Matriz Desvio Padrão valores
        matrizanormal = np.array(arrayanormal).reshape(-1,1) #Matriz Assimetria valores
        matrizcnormal = np.array(arraycnormal).reshape(-1,1) #Matriz Curtose valores
        # Concatenar as matrizes de características normais
        x1 = np.concatenate((matriznormal[:,:-1], matrizmnormal, matrizdpnormal, matrizanormal, matrizcnormal), axis=1)
        #print(f'Matriz normal: {x1.shape}')

        #array1marjorado
        array1marjorado = []
        for i in range(len(array1)):
            if array1[i] <= 2:
                array1marjorado.append(1.0)
            elif array1[i] >= 4:
                array1marjorado.append(4.0)
            else:
                array1marjorado.append(array1[i])
        matrizmarjorado = self.matriz(120, array1marjorado)
        ##array1mediamovel, array1desviopadrao, array1entropia, array1assimetria, array1curtose
        arraymmarjorado, arraydpmarjorado, arrayamarjorado, arraycmarjorado = [], [], [], []
        for i in range(matrizmarjorado.shape[0]):
            media = np.mean(matrizmarjorado[i,:-1])
            desvio = np.std(matrizmarjorado[i,:-1], ddof=1)  # ddof=1 para amostra
            skewness = skew(matrizmarjorado[i,:-1])
            curtose = kurtosis(matrizmarjorado[i,:-1])

            arraycmarjorado.append(curtose)
            arrayamarjorado.append(skewness)   
            arraydpmarjorado.append(desvio)
            arraymmarjorado.append(media)
        matrizmmarjorado = np.array(arraymmarjorado).reshape(-1,1) #Matriz Media valores 
        matrizdpmarjorado = np.array(arraydpmarjorado).reshape(-1,1) #Matriz Desvio Padrão valores
        matrizamarjorado = np.array(arrayamarjorado).reshape(-1,1) #Matriz Assimetria valores
        matrizcmarjorado = np.array(arraycmarjorado).reshape(-1,1) #Matriz Curtose valores
        # Concatenar as matrizes de características normais
        x2 = np.concatenate((matrizmarjorado[:,:-1], matrizmmarjorado, matrizdpmarjorado, matrizamarjorado, matrizcmarjorado), axis=1)
        #print(f'Matriz Marjorada: {x2.shape}')

        #array1fuzzy
        array1fuzzy = [self.fuzzy_classification(odd) for odd in array1]
        matrizfuzzy = self.matriz(120, array1fuzzy)
        ##array1mediamovel, array1desviopadrao, array1entropia, array1assimetria, array1curtose
        arraymfuzzy, arraydpfuzzy, arrayafuzzy, arraycfuzzy = [], [], [], []
        for i in range(matrizfuzzy.shape[0]):
            media = np.mean(matrizfuzzy[i,:-1])
            desvio = np.std(matrizfuzzy[i,:-1], ddof=1)  # ddof=1 para amostra
            skewness = skew(matrizfuzzy[i,:-1])
            curtose = kurtosis(matrizfuzzy[i,:-1])

            arraycfuzzy.append(curtose)
            arrayafuzzy.append(skewness)   
            arraymfuzzy.append(media)
            arraydpfuzzy.append(desvio)
        matrizmfuzzy = np.array(arraymfuzzy).reshape(-1,1) #Matriz Media valores fuzzy
        matrizdpfuzzy = np.array(arraydpfuzzy).reshape(-1,1) #Matriz Desvio Padrão valores fuzzy
        matrizafuzzy = np.array(arrayafuzzy).reshape(-1,1) #Matriz Assimetria valores
        matrizcfuzzy = np.array(arraycfuzzy).reshape(-1,1) #Matriz Curtose valores
        # Concatenar as matrizes de características normais
        x3 = np.concatenate((matrizfuzzy[:,:-1], matrizmfuzzy, matrizdpfuzzy, matrizafuzzy, matrizcfuzzy), axis=1)
        #print(f'Matriz fuzzy: {x3.shape}')

        #array1binario
        array1binario = [1 if odd >= 3 else 0 for odd in array1]
        matrizbinario = self.matriz(120, array1binario)
        ##array1mediamovel, array1desviopadrao, array1entropia, array1assimetria, array1curtose
        arraymbinario, arraydpbinario, arrayebinario, arrayabinario, arraycbinario = [], [], [], [], []
        for i in range(matrizbinario.shape[0]):
            media = np.mean(matrizbinario[i,:-1])
            desvio = np.std(matrizbinario[i,:-1], ddof=1)  # ddof=1 para amostra
            probas = np.bincount(matrizbinario[i,:-1].astype(int), minlength=10)
            probas = probas / probas.sum()
            entropia = entropy(probas, base=2)
            skewness = skew(matrizbinario[i,:-1])
            curtose = kurtosis(matrizbinario[i,:-1])

            arraycbinario.append(curtose)
            arrayabinario.append(skewness)   
            arrayebinario.append(entropia)
            arraydpbinario.append(desvio)
            arraymbinario.append(media)
        matrizmbinario = np.array(arraymbinario).reshape(-1,1) #Matriz Media valores binário
        matrizdpbinario = np.array(arraydpbinario).reshape(-1,1) #Matriz Desvio Padrão valores binário
        matrizebinario = np.array(arrayebinario).reshape(-1,1) #Matriz Entropia valores
        matrizabinario = np.array(arrayabinario).reshape(-1,1) #Matriz Assimetria valores
        matrizcbinario = np.array(arraycbinario).reshape(-1,1) #Matriz Curtose valores
        # Concatenar as matrizes de características normais
        x4 = np.concatenate((matrizbinario[:,:-1], matrizmbinario, matrizdpbinario, matrizebinario, matrizabinario, matrizcbinario), axis=1)
        #print(f'Matriz binario: {x4.shape}')

        matrizX_final = np.concatenate((x1, x2, x3, x4), axis=1)
        matrizy_final = np.array(matrizbinario[:, -1]).reshape(-1, 1)  # Última coluna de matrizbinario como y

        return matrizX_final, matrizy_final

    def transformar_entrada_predicao(self, array1):
        """
        Prepara a estrutura de entrada para predição com .predict().
        Assume que array1 contém as últimas 120 entradas (119 anteriores + 1 atual).
        
        Returns:
            np.ndarray: Array com shape (1, n_features) pronto para model.predict().
        """
        if len(array1) < 120:
            raise ValueError("É necessário fornecer ao menos 120 entradas para predição.")

        # Usa apenas os últimos 120 valores
        array1 = array1[-119:]

        #array1normal
        array1 = np.clip(np.array(array1, dtype=np.float32), 1.0, 6.0).tolist()
        media = np.mean(array1)
        desvio = np.std(array1, ddof=1)  # ddof=1 para amostra
        skewness = skew(array1)
        curtose = kurtosis(array1)

        # Concatenar as matrizes de características normais
        x1 = np.append(array1, [media, desvio, skewness, curtose])
        #print(f'Matriz normal: {x1.shape}')

        #array1marjorado
        array1marjorado = []
        for i in range(len(array1)):
            if array1[i] <= 2:
                array1marjorado.append(1.0)
            elif array1[i] >= 4:
                array1marjorado.append(4.0)
            else:
                array1marjorado.append(array1[i])
        media = np.mean(array1marjorado)
        desvio = np.std(array1marjorado, ddof=1)  # ddof=1 para amostra
        skewness = skew(array1marjorado)
        curtose = kurtosis(array1marjorado)
        # Concatenar as matrizes de características normais
        x2 = np.append(array1marjorado, [media, desvio, skewness, curtose])
        #print(f'Matriz Marjorada: {x2.shape}')

        #array1fuzzy
        array1fuzzy = [self.fuzzy_classification(odd) for odd in array1]
        media = np.mean(array1fuzzy)
        desvio = np.std(array1fuzzy, ddof=1)  # ddof=1 para amostra
        skewness = skew(array1fuzzy)
        curtose = kurtosis(array1fuzzy)

        # Concatenar as matrizes de características normais
        x3 = np.append(array1fuzzy, [media, desvio, skewness, curtose])
        #print(f'Matriz fuzzy: {x3.shape}')

        #array1binario
        array1binario = [1 if odd >= 3 else 0 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x4 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        matrizX_final = np.concatenate((x1, x2, x3, x4), axis=0)
        
        # Retorna somente a última linha (única janela possível)
        return matrizX_final.reshape(1, -1)
