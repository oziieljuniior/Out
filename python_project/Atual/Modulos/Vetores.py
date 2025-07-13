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

    def dual_fuzzy_classification_invertida(self, odd):
        """
        Classificação fuzzy invertida:
        Odds BAIXAS → alta confiança (1.0)
        Odds ALTAS → baixa confiança (0.25)
        
        Retorna:
            - fuzzy_val: valor simbólico invertido (1.0 = alta confiança)
            - tsk_val: valor contínuo invertido (menor odd → maior confiança)
        """
        odd_range = np.arange(1.0, 6.1, 0.1)
        
        # Conjuntos fuzzy
        baixo = fuzz.trimf(odd_range, [1, 1, 2])
        medio = fuzz.trimf(odd_range, [1.5, 3, 4.5])
        alto = fuzz.trimf(odd_range, [3.5, 5, 6])
        muito_alto = fuzz.trimf(odd_range, [4.5, 6, 6])
        
        # Graus de pertinência
        pert_baixo = fuzz.interp_membership(odd_range, baixo, odd)
        pert_medio = fuzz.interp_membership(odd_range, medio, odd)
        pert_alto = fuzz.interp_membership(odd_range, alto, odd)
        pert_muito_alto = fuzz.interp_membership(odd_range, muito_alto, odd)
        
        # Lógica fuzzy invertida (simbólica)
        pertinencias = {
            1.0:  pert_baixo,
            0.75: pert_medio,
            0.5:  pert_alto,
            0.25: pert_muito_alto
        }
        fuzzy_val = max(pertinencias, key=pertinencias.get)

        # TSK invertida – odds baixas → confiança alta
        y_baixo = 0.1 * odd + 0.9     # ex: odd = 1.2 → y ≈ 1.02
        y_medio = 0.3 * odd + 0.6
        y_alto = 0.5 * odd + 0.3
        y_muito_alto = 0.7 * odd + 0.1 # ex: odd = 5.5 → y ≈ 4.95

        # Mas queremos inverter: quanto MAIOR a odd, MENOR a confiança
        # Então calculamos o inverso proporcional (normalizado)
        pesos = np.array([pert_baixo, pert_medio, pert_alto, pert_muito_alto])
        saidas_orig = np.array([y_baixo, y_medio, y_alto, y_muito_alto])

        # Inversão linear da saída: maior valor → menor confiança
        saidas_norm = (np.max(saidas_orig) - saidas_orig) / (np.max(saidas_orig) - np.min(saidas_orig) + 1e-8)

        if pesos.sum() == 0:
            tsk_val = 0
        else:
            tsk_val = np.dot(pesos, saidas_norm) / pesos.sum()
        
        return fuzzy_val, tsk_val

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
    
    def corteArrayBinario(self, array1):
        """
        Garante que o array seja transformado em matriz.
        
        Args:
            array1 (list ou np.ndarray): Array de entrada.
            tamanho (int): Tamanho desejado do array.

        Returns:
            np.ndarray: Matriz ajustado para o tamanho especificado.
        """
        matrizbinario1 = array1
        print(f'Matriz binario1: {matrizbinario1.shape}')
        ##array1mediamovel, array1desviopadrao, array1entropia, array1assimetria, array1curtose
        arraymbinario1, arraydpbinario1, arrayebinario1, arrayabinario1, arraycbinario1 = [], [], [], [], []
        for i in range(matrizbinario1.shape[0]):
            media = np.mean(matrizbinario1[i,:-1])
            desvio = np.std(matrizbinario1[i,:-1], ddof=1)  # ddof=1 para amostra
            probas = np.bincount(matrizbinario1[i,:-1].astype(int), minlength=10)
            probas = probas / probas.sum()
            entropia = entropy(probas, base=2)
            skewness = skew(matrizbinario1[i,:-1])
            curtose = kurtosis(matrizbinario1[i,:-1])

            arraycbinario1.append(curtose)
            arrayabinario1.append(skewness)   
            arrayebinario1.append(entropia)
            arraydpbinario1.append(desvio)
            arraymbinario1.append(media)
        matrizmbinario1 = np.array(arraymbinario1).reshape(-1,1) #Matriz Media valores binário
        matrizdpbinario1 = np.array(arraydpbinario1).reshape(-1,1) #Matriz Desvio Padrão valores binário
        matrizebinario1 = np.array(arrayebinario1).reshape(-1,1) #Matriz Entropia valores
        matrizabinario1 = np.array(arrayabinario1).reshape(-1,1) #Matriz Assimetria valores
        matrizcbinario1 = np.array(arraycbinario1).reshape(-1,1) #Matriz Curtose valores
        # Concatenar as matrizes de características normais
        x6 = np.concatenate((matrizbinario1[:,:-1], matrizmbinario1, matrizdpbinario1, matrizebinario1, matrizabinario1, matrizcbinario1), axis=1)

        return x6
        


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
        matriznormal = self.matriz(60, array1)
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
                array1marjorado.append(2.0)
            elif array1[i] >= 4:
                array1marjorado.append(4.0)
            else:
                array1marjorado.append(array1[i])
        matrizmarjorado = self.matriz(60, array1marjorado)
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

        #array1fuzzy e array1valorcontínuo fuzzyficado
        array1fuzzy, array1fcontinuo = [], []
        for odd in array1:
            value1, value2 = self.dual_fuzzy_classification_invertida(odd)
            array1fuzzy.append(value1), array1fcontinuo.append(value2)
        matrizfuzzy = self.matriz(60, array1fuzzy)
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

        # Continuação array fuzzy contínuo
        matrizfcontinuo = self.matriz(60, array1fcontinuo)
        ##array1mediamovel, array1desviopadrao, array1entropia, array1assimetria, array1curtose
        arraymfcontinuo, arraydpfcontinuo, arrayafcontinuo, arraycfcontinuo = [], [], [], []
        for i in range(matrizfcontinuo.shape[0]):
            media = np.mean(matrizfcontinuo[i,:-1])
            desvio = np.std(matrizfcontinuo[i,:-1], ddof=1)  # ddof=1 para amostra
            skewness = skew(matrizfcontinuo[i,:-1])
            curtose = kurtosis(matrizfcontinuo[i,:-1])

            arraycfcontinuo.append(curtose)
            arrayafcontinuo.append(skewness)   
            arraymfcontinuo.append(media)
            arraydpfcontinuo.append(desvio)
        matrizmfcontinuo = np.array(arraymfcontinuo).reshape(-1,1) #Matriz Media valores fuzzy
        matrizdpfcontinuo = np.array(arraydpfcontinuo).reshape(-1,1) #Matriz Desvio Padrão valores fuzzy
        matrizafcontinuo = np.array(arrayafcontinuo).reshape(-1,1) #Matriz Assimetria valores
        matrizcfcontinuo = np.array(arraycfcontinuo).reshape(-1,1) #Matriz Curtose valores
        # Concatenar as matrizes de características normais
        x4 = np.concatenate((matrizfcontinuo[:,:-1], matrizmfcontinuo, matrizdpfcontinuo, matrizafcontinuo, matrizcfcontinuo), axis=1)
        #print(f'Matriz fuzzy: {x3.shape}')

        #array1binario1
        array1binario0 = [0 if odd >= 3 else 1 for odd in array1]
        matrizbinario0 = self.matriz(60, array1binario0)

        x5 = self.corteArrayBinario(matrizbinario0)
        #print(f'Matriz binario1: {x5.shape}')
        
        #array1binario1
        array1binario1 = [0 if odd >= 2 else 1 for odd in array1]
        matrizbinario1 = self.matriz(60, array1binario1)

        x6 = self.corteArrayBinario(matrizbinario1)
        #print(f'Matriz binario1: {x6.shape}')
        
        #array1binario2
        array1binario2 = [0 if odd >= 4 else 1 for odd in array1]
        matrizbinario2 = self.matriz(60, array1binario2)

        x7 = self.corteArrayBinario(matrizbinario2)
        #print(f'Matriz binario2: {x7.shape}')
        
        #array1binario3
        array1binario3 = [0 if odd >= 5 else 1 for odd in array1]
        matrizbinario3 = self.matriz(60, array1binario3)

        x8 = self.corteArrayBinario(matrizbinario3)
        #print(f'Matriz binario2: {x8.shape}')
        
        #array1binario4
        array1binario4 = [0 if odd >= 1.5 else 1 for odd in array1]
        matrizbinario4 = self.matriz(60, array1binario4)

        x9 = self.corteArrayBinario(matrizbinario4)
        #print(f'Matriz binario3: {x9.shape}')

        #array1binario5
        array1binario5 = [0 if odd >= 1.75 else 1 for odd in array1]
        matrizbinario5 = self.matriz(60, array1binario5)

        x10 = self.corteArrayBinario(matrizbinario5)
        #print(f'Matriz binario3: {x9.shape}')

        #array1binario6
        array1binario6 = [0 if odd >= 2.25 else 1 for odd in array1]
        matrizbinario6 = self.matriz(60, array1binario6)

        x11 = self.corteArrayBinario(matrizbinario6)
        #print(f'Matriz binario3: {x9.shape}')

        #array1binario7
        array1binario7 = [0 if odd >= 2.5 else 1 for odd in array1]
        matrizbinario7 = self.matriz(60, array1binario7)

        x12 = self.corteArrayBinario(matrizbinario7)
        #print(f'Matriz binario3: {x9.shape}')

        #array1binario8
        array1binario8= [0 if odd >= 2.75 else 1 for odd in array1]
        matrizbinario8 = self.matriz(60, array1binario8)

        x13 = self.corteArrayBinario(matrizbinario8)
        #print(f'Matriz binario3: {x9.shape}')

        #array1binario9
        array1binario9= [0 if odd >= 1.25 else 1 for odd in array1]
        matrizbinario9 = self.matriz(60, array1binario9)

        x14 = self.corteArrayBinario(matrizbinario9)
        #print(f'Matriz binario3: {x9.shape}')

        #array1binario10
        array1binario10= [0 if odd >= 1.4 else 1 for odd in array1]
        matrizbinario10 = self.matriz(60, array1binario10)

        x15 = self.corteArrayBinario(matrizbinario10)
        #print(f'Matriz binario3: {x9.shape}')

        #array1binario11
        array1binario11 = [0 if odd >= 1.6 else 1 for odd in array1]
        matrizbinario11 = self.matriz(60, array1binario11)

        x16 = self.corteArrayBinario(matrizbinario11)
        #print(f'Matriz binario3: {x9.shape}')
        
        matrizX_final = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), axis=1)

        array1binario1 = [0 if odd >= 3 else 1 for odd in array1]
        matrizbinario1 = self.matriz(60, array1binario1)
        
        matrizy_final = np.array(matrizbinario1[:, -1]).reshape(-1, 1)  # Última coluna de matrizbinario1 como y

        return matrizX_final, matrizy_final

    def transformar_entrada_predicao(self, array1):
        """
        Prepara a estrutura de entrada para predição com .predict().
        Assume que array1 contém as últimas 60 entradas (59 anteriores + 1 atual).
        
        Returns:
            np.ndarray: Array com shape (1, n_features) pronto para model.predict().
        """
        if len(array1) < 60:
            raise ValueError("É necessário fornecer ao menos 60 entradas para predição.")

        # Usa apenas os últimos 120 valores
        array1 = array1[-59:]

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

        #array1fuzzy e array1valorcontínuo fuzzyficado
        array1fuzzy, array1fcontinuo = [], []
        for odd in array1:
            value1, value2 = self.dual_fuzzy_classification_invertida(odd)
            array1fuzzy.append(value1), array1fcontinuo.append(value2)
        media = np.mean(array1fuzzy)
        desvio = np.std(array1fuzzy, ddof=1)  # ddof=1 para amostra
        skewness = skew(array1fuzzy)
        curtose = kurtosis(array1fuzzy)

        # Concatenar as matrizes de características normais
        x3 = np.append(array1fuzzy, [media, desvio, skewness, curtose])
        #print(f'Matriz fuzzy: {x3.shape}')

        # Continuação array fuzzy contínuo
        media = np.mean(array1fcontinuo)
        desvio = np.std(array1fcontinuo, ddof=1)  # ddof=1 para amostra
        skewness = skew(array1fcontinuo)
        curtose = kurtosis(array1fcontinuo)

        x4 = np.append(array1fcontinuo, [media, desvio, skewness, curtose])
        #print(f'Matriz fuzzy: {x3.shape}')
        
        #array1binario0
        array1binario = [0 if odd >= 1.5 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x5 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario1
        array1binario = [0 if odd >= 2 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x6 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario2
        array1binario = [0 if odd >= 3 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x7 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario3
        array1binario = [0 if odd >= 4 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x8 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario4
        array1binario = [0 if odd >= 5 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x9 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario5
        array1binario = [0 if odd >= 1.75 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x10 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario6
        array1binario = [0 if odd >= 2.25 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x11 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario7
        array1binario = [0 if odd >= 2.75 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x12 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario8
        array1binario = [0 if odd >= 2.5 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x13 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario9
        array1binario = [0 if odd >= 1.25 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x14 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario10
        array1binario = [0 if odd >= 1.4 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x15 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        #array1binario16
        array1binario = [0 if odd >= 1.6 else 1 for odd in array1]
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        probas = np.bincount(array1binario, minlength=10)
        probas = probas / probas.sum()
        entropia = entropy(probas, base=2)
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x16 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        #print(f'Matriz binario: {x4.shape}')

        matrizX_final = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), axis=0)
        
        # Retorna somente a última linha (única janela possível)
        return matrizX_final.reshape(1, -1)
