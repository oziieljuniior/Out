import numpy as np
import pandas as pd
import skfuzzy as fuzz
from scipy.stats import entropy,skew, kurtosis
import bisect
           

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
                odd = data['Odd'][i] 

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
        #print(f'Matriz binario1: {matrizbinario1.shape}')
        ##array1mediamovel, array1desviopadrao, array1entropia, array1assimetria, array1curtose
        arraymbinario1, arraydpbinario1, arrayebinario1, arrayabinario1, arraycbinario1 = [], [], [], [], []
        for i in range(matrizbinario1.shape[0]):
            media = np.mean(matrizbinario1[i,:-1])
            desvio = np.std(matrizbinario1[i,:-1], ddof=1)  # ddof=1 para amostra
            counts = np.bincount(matrizbinario1[i,:-1].astype(int), minlength=2)
            probas = (counts + 1e-9) / (counts.sum() + 2e-9)
            entropia = entropy(probas, base=2)          # sempre definido  ∈ [0,1]
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
    
    def corteArrayFloat(self, array):
        matriznormal = array
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
        return x1

    def gerar_matriz_float(self, array):
        return self.corteArrayFloat(self.matriz(120, array))

    def gerar_matriz_binaria(self, array, threshold):
        binarizado = [0 if val >= threshold else 1 for val in array]
        return self.corteArrayBinario(self.matriz(120, binarizado))

    def gerar_direcionalidade(self, array, limite, inverter=False):
        direcao, contagem = [], []
        flag, count = 1, 0
        for val in array:
            if (val > limite and not inverter) or (val <= limite and inverter):
                count = -1
                flag ^= 1  # alterna entre 0 e 1
            count += 1
            direcao.append(flag)
            contagem.append(count if not inverter else count)
        matriz_direcao = self.matriz(120, direcao)[:, :-1]
        matriz_contagem = self.matriz(120, contagem)[:, :-1]
        return matriz_direcao, matriz_contagem

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
        array2 = np.array(array1, dtype=np.float32)  # Converte para lista se necessário
        array1 = np.clip(np.array(array1, dtype=np.float32), 1.0, 6.0).tolist()
        print(len(array1), len(array2))
        
        # 1. Matriz normal
        x1 = self.gerar_matriz_float(array1)

        # 2. Matriz majorada
        array_majorado = [2.0 if v <= 2 else 4.0 if v >= 4 else v for v in array1]
        x2 = self.gerar_matriz_float(array_majorado)

        # 3. Fuzzy
        array_fuzzy, array_fcontinuo = zip(*[self.dual_fuzzy_classification_invertida(v) for v in array1])
        x3 = self.gerar_matriz_float(array_fuzzy)
        x4 = self.gerar_matriz_float(array_fcontinuo)

        # 4. Binarizações múltiplas
        thresholds = [3, 2, 4, 5, 1.5, 1.75, 2.25, 2.5, 2.75, 1.25, 1.4, 1.6]
        binarizados = [self.gerar_matriz_binaria(array1, th) for th in thresholds]
        x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = binarizados

        # 5. Direcionalidade 1 (> 10)
        x17, x18 = self.gerar_direcionalidade(array2, limite=10, inverter=False)

        # 6. Direcionalidade 2 (<= 1.02)
        x19, x20 = self.gerar_direcionalidade(array2, limite=1.02, inverter=True)

        # Limiares definidos em ordem crescente
        limiares = [1.05, 1.15, 1.30, 1.50, 1.75, 2.12, 2.70, 3.70, 5.88, 20, 50]

        array111 = [bisect.bisect(limiares, odd) for odd in array2]
        print(len(array111))
        matriz111 = self.matriz(120, array111)
        x21 = matriz111[:,:-1]                
               

        matrizX_final = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21), axis=1)

        array1binario1 = [0 if odd >= 3 else 1 for odd in array1]
        matrizbinario1 = self.matriz(120, array1binario1)
        
        matrizy_final = np.array(matrizbinario1[:, -1]).reshape(-1, 1)  # Última coluna de matrizbinario1 como y

        return matrizX_final, matrizy_final
    
    def estatisticaArrayFloat(self, array):
        array1 = array
        media = np.mean(array1)
        desvio = np.std(array1, ddof=1)  # ddof=1 para amostra
        skewness = skew(array1)
        curtose = kurtosis(array1)

        # Concatenar as matrizes de características normais
        x1 = np.append(array1, [media, desvio, skewness, curtose])
        #print(f'Matriz normal: {x1.shape}')

        
        return x1
    
    def estatisticaArrayBinario(self, array):
        array1binario = array
        media = np.mean(array1binario)
        desvio = np.std(array1binario, ddof=1)  # ddof=1 para amostra
        counts = np.bincount(array1binario, minlength=2)
        probas = (counts + 1e-9) / (counts.sum() + 2e-9)
        entropia = entropy(probas, base=2)          # sempre definido  ∈ [0,1]
        skewness = skew(array1binario)
        curtose = kurtosis(array1binario)

        # Concatenar as matrizes de características normais
        x5 = np.append(array1binario, [media, desvio, entropia, skewness, curtose])
        return x5
        
    def binarizar_array(array, threshold):
        return [0 if val >= threshold else 1 for val in array]

    def processa_direcao(self, array, limite, inverter=False):
        direcao, contagem = [], []
        flag, count = 1, 0
        for val in array:
            if (val > limite and not inverter) or (val <= limite and inverter):
                count = -1
                flag ^= 1  # Alterna entre 0 e 1
            count += 1
            direcao.append(flag)
            contagem.append(count if not inverter else flag)
        return direcao, contagem        

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
        array2 = array1

        # 1. Array normalizado
        array1 = np.clip(np.array(array1, dtype=np.float32), 1.0, 6.0).tolist()
        x1 = self.estatisticaArrayFloat(array1)

        # 2. Array majorado
        array1marjorado = [
            1.0 if val <= 2 else 4.0 if val >= 4 else val
            for val in array1
        ]
        x2 = self.estatisticaArrayFloat(array1marjorado)

        # 3. Fuzzy
        array1fuzzy, array1fcontinuo = zip(*[self.dual_fuzzy_classification_invertida(odd) for odd in array1])
        x3 = self.estatisticaArrayFloat(array1fuzzy)
        x4 = self.estatisticaArrayFloat(array1fcontinuo)

        # 4. Binarizações com múltiplos thresholds
        thresholds = [1.5, 2, 3, 4, 5, 1.75, 2.25, 2.75, 2.5, 1.25, 1.4, 1.6]
        estatisticas_binarias = []
        for t in thresholds:
            array_bin = self.binarizar_array(array1, t)
            estatisticas_binarias.append(self.estatisticaArrayBinario(array_bin))

        # desempacotar resultados em x5 a x16
        x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = estatisticas_binarias

        # 5. Direcionalidades baseadas em condições
        x17, x18 = self.processa_direcao(array2, limite=10, inverter=False)
        x19, x20 = self.processa_direcao(array2, limite=1.02, inverter=True)

        # Limiares definidos em ordem crescente
        limiares = [1.05, 1.15, 1.30, 1.50, 1.75, 2.12, 2.70, 3.70, 5.88, 20, 50]

        array111 = [bisect.bisect(limiares, odd) for odd in array2]
        print(len(array111))
        x21 = array111                
        
                
        matrizX_final = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21), axis=0)
        
        # Retorna somente a última linha (única janela possível)
        return matrizX_final.reshape(1, -1)
