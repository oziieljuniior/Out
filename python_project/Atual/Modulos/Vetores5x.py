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

    def dual_fuzzy_classification_invertida(self, odd, *, odd_min=1.0, odd_max=10.0):
        """
        Classificação fuzzy invertida (1 → maior confiança ; 10 → menor confiança).

        Args
        ----
        odd : float
            Odd a ser classificada.
        odd_min, odd_max : float, optional
            Limites do universo (default 1-10).

        Returns
        -------
        fuzzy_val : float
            Confiança simbólica (discreto): 1.00, 0.85, 0.70, 0.55, 0.40 ou 0.25
        tsk_val   : float
            Confiança contínua normalizada (0-1), via inferência TSK invertida.
        """

        # Universo
        odd_range = np.arange(odd_min, odd_max + 0.1, 0.1)

        # ---------- Conjuntos fuzzy (triangulares) ----------
        muito_baixa   = fuzz.trimf(odd_range, [odd_min, odd_min, 2])
        baixa         = fuzz.trimf(odd_range, [1.5, 3, 4])
        media_baixa   = fuzz.trimf(odd_range, [3.5, 5, 6])
        media_alta    = fuzz.trimf(odd_range, [5.5, 7, 8])
        alta          = fuzz.trimf(odd_range, [7.5, 9, 9.5])
        muito_alta    = fuzz.trimf(odd_range, [9, odd_max, odd_max])

        # ---------- Graus de pertinência ----------
        pert = [
            fuzz.interp_membership(odd_range, muito_baixa, odd),
            fuzz.interp_membership(odd_range, baixa, odd),
            fuzz.interp_membership(odd_range, media_baixa, odd),
            fuzz.interp_membership(odd_range, media_alta, odd),
            fuzz.interp_membership(odd_range, alta, odd),
            fuzz.interp_membership(odd_range, muito_alta, odd)
        ]

        # ---------- Saídas fuzzy simbólicas (confiança) ----------
        simbolos = [1.00, 0.85, 0.70, 0.55, 0.40, 0.25]
        pertinencias = dict(zip(simbolos, pert))
        fuzzy_val = max(pertinencias, key=pertinencias.get)

        # ---------- Saídas TSK lineares (antes da inversão) ----------
        y_vals = np.array([
            0.01 * odd + 0.99,   # muito_baixa
            0.02 * odd + 0.90,   # baixa
            0.03 * odd + 0.80,   # media_baixa
            0.04 * odd + 0.60,   # media_alta
            0.05 * odd + 0.40,   # alta
            0.06 * odd + 0.20    # muito_alta
        ])

        pesos = np.array(pert)

        # ---------- Inversão (quanto maior a odd, menor confiança) ----------
        ymax, ymin = y_vals.max(), y_vals.min()
        saidas_norm = (ymax - y_vals) / (ymax - ymin + 1e-12)

        tsk_val = 0.0 if pesos.sum() == 0 else float(np.dot(pesos, saidas_norm) / pesos.sum())

        return fuzzy_val, tsk_val

    def fuzzy_classification(self, odd, *, odd_min=1.0, odd_max=10.0):
        """
        Classificação fuzzy direta (odds baixas = pouca confiança,
        odds altas = muita confiança).

        Retorna um valor simbólico discreto:
            0.25, 0.40, 0.55, 0.70, 0.85 ou 1.00
        """

        # Universo
        odd_range = np.arange(odd_min, odd_max + 0.1, 0.1)

        # -------- Conjuntos fuzzy (triangulares) --------
        muito_baixa   = fuzz.trimf(odd_range, [odd_min, odd_min, 2])
        baixa         = fuzz.trimf(odd_range, [1.5, 3, 4])
        media_baixa   = fuzz.trimf(odd_range, [3.5, 5, 6])
        media_alta    = fuzz.trimf(odd_range, [5.5, 7, 8])
        alta          = fuzz.trimf(odd_range, [7.5, 9, 9.5])
        muito_alta    = fuzz.trimf(odd_range, [9, odd_max, odd_max])

        # -------- Graus de pertinência --------
        pert = [
            fuzz.interp_membership(odd_range, muito_baixa, odd),
            fuzz.interp_membership(odd_range, baixa, odd),
            fuzz.interp_membership(odd_range, media_baixa, odd),
            fuzz.interp_membership(odd_range, media_alta, odd),
            fuzz.interp_membership(odd_range, alta, odd),
            fuzz.interp_membership(odd_range, muito_alta, odd)
        ]

        # -------- Valores simbólicos de confiança --------
        simbolos = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]

        max_pert = max(pert)
        if max_pert == 0:
            return 0.0          # sem pertinência → sem confiança

        fuzzy_val = simbolos[pert.index(max_pert)]
        return fuzzy_val
    
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
    
    def corteArrayBinario(self, matriz):
        """
        Calcula, para cada linha (ignorando a última coluna), as
        estatísticas {média, desvio-padrão, entropia, skew, curtose}
        nas janelas finais de 599, 479, 359, 239 e 119 elementos.

        Retorna
        -------
        X_out : np.ndarray
            Concatenação da parte original (sem a última coluna)
            com 25 novas colunas de features na ordem:
                [mean, std, ent, skew, kurt] × [599, 479, 359, 239, 119]
        """
        # ------------------ pré-processamento ---------------------------
        base = np.asarray(matriz, dtype=np.float32)[:, :-1]   # remove última coluna
        n_rows, n_cols = base.shape
        win_sizes = [599, 479, 359, 239, 119]

        # ------------------ cálculo das features ------------------------
        feats = []   # cada linha conterá 25 valores

        for row in base:
            line_feats = []
            for w in win_sizes:
                seg = row[-w:] if w <= n_cols else row        # trata linhas curtas

                # média e desvio-padrão
                line_feats.append(np.mean(seg, dtype=np.float32))
                line_feats.append(np.std(seg, ddof=1, dtype=np.float32))

                # entropia binária
                counts = np.bincount(seg.astype(int), minlength=2)
                probas  = (counts + 1e-9) / (counts.sum() + 2e-9)
                line_feats.append(float(entropy(probas, base=2)))  # ∈ [0,1]

                # skewness e curtose
                line_feats.append(float(skew(seg)))
                line_feats.append(float(kurtosis(seg)))
            feats.append(line_feats)

        feats = np.asarray(feats, dtype=np.float32)           # shape = (n_rows, 25)

        # ------------------ saída final ---------------------------------
        X_out = np.concatenate([base, feats], axis=1)         # (n_rows, n_cols-1 + 25)
        return X_out
    
    def corteArrayFloat(self, matriz):
        """
        Enriquecer cada linha da matriz com estatísticas das janelas finais.
        
        Retorna:
            X_out (np.ndarray):
                [dados_originais_sem_última_col | 20 novas features]
                Cada grupo de 4 features corresponde a um tamanho de janela
                na ordem: média, desvio-padrão, skew, curtose
                para janelas 599, 479, 359, 239 e 119 (nessa ordem).
        """
        # ---- preparação --------------------------------------------------------
        base = np.asarray(matriz, dtype=np.float32)[:, :-1]   # descarta última coluna
        n_rows, n_cols = base.shape
        win_sizes = [599, 479, 359, 239, 119]                # janelas desejadas
        
        # ---- cálculo das features ---------------------------------------------
        feats = []  # lista de linhas de features
        
        for row in base:
            row_feats = []
            for w in win_sizes:
                seg = row[-w:] if w <= n_cols else row       # trata janelas > n_cols
                row_feats.extend([
                    np.mean(seg, dtype=np.float32),
                    np.std(seg, ddof=1, dtype=np.float32),
                    skew(seg),                               # scipy retorna float64
                    kurtosis(seg)
                ])
            feats.append(row_feats)
        
        feats = np.asarray(feats, dtype=np.float32)          # shape = (n_rows, 20)
        
        # ---- saída final -------------------------------------------------------
        X_out = np.concatenate([base, feats], axis=1)        # shape = (n_rows, n_cols-1+20)
        return X_out

    def gerar_matriz_float(self, array):
        return self.corteArrayFloat(self.matriz(1200, array))

    def gerar_matriz_binaria(self, array, threshold):
        binarizado = [0 if val >= threshold else 1 for val in array]
        return self.corteArrayBinario(self.matriz(1200, binarizado))

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
        matriz_direcao = self.matriz(1200, direcao)[:, :-1]
        matriz_contagem = self.matriz(1200, contagem)[:, :-1]
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
        array1 = np.clip(np.array(array1, dtype=np.float32), 1.0, 10.0).tolist()
        print(len(array1), len(array2))
        
        # 1. Matriz normal
        x1 = self.gerar_matriz_float(array1)

        # 2. Matriz majorada
        array_majorado = [3 if v <= 3 else 6 if v >= 6 else v for v in array1]
        x2 = self.gerar_matriz_float(array_majorado)

        # 3. Fuzzy
        array_fuzzy, array_fcontinuo = zip(*[self.dual_fuzzy_classification_invertida(v) for v in array1])
        x3 = self.gerar_matriz_float(array_fuzzy)
        x4 = self.gerar_matriz_float(array_fcontinuo)

        # 4. Binarizações múltiplas
        thresholds = [1.05, 1.5, 2, 3, 6]
        binarizados = [self.gerar_matriz_binaria(array1, th) for th in thresholds]
        x5, x6, x7, x8, x9 = binarizados

        arrayint1 = []
        for i in range(len(array2)):
            if array2[i] <= 10:
                if array2[i] <= 3:
                    arrayint1.append(0)
                else:
                    arrayint1.append(1)
            else:
                arrayint1.append(10)
        x10 = self.gerar_matriz_float(arrayint1)   

        arrayint2 = []
        for i in range(len(array2)):
            if array2[i] <= 10:
                if array2[i] <= 5:
                    if array2[i] <= 3:
                        arrayint2.append(0)
                    else:
                        arrayint2.append(1)
                else:
                    arrayint2.append(5)
            else:
                arrayint2.append(10)
        x11 = self.gerar_matriz_float(arrayint2)
        
        arrayint3 = []
        for i in range(len(array2)):
            if array2[i] <= 10:
                if array2[i] <= 3:
                    if array2[i] <= 1.01:
                        arrayint3.append(-10)
                    else:
                        arrayint3.append(0)
                else:
                    arrayint3.append(1)
            else:
                arrayint3.append(10)
        x12 = self.gerar_matriz_float(arrayint3)   

        arrayint4 = []
        for i in range(len(array2)):
            if array2[i] <= 10:
                if array2[i] <= 5:
                    if array2[i] <= 3:
                        if array2[i] <= 1.01:
                            arrayint4.append(-10)
                        else:
                            arrayint4.append(0)
                    else:
                        arrayint4.append(1)
                else:
                    arrayint4.append(5)
            else:
                arrayint4.append(10)
        x13 = self.gerar_matriz_float(arrayint2)
        
        # 5. Direcionalidade 1 (> 3)
        x17, x18 = self.gerar_direcionalidade(array2, limite=10, inverter=False)

        # 6. Direcionalidade 2 (<= 1)
        x19, x20 = self.gerar_direcionalidade(array2, limite=1, inverter=True)       

        matrizX_final = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x17, x18, x19, x20), axis=1)

        array1binario1 = [0 if odd >= 5 else 1 for odd in array1]
        matrizbinario1 = self.matriz(1200, array1binario1)
        
        matrizy_final = np.array(matrizbinario1[:, -1]).reshape(-1, 1)  # Última coluna de matrizbinario1 como y

        return matrizX_final, matrizy_final
    
    def estatisticaArrayFloat(self, array):
        """
        Calcula estatísticas em janelas finais menores de um vetor (len = 599)
        e retorna um único array concatenando o vetor original + features.

        Retorna
        -------
        x_out : np.ndarray   (shape = (619,))
            [array_original | média, std, skew, kurt] × 5 janelas
        """
        arr = np.asarray(array, dtype=np.float32)
        
        win_sizes = [599, 479, 359, 239, 119]   # “escala menor”

        feats = []
        for w in win_sizes:
            seg = arr[-w:]
            feats.extend([
                np.mean(seg, dtype=np.float32),
                np.std (seg, ddof=1, dtype=np.float32),
                float(skew(seg)),
                float(kurtosis(seg))
            ])

        x_out = np.concatenate([arr, np.array(feats, dtype=np.float32)])
        return x_out

    def estatisticaArrayBinario(self, array):
        """
        Recebe um vetor binário de tamanho 599 e devolve:
            [array_original | média, std, entropia, skew, kurt] × 5 janelas

        Saída:
            np.ndarray de shape (624,)
        """
        arr = np.asarray(array, dtype=np.float32)
        
        win_sizes = [599, 479, 359, 239, 119]   # janelas “escala menor”
        feats = []

        for w in win_sizes:
            seg = arr[-w:]

            # média e desvio-padrão
            feats.append(np.mean(seg, dtype=np.float32))
            feats.append(np.std(seg, ddof=1, dtype=np.float32))

            # entropia binária
            counts = np.bincount(seg.astype(int), minlength=2)
            probas  = (counts + 1e-9) / (counts.sum() + 2e-9)
            feats.append(float(entropy(probas, base=2)))  # ∈ [0,1]

            # skewness e curtose
            feats.append(float(skew(seg)))
            feats.append(float(kurtosis(seg)))

        x_out = np.concatenate([arr, np.array(feats, dtype=np.float32)])
        return x_out

    def binarizar_array(self, array, threshold):
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
        Assume que array1 contém as últimas 1200 entradas (1199 anteriores + 1 atual).
        
        Returns:
            np.ndarray: Array com shape (1, n_features) pronto para model.predict().
        """
        if len(array1) < 1200:
            raise ValueError("É necessário fornecer ao menos 1200 entradas para predição.")

        # Usa apenas os últimos 1200 valores
        array1 = array1[-1199:]
        array2 = array1

        # 1. Array normalizado
        array1 = np.clip(np.array(array1, dtype=np.float32), 1.0, 10.0).tolist()
        x1 = self.estatisticaArrayFloat(array1)

        # 2. Array majorado
        array1marjorado = [
            3.0 if val <= 3 else 6 if val >= 6 else val
            for val in array1
        ]
        x2 = self.estatisticaArrayFloat(array1marjorado)

        # 3. Fuzzy
        array1fuzzy, array1fcontinuo = zip(*[self.dual_fuzzy_classification_invertida(odd) for odd in array1])
        x3 = self.estatisticaArrayFloat(array1fuzzy)
        x4 = self.estatisticaArrayFloat(array1fcontinuo)

        # 4. Binarizações com múltiplos thresholds
        thresholds = [1.05, 1.5, 2, 3 , 6]
        estatisticas_binarias = []
        for t in thresholds:
            array_bin = self.binarizar_array(array1, t)
            estatisticas_binarias.append(self.estatisticaArrayBinario(array_bin))

        # desempacotar resultados em x5 a x16
        x5, x6, x7, x8, x9 = estatisticas_binarias

        # 5. Direcionalidades baseadas em condições
        x17, x18 = self.processa_direcao(array2, limite=10, inverter=False)
        x19, x20 = self.processa_direcao(array2, limite=1, inverter=True)

        
        arrayint1 = []
        for i in range(len(array2)):
            if array2[i] <= 10:
                if array2[i] <= 3:
                    arrayint1.append(0)
                else:
                    arrayint1.append(1)
            else:
                arrayint1.append(10)
        x10 = self.estatisticaArrayFloat(arrayint1)
        
        arrayint2 = []
        for i in range(len(array2)):
            if array2[i] <= 10:
                if array2[i] <= 5:
                    if array2[i] <= 3:
                        arrayint2.append(0)
                    else:
                        arrayint2.append(1)
                else:
                    arrayint2.append(5)
            else:
                arrayint2.append(10)
        x11 = self.estatisticaArrayFloat(arrayint2) 
        
        arrayint3 = []
        for i in range(len(array2)):
            if array2[i] <= 10:
                if array2[i] <= 3:
                    if array2[i] <= 1.01:
                        arrayint3.append(-10)
                    else:
                        arrayint3.append(0)
                else:
                    arrayint3.append(1)
            else:
                arrayint3.append(10)
        x12 = self.estatisticaArrayFloat(arrayint3)
        
        arrayint4 = []
        for i in range(len(array2)):
            if array2[i] <= 10:
                if array2[i] <= 5:
                    if array2[i] <= 3:
                        if array2[i] <= 1.01:
                            arrayint4.append(-10)
                        else:
                            arrayint4.append(0)
                    else:
                        arrayint4.append(1)
                else:
                    arrayint4.append(5)
            else:
                arrayint4.append(10)
        x13 = self.estatisticaArrayFloat(arrayint4)       
        
                
        matrizX_final = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x17, x18, x19, x20), axis=0)
        
        # Retorna somente a última linha (única janela possível)
        return matrizX_final.reshape(1, -1)
