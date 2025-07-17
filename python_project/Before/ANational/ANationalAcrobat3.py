# Bibliotecas do sistema
import os
import sys
import time

# Caminho para os módulos internos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modulos.Placares import Placar
from Modulos.Vetores import AjustesOdds

# Bibliotecas externas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


    
### Carregar data
#/content/drive/MyDrive/Out/dados/odds_200k.csv
#/home/darkcover/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10 - 11-07-25_teste1.csv
#/home/darkcover/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10/Vitoria1_10 - game_teste3x1.csv
data = pd.read_csv('/home/darkcover01/Documentos/Out/python_project/Atual/data_treino/Vitoria1_10 - 11-07-25_teste1.csv')

array1, i = [], 0

inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

## Variáveis para salvar em um dataframe
data_matriz_float, data_matriz_int, array_geral_float, historico_janelas = [], [], [], [] 

placar = Placar()  # Inicializando o placar
vetores = AjustesOdds(array1)  # Inicializando a classe de ajustes de odds
### Produção
while i <= 210000:
    print(24*'---')
    print(f'Rodada - {i}')

######## -> Vetor de Entradas Unidimensional ##########        
    arrayodd, odd = vetores.coletarodd(i, inteiro, data)
    array_geral_float.append(odd)

    if odd == 0:
        break
######################################################

######## -> Placar ###################################      
    if i >= 241:
        print(24*"-'-")
        array_placar = placar.atualizar_geral(i, resultado, odd)
        print(f'Precisão Geral: {array_placar["Precisao_Geral"]:.2f}% \nPrecisão Rede Neural: {array_placar["Precisao_Sintetica"]:.2f}%')
        print(24*"-'-")
######################################################

######## -> Treinamento da Rede Neural ###############
    if i >= 240 and (i % 60) == 0:
        print('***'*20)
        ##### -> Vetores de Entradas #################
        print(f'Carregando dados ...')
        matriz_final_float, matriz_final_int = vetores.tranforsmar_final_matriz(arrayodd)
        print(f'Matrix: {[matriz_final_float.shape, matriz_final_int.shape]}')
        data_matriz_float.append(matriz_final_float), data_matriz_int.append(matriz_final_int)
        n = matriz_final_float.shape[1]
        array1, array2 = matriz_final_float, matriz_final_int
        ##############################################
        ##### -> Treinamento da Rede Neural ##########
        X = pd.DataFrame(array1)  # suas features
        y = array2.flatten()      # saída binária

        # Escalonamento
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Loop principal
        continua = True
        while continua is True:
            # Divisão treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Lista de modelos
            modelos = {
                "Regressão Logística": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "MLP (Rede Neural)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
            }

            # Treinamento e avaliação
            resultados = {}
            for nome, modelo in modelos.items():
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                resultados[nome] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Impressão dos resultados
            print("\n📊 Resultados dos Modelos:")
            for nome, resultado in resultados.items():
                metrica_geral = resultado['weighted avg']
                precision = metrica_geral['precision']
                recall = metrica_geral['recall']
                f1 = metrica_geral['f1-score']
                resultado0 = resultado['0']
                resultado1 = resultado['1']
                acuracia = resultado['accuracy']
                
                print(f"\nModelo: {nome}")
                print(f"  → Precisão: {precision:.4f}")
                print(f"  → Recall:   {recall:.4f}")
                print(f"  → F1-Score: {f1:.4f}")
                print(f"  → Acurácia: {acuracia:.4f}")
                print(f"  → Resultado 0: {resultado0['f1-score']:.4f} (Precision: {resultado0['precision']:.4f}, Recall: {resultado0['recall']:.4f})")
                print(f"  → Resultado 1: {resultado1['f1-score']:.4f} (Precision: {resultado1['precision']:.4f}, Recall: {resultado1['recall']:.4f})")

            # Visualização
            #plt.figure(figsize=(10, 6))
            #sns.barplot(x=list(resultados.keys()), y=[r['weighted avg']['f1-score'] for r in resultados.values()])
            #plt.title('F1-Score dos Modelos')
            #plt.xticks(rotation=45)
            #plt.ylabel('F1-Score')
            #plt.xlabel('Modelos')
            #plt.tight_layout()
            #plt.show()

            # Escolha de modelo
            print("\nMenu de Modelos:")
            print(" 0. Treinar novamente")
            print(" 1. Regressão Logística")
            print(" 2. Random Forest")
            print(" 3. Gradient Boosting")
            print(" 4. SVM")
            print(" 5. KNN")
            print(" 6. MLP (Rede Neural)")

            try:
                entrada = int(input("Escolha o modelo para produção (0 a 6): "))
            except ValueError:
                print("Entrada inválida. Digite um número de 0 a 6.")
                continue

            if entrada == 0:
                print("\nReiniciando treinamento...\n")
                continue
            elif entrada in range(1, 7):
                modelo_escolhido_nome = list(modelos.keys())[entrada - 1]
                modelo_escolhido = modelos[modelo_escolhido_nome]
                print(f"\n✅ Modelo escolhido para produção: {modelo_escolhido_nome}")
                continua = False
                break
            else:
                print("Número inválido. Tente novamente.\n")

        
        ##############################################
######################################################
            
    if i >= 240:
        #### -> Predição da Rede Neural ##############
        print(24*'*-')
        Apredicao = vetores.transformar_entrada_predicao(arrayodd)
        #print(f'Predição: {type(Apredicao)} | {len(Apredicao)}')
        res = modelo_escolhido.predict(Apredicao)
        if res[0] == 1:
            resultado = 0
        else:
            resultado = 1
        
        print(f'Proxima Entrada: {resultado}')
        print(24*'*-')
        ##############################################

    i += 1


