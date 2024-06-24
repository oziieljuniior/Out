import pandas as pd
import numpy as np

def carregar_dados(caminho):
    data = pd.read_csv(caminho)
    data = data.drop(columns=['Unnamed: 0'])
    data = data.rename(columns={'Odd_Categoria': 'odd_saida'})
    return data

def inicializar_variaveis():
    return [], 0, 0, 1

def criar_dataframe_inicial():
    entrada_inicial = {
        'Rodada': [0], 'level': [1], 'apostar': [0], 
        'acerto': [0], 'contagem': [0], 'odd':[0], 'odd_entrada': [0],
        'odd_saida': [0], 'media5': [0], 'media10': [0], 'media20': [0], 
        'media40': [0], 'media80': [0], 'media160': [0], 
        'media320': [0], 'media640': [0]
    }
    return pd.DataFrame(entrada_inicial)

def calcular_medias(lista_entradas):
    ultimas5 = lista_entradas[-5:]
    medias = {'media5': sum(ultimas5) / 5}
    
    for i in [10, 20, 40, 80, 160, 320, 640]:
        if len(lista_entradas) >= i:
            medias[f'media{i}'] = sum(lista_entradas[-i:]) / i
        else:
            medias[f'media{i}'] = 0
    return medias

def decidir_apostar(medias):
    condicoes = [
        medias['media5'] <= 2.5 and medias['media5'] != 0,
        medias['media10'] <= 3.5 and medias['media10'] != 0,
        medias['media20'] <= 4.5 and medias['media20'] != 0,
        medias['media40'] <= 4.8 and medias['media40'] != 0,
        medias['media80'] <= 5.44 and medias['media80'] != 0,
        medias['media160'] <= 5.47 and medias['media160'] != 0,
        medias['media320'] <= 5.718 and medias['media320'] != 0,
        medias['media640'] <= 5.889 and medias['media640'] != 0
    ]
    return any(condicoes)

def processar_entradas(data_inicial):
    lista_entradas, apostar, contagem, level = inicializar_variaveis()
    data_final = criar_dataframe_inicial()

    for idx, row in data_inicial.iterrows():
        odd = row['Odd']
        odd_saida = row['odd_saida']
        odd_entrada = row['odd_entrada']
        
        i = odd_saida
        acerto = 0

        if apostar == 1:
            if odd_saida >= 4:
                acerto = 1
                contagem += 1
                if contagem == 15:
                    level += 1
                    contagem = 0
            else:
                acerto = 0
                contagem -= 2
                if contagem <= -10:
                    level = 1
                    contagem = 0
        else:
            if odd_saida >= 4:
                acerto = 1
            else:
                acerto = 0

        lista_entradas.append(i)

        if len(lista_entradas) < 6:
            data_final.loc[len(data_final.index)] = [len(lista_entradas), level, apostar, acerto, contagem, odd, odd_entrada, odd_saida, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            medias = calcular_medias(lista_entradas)
            data_final.loc[len(data_final.index)] = [len(lista_entradas), level, apostar, acerto, contagem, odd, odd_entrada, odd_saida] + list(medias.values())
            apostar = 0

        if len(lista_entradas) >= 5:
            medias = calcular_medias(lista_entradas)
            if decidir_apostar(medias):
                apostar = 1
                #print("APOSTAR NA PROXIMA RODADA")
            else:
                apostar = 0

    return data_final

def salvar_dados(data_final, caminho):
    data_final.to_csv(caminho, index=False)

if __name__ == "__main__":
    caminho_entrada = '/home/darkcover/Documentos/Out/dados/odds_200k.csv'
    caminho_saida = '/home/darkcover/Documentos/Out/dados/data_final2.csv'
    
    data_inicial = carregar_dados(caminho_entrada)
    data_final = processar_entradas(data_inicial)
    salvar_dados(data_final, caminho_saida)
    print("Processo concluído!")
