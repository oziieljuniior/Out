import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, binomtest
import time
import os
import pickle
from datetime import datetime

class ModelManager:
    def __init__(self, path_modelo='/home/darkcover/Documentos/Out/python_project/OTAMA/Funcoes', path_data='/home/darkcover/Documentos/Out/dados/Saidas/FUNCOES'):
        self.path_modelo = path_modelo
        self.path_data = path_data

    # Função para salvar o modelo
    def salvar_modelo(self, modelo, nome_arquivo):
        if not os.path.exists(self.path_modelo):
            os.makedirs(self.path_modelo)
        
        caminho_arquivo = os.path.join(self.path_modelo, f"{nome_arquivo}.pkl")
        with open(caminho_arquivo, 'wb') as arquivo:
            pickle.dump(modelo, arquivo)
        print(f"Modelo salvo como {nome_arquivo}.pkl na pasta {self.path_modelo}")

    # Função para carregar os modelos salvos e listar as opções
    def listar_modelos_salvos(self):
        arquivos = [f for f in os.listdir(self.path_modelo) if f.endswith('.pkl')]
        
        if len(arquivos) == 0:
            print("Nenhum modelo salvo encontrado.")
            return None
        
        print("Modelos disponíveis:")
        for i, arquivo in enumerate(arquivos):
            print(f"{i+1}. {arquivo}")
        
        escolha = int(input("Escolha o número do modelo que deseja carregar: "))
        
        if escolha < 1 or escolha > len(arquivos):
            print("Escolha inválida.")
            return None
        
        caminho_arquivo = os.path.join(self.path_modelo, arquivos[escolha-1])
        with open(caminho_arquivo, 'rb') as arquivo:
            modelo = pickle.load(arquivo)
        
        print(f"Modelo {arquivos[escolha-1]} carregado com sucesso.")
        return modelo

    # Função para salvar os dados em formato CSV com data e hora no nome do arquivo
    def salvar_data(self, data, nome_arquivo_base):
        if not os.path.exists(self.path_data):
            os.makedirs(self.path_data)
        
        # Capturar a data e hora atual
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Combinar o nome do arquivo base com a data e hora
        nome_arquivo = f"{nome_arquivo_base}_{timestamp}.csv"
        caminho_arquivo = os.path.join(self.path_data, nome_arquivo)
        
        # Verificando se 'data' é um DataFrame ou lista de dados
        if isinstance(data, pd.DataFrame):
            data.to_csv(caminho_arquivo, index=False)
        elif isinstance(data, list):
            # Se for uma lista de listas ou lista de dicionários
            df = pd.DataFrame(data)
            df.to_csv(caminho_arquivo, index=False)
        else:
            raise ValueError("O formato dos dados não é suportado. Use um DataFrame ou uma lista.")
        
        print(f"Dados salvos como {nome_arquivo} na pasta {self.path_data}")

    # Função para carregar e listar arquivos CSV
    def carregar_data(self):
        arquivos = [f for f in os.listdir(self.path_data) if f.endswith('.csv')]
        
        if len(arquivos) == 0:
            print("Nenhum arquivo CSV encontrado.")
            return None
        
        print("Arquivos CSV disponíveis:")
        for i, arquivo in enumerate(arquivos):
            print(f"{i+1}. {arquivo}")
        
        escolha = int(input("Escolha o número do arquivo que deseja carregar: "))
        
        if escolha < 1 or escolha > len(arquivos):
            print("Escolha inválida.")
            return None
        
        caminho_arquivo = os.path.join(self.path_data, arquivos[escolha-1])
        
        # Carregando o arquivo CSV usando pandas
        data = pd.read_csv(caminho_arquivo)
        print(f"Arquivo {arquivos[escolha-1]} carregado com sucesso.")
        
        return data
