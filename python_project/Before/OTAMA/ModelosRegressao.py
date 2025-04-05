import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class ModeloRegressao:
    def __init__(self, peso_1=0.5, peso_2=0.5):
        """
        Inicializa a classe com os pesos atribuídos aos dois métodos de cálculo de correlação.
        
        Args:
        - peso_1: Peso atribuído ao cálculo de médias móveis (padrão = 0.5).
        - peso_2: Peso atribuído ao cálculo de desvio padrão (padrão = 0.5).
        """
        self.peso_1 = peso_1
        self.peso_2 = peso_2
        self.modelo = None

    def calcular_coeficiente(self, correlacao_1, correlacao_2):
        """
        Calcula o coeficiente ideal baseado nos dois métodos de correlação e seus pesos.
        
        Args:
        - correlacao_1: Correlação do método 1 (ex: médias móveis).
        - correlacao_2: Correlação do método 2 (ex: desvio padrão).
        
        Retorna:
        - coeficiente ponderado.
        """
        # Normalizar os pesos
        peso_total = self.peso_1 + self.peso_2
        peso_1_normalizado = self.peso_1 / peso_total
        peso_2_normalizado = self.peso_2 / peso_total
        
        # Calcula o coeficiente ponderado
        coeficiente = (peso_1_normalizado * correlacao_1) + (peso_2_normalizado * correlacao_2)
        return coeficiente

    def ajustar_regressao(self, X, y, plotar_grafico=True):
        """
        Ajusta uma regressão linear com os dados fornecidos e, opcionalmente, gera um gráfico.
        
        Args:
        - X: Variáveis independentes (entrada).
        - y: Variáveis dependentes (alvo).
        - plotar_grafico: Booleano para definir se o gráfico será gerado.
        
        Retorna:
        - O modelo de regressão linear treinado.
        """
        # Divisão dos dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Cria e treina o modelo de regressão linear
        self.modelo = LinearRegression()
        self.modelo.fit(X_train, y_train)

        # Realiza previsões com os dados de teste
        y_pred = self.modelo.predict(X_test)

        # Gera o gráfico se solicitado
        if plotar_grafico:
            plt.figure(figsize=(10, 6))
            plt.plot(y_test, label="Real", color='red')
            plt.plot(y_pred, label="Previsão", color='blue', linestyle='--')
            plt.title("Regressão Modular: Valores Reais vs Previstos")
            plt.legend()
            plt.show()

        return self.modelo

# Exemplo de uso da classe
# X = suas entradas (np.array ou DataFrame)
# y = suas saídas (valores reais)
# modelo = ModeloRegressao(peso_1=0.6, peso_2=0.4)
# coeficiente = modelo.calcular_coeficiente(correlacao_medias_moveis, correlacao_desvio_padrao)
# regressao_modelo = modelo.ajustar_regressao(X, y)
