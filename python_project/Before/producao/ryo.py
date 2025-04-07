import numpy as np
import matplotlib.pyplot as plt


"""
Aqui faz sentido trabalhar com este modelo, visto que as últimas 60 odds devem ter uma oscilação similar, olhar imagem de comparação entre esses modelos
"""


# Função que gera oscilações pseudo-aleatórias com erro controlado
def pseudo_oscillation(x, amplitude, base, noise_factor):
    noise = np.random.normal(0, noise_factor, len(x))
    oscillation = base + amplitude * np.sin(2 * np.pi * x / 500) + noise
    return oscillation

x_data = np.linspace(0, 1000, 1000)
y_data = pseudo_oscillation(x_data, 0.15, 0.45, 0.05)  # Oscila entre 0.30 e 0.60 com erro de 0.05

plt.plot(x_data, y_data)
plt.show()
