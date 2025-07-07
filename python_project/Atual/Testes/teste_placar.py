import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from Modulos.Placares import Placar

placar = Placar()
for i in range(240, 260):
    resultado = np.random.choice([0, 1])
    odd = np.random.uniform(1.0, 5.0)
    saida = placar.atualizar_geral(i, resultado, odd)
    print(f"[{i}] Resultado={resultado} | Odd={odd:.2f} →", saida)
