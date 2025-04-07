# Instanciar a classe
processor = FuzzyOddsProcessor()

# Simulando uso com DataFrame
import pandas as pd
data = pd.DataFrame({'Entrada': ['2,3', '4,8', '6,0', '0', '7,2']})
array2s = []
array2n = []

# Usar a função
for i in range(len(data)):
    array2s, array2n, odd = processor.coletar_odd(i, 4, data, array2s, array2n)

print("Array de confiança:", array2s)
print("Array de classe binária:", array2n)
