##Como criar um data frame e adicionar novos elementos nele.

# Importando a biblioteca pandas
import pandas as pd

# Dados para o DataFrame
dados = {
    'Coluna1': ['Valor1', 'Valor2', 'Valor3'],
    'Coluna2': ['Valor4', 'Valor5', 'Valor6'],
    'Coluna3': ['Valor7', 'Valor8', 'Valor9']
}

# Criando o DataFrame
df = pd.DataFrame(dados)

# Exibindo o DataFrame
print(df)

# Adicionando um novo dado ao DataFrame existente
df.loc[len(df.index)] = ['NovoValor1', 'NovoValor2', 'NovoValor3']

# Exibindo o DataFrame atualizado
print(df)



