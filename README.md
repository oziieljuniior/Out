![](Imagens/Logo/logo1.jpg "Out")
---

# Issues
* No código codec1_1.py, adicionar array de contagem de 1 geral. [LOCAL](https://github.com/oziieljuniior/Out/blob/main/python_project/Atual/DRoger/Kardec/codec1.1.py) 
* Testar e atualizar datas. [LOCAL](https://github.com/oziieljuniior/Out/blob/main/Documentos/dados/Saidas/FUNCOES/DOUBLE%20-%2017_09_s1.csv)
* No código codec1_2.py, ajustar predição para um array. [LOCAL](https://github.com/oziieljuniior/Out/blob/main/python_project/Atual/DRoger/Kardec/codec1_2.py) 

---

# Blog News[130425]

## Informações uteis
* []()
* [Manual de Instalação Warp](https://github.com/oziieljuniior/Out/blob/main/Blog/infoutils3.md)
* [Trabalhando com Ambientes pythons - Avançado](https://github.com/oziieljuniior/Out/blob/main/Blog/infoutils2_3.md)
* [Trabalhando com Ambientes pythons - Intermediário - Parte 2](https://github.com/oziieljuniior/Out/blob/main/Blog/infoutils2_2.md)
* [Manual de uso Pyenv](https://github.com/oziieljuniior/Out/blob/main/Blog/infoutils2_1.md)
* [Trabalhando com Ambientes pythons - Intermediário - Parte 1](https://github.com/oziieljuniior/Out/blob/main/Blog/infoutils2.md)
* [Salvar todas as saídas do terminal](https://github.com/oziieljuniior/Out/blob/main/Blog/infoutils1.md)

---

# OUT

[PROJETO EM DESENVOLVIMENTO](https://github.com/oziieljuniior/Out/blob/main/python_project/Atual/DRoger/Kardec/codec1_1.py)

Análise e predição de eventos binários desbalanceados com janelas temporais e arquitetura customizada.  
Inclui o uso de lógica fuzzy para interpretar padrões em dados temporais e avaliação de aleatoriedade e compressibilidade em sequências.

---

## 📦 Tecnologias & Dependências Principais

- **Python**: 3.10 (gerenciado com `pyenv`)
- **Poetry**: gerenciamento de ambiente e dependências
- **TensorFlow**: `2.15.0`
- **TensorFlow Addons**: `0.23.0` (⚠️ em modo EOL até maio de 2024)
- **Bibliotecas auxiliares**:
  - NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, Scikit-Fuzzy, entre outras (instaladas conforme necessidade)

---

## ⚠️ Aviso importante

> O projeto **depende de funcionalidades específicas do TensorFlow Addons**, que foi descontinuado e entra em fim de vida (EOL) em **maio de 2024**.  
> A manutenção do ambiente depende da permanência em versões compatíveis do TensorFlow (até 2.15) e Python (até 3.11).  
> Futuras versões de TensorFlow e TFA não são garantidas como compatíveis.

---

## 🛠️ Ambiente recomendado

- **Python**: `3.10.x`
- **Poetry**: `>=2.1.0`
- **TensorFlow**: `2.15.0`
- **TensorFlow Addons**: `0.23.0`

Instalação típica com Poetry:

```bash
pyenv install 3.10.17
pyenv virtualenv 3.10.17 out-py310
pyenv local 3.10.17

poetry env use $(pyenv which python)
poetry install


## Como o código funciona

Informações como o jogo funciona, acesse:

[Leia sobre o jogo](https://github.com/oziieljuniior/Out/blob/main/Documentos/notes/sobre_jogo.md)

[Coisas Passadas](https://github.com/oziieljuniior/Out/blob/main/Documentos/notes/CoisasP.md)

[Últimas Atualizações](https://github.com/oziieljuniior/Out/blob/Documentos/main/notes/update_27_07.md)

[Ideias Futuras](https://github.com/oziieljuniior/Out/tree/main/python_project/Atual/DRoger)

