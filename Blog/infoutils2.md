Boa! Você tá no caminho certo de pensar em ambientes portáveis e consistentes. Isso que você tá passando é bem comum: versões diferentes do Python, bibliotecas que quebram, dependências específicas... Mas dá pra resolver com uma estratégia bem feita. Vamos montar uma solução ideal pro seu caso, considerando:

✅ **Dois computadores Ubuntu**  
✅ **Problemas com versões de bibliotecas e do próprio Python**  
✅ **Quer algo que funcione de forma reprodutível nos dois**  

---

## 🧩 SOLUÇÃO IDEAL PRA VOCÊ: `pyenv` + `poetry`

Essa combinação resolve tanto:
- ✔️ **Controle da versão do Python**
- ✔️ **Ambientes virtuais isolados**
- ✔️ **Dependências versionadas e reprodutíveis**
- ✔️ **Portabilidade entre máquinas**

---

### 🔧 1. Instalar `pyenv` (gerenciar múltiplas versões do Python)

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash
```

Depois, adicione essas linhas no final do seu `~/.bashrc` ou `~/.zshrc`:

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

E então reinicie o terminal ou rode:
```bash
source ~/.bashrc  # ou ~/.zshrc
```

#### Instalar uma versão específica do Python:
```bash
pyenv install 3.11.6
pyenv global 3.11.6
```

> Com isso, você consegue ter exatamente a **mesma versão do Python** nas duas máquinas.

---

### 🧪 2. Instalar `poetry` (para dependências e ambiente virtual)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Adicione ao seu `~/.bashrc` ou `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Verifique se instalou:
```bash
poetry --version
```

---

### 🚀 3. Criar seu projeto Python com Poetry

```bash
poetry new meu_projeto
cd meu_projeto
poetry env use $(pyenv which python)  # Garante que usa a versão do pyenv
poetry add requests pandas # ou o que precisar
```

Isso cria:
- Um ambiente virtual isolado
- Um `pyproject.toml` com todas as dependências
- Um `poetry.lock` com versões fixas

---

### 🔁 4. Usar no segundo computador

No outro Ubuntu:
1. Instale `pyenv` e `poetry` igualzinho
2. Clone o repositório do projeto
3. Rode:

```bash
cd meu_projeto
pyenv install 3.11.6  # mesma versão
poetry install
poetry shell
```

✨ Pronto! Agora você tem:
- Mesmo Python
- Mesmas dependências
- Mesmo ambiente
- Sem treta com caminhos ou libs quebradas

---

Se quiser, posso te montar um script `.sh` pra automatizar isso em qualquer máquina. Curte essa ideia?