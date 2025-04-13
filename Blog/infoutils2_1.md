Perfeito, Oziel! Aqui está um **manual prático e direto de uso do `pyenv`**, especialmente pensado pra você que está em ambiente Ubuntu e quer controlar versões de Python de forma consistente entre máquinas.

---

# 📘 Manual de Uso do `pyenv` (Ubuntu/Linux)

## 📌 O que é o `pyenv`?

O `pyenv` é uma ferramenta que permite instalar e alternar entre várias versões do Python facilmente, sem depender da versão instalada pelo sistema.

---

## ⚙️ 1. Instalação do `pyenv`

### 🔧 Requisitos:
Rode isso primeiro para garantir que você tem as bibliotecas necessárias:

```bash
sudo apt update && sudo apt install -y \
make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev
```

### ⬇️ Instalar o `pyenv`:

```bash
curl https://pyenv.run | bash
```

### 🧠 Adicionar no seu shell (para carregar o `pyenv` automaticamente):

Edite seu `~/.bashrc` (ou `~/.zshrc` se usar zsh):

```bash
nano ~/.bashrc
```

E adicione ao final do arquivo:

```bash
# pyenv
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Salve e saia com `CTRL+O`, `ENTER`, `CTRL+X`.

Depois, aplique:

```bash
source ~/.bashrc
```

---

## 🐍 2. Usando o `pyenv`

### 🔍 Verificar se está funcionando:

```bash
pyenv --version
```

---

### 📥 Instalar uma versão do Python:

```bash
pyenv install 3.11.6
```

Veja todas as versões disponíveis:

```bash
pyenv install --list
```

---

### 🗺️ Ver todas as versões instaladas localmente:

```bash
pyenv versions
```

---

### 🌐 Definir versão global (usada em todo o sistema):

```bash
pyenv global 3.11.6
```

---

### 📂 Definir versão local (usada só na pasta do projeto):

```bash
cd meu_projeto/
pyenv local 3.11.6
```

Isso cria um arquivo `.python-version` na pasta.

---

### 📌 Ver versão do Python em uso:

```bash
python --version
```

---

## 🧪 Integração com `poetry` (opcional e recomendado)

Depois de instalar o `poetry`:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Configure o poetry pra usar a versão do `pyenv`:

```bash
poetry env use $(pyenv which python)
```

Agora qualquer `poetry add`, `poetry run`, `poetry shell` vai usar o Python controlado pelo `pyenv`.

---

## 🧹 Desinstalar uma versão do Python

```bash
pyenv uninstall 3.11.6
```

---

## ❓ Comandos Úteis

| Comando                        | Descrição                                 |
|-------------------------------|-------------------------------------------|
| `pyenv install 3.x.x`         | Instala uma versão do Python              |
| `pyenv uninstall 3.x.x`       | Remove uma versão                         |
| `pyenv versions`              | Lista as versões instaladas               |
| `pyenv global 3.x.x`          | Define a versão padrão no sistema         |
| `pyenv local 3.x.x`           | Define a versão específica no projeto     |
| `pyenv shell 3.x.x`           | Define a versão só no terminal atual      |
| `pyenv which python`          | Caminho exato da versão usada             |
| `pyenv install --list`        | Lista todas as versões disponíveis        |

---

## 📦 Dica extra: `pyenv-virtualenv` (ambientes isolados)

Você pode criar ambientes virtuais diretamente com pyenv:

```bash
pyenv virtualenv 3.11.6 meu-env
pyenv activate meu-env
```

Pra desativar:
```bash
pyenv deactivate
```

---

Se quiser, posso gerar uma **versão em PDF ou Markdown** disso tudo pra você guardar ou usar nos dois PCs. Quer?