Excelente pergunta, Oziel. Se você **já tem um projeto em andamento** usando só o `pyenv` (sem `poetry` ainda) e quer começar a usar o `poetry` **sem perder nada**, é super possível e tranquilo de fazer. Vamos passo a passo:

---

# 🧪 Adicionando `poetry` em um projeto existente com `pyenv`

---

## 🗂️ Supondo:

- Você já está na pasta do seu projeto: `~/meu_projeto`
- Já está usando o `pyenv` com uma versão definida (por `pyenv local 3.x.x`)
- Quer agora usar o `poetry` pra gerenciar dependências e ambiente virtual

---

## ✅ Passo 1: Instalar o `poetry` (se ainda não tiver)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Depois adicione ao `~/.bashrc` (ou `~/.zshrc`):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

E aplique:

```bash
source ~/.bashrc
```

Verifique se funcionou:

```bash
poetry --version
```

---

## ✅ Passo 2: Inicializar o `poetry` no seu projeto existente

No diretório do seu projeto, rode:

```bash
poetry init
```

Ele vai te perguntar sobre nome do projeto, versão, autor, etc.

💡 Se quiser só aceitar os padrões e ir adicionando dependências depois, pode usar a flag:

```bash
poetry init --no-interaction
```

Isso vai criar o arquivo `pyproject.toml`.

---

## ✅ Passo 3: Dizer ao `poetry` para usar o Python do `pyenv`

Esse passo é crucial pra garantir que o `poetry` use a mesma versão do Python já configurada pelo `pyenv`.

No seu projeto:

```bash
poetry env use $(pyenv which python)
```

Ele vai criar um novo ambiente virtual baseado **na versão do pyenv**, dentro de `.cache` ou `.venv` (dependendo da config).

---

## ✅ Passo 4: Instalar as dependências do projeto (se já tiver `requirements.txt`)

Se você tem um arquivo `requirements.txt`, converta pro `pyproject.toml` assim:

```bash
poetry add $(cat requirements.txt)
```

Ou adicione manualmente com:

```bash
poetry add requests pandas flask  # exemplo
```

---

## ✅ Passo 5: Usar o ambiente com `poetry`

Ativar o ambiente:

```bash
poetry shell
```

Rodar scripts dentro dele:

```bash
poetry run python script.py
```

---

## 🧹 Dica extra: Ignorar o virtualenv local (se preferir usar um `.venv` na raiz)

Se você quiser que o virtualenv seja criado na pasta do projeto (em `.venv`), configure:

```bash
poetry config virtualenvs.in-project true
```

Assim fica mais fácil mover/copiar com o projeto, e o ambiente fica todo junto.

---

## ✅ Resumo rápido:

```bash
cd ~/meu_projeto
poetry init
poetry env use $(pyenv which python)
poetry add -D black flake8  # ou o que precisar
poetry shell
```

---

Se quiser, posso gerar um `pyproject.toml` de exemplo baseado no seu `requirements.txt`. Só me manda ou cola ele aqui!