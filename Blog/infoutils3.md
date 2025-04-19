Claro! Aqui vai um **manual básico do Warp** — uma ferramenta que moderniza o terminal para desenvolvedores. Warp é um terminal reimaginado, com foco em produtividade, colaboração e experiência de usuário moderna.

---

## 🌀 **Manual Básico do Warp**

---

### 🔧 **1. O que é o Warp?**

Warp é um terminal moderno para desenvolvedores. Ele combina:

- Interface gráfica moderna (GUI e TUI ao mesmo tempo)
- Blocos de comandos com histórico visual
- Editor de texto embutido no terminal
- Colaboração e compartilhamento de sessões
- Inteligência artificial integrada (Warp AI)

> Funciona **nativamente no macOS** e, atualmente, também no **Linux (beta)** e **Windows (via WSL)**.

---

### 🚀 **2. Como instalar o Warp**

#### **macOS:**
```bash
brew install --cask warp
```

Ou baixe diretamente pelo site:  
🔗 [https://warp.dev/download](https://warp.dev/download)

#### **Linux (Ubuntu/Debian):**
```bash
curl https://releases.warp.dev/linux/warp-latest.deb --output warp.deb
sudo apt install ./warp.deb
```

> Para Arch Linux: use o AUR (`yay -S warp-terminal`)

#### **Windows:**
Atualmente só funciona com **WSL + X11**, ou via cliente SSH para máquinas remotas com Warp.

---

### ⚙️ **3. Funcionalidades principais**

#### 🧱 **Blocos de Comandos**
Cada comando executado vira um "bloco". Isso facilita:

- Ver o que você rodou.
- Compartilhar blocos.
- Rodar apenas aquele bloco novamente.

#### ✍️ **Editor embutido**
Você edita os comandos como se estivesse num editor de texto real (tipo VSCode), com:

- Autocompletar
- Multilinhas
- Syntax highlight

#### 🤖 **Warp AI**
Atalho: `⌘ + i`  
Você pode perguntar algo diretamente ao terminal com IA integrada (em inglês), como:

> "How to use rsync to copy a directory excluding some files?"

#### 📑 **Histórico e Navegação**
Use `Ctrl + R` para navegar visualmente pelos comandos anteriores — bem mais amigável que o terminal tradicional.

#### ⌨️ **Atalhos úteis**

| Atalho             | Ação                                  |
|--------------------|---------------------------------------|
| `Cmd + K`          | Limpa a tela                          |
| `Cmd + P`          | Busca comandos anteriores (como fuzzy search) |
| `Cmd + Shift + D`  | Divide o terminal em painéis          |
| `Cmd + Enter`      | Executa o comando                     |
| `Cmd + I`          | Warp AI                               |

---

### 🧩 **4. Personalizações**

Você pode alterar temas, cores e atalhos acessando o menu com `Cmd + ,` (Configurações).

> Warp já vem com suporte a temas escuros, modo vi, e atalhos personalizados.

---

### 🌐 **5. Integração com Git e SSH**

Warp detecta repositórios Git automaticamente e fornece:

- Status visual dos arquivos
- Auto-complete para comandos Git
- Branch picker interativo

Você também pode salvar conexões SSH facilmente.

---

### 🧪 **6. Dicas extras**

- Use o **Warp Drive** para salvar comandos e snippets reutilizáveis.
- Comandos podem ser comentados e compartilhados com equipe.
- Integração com **VS Code**, **zsh**, **bash**, etc.

---

### 📘 **7. Documentação oficial**

👉 [https://docs.warp.dev](https://docs.warp.dev)

---

Se quiser, posso montar um cheat sheet em PDF com os atalhos e comandos principais do Warp. Você quer?