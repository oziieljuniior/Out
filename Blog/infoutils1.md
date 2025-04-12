Boa! Se você está rodando um script Python a partir do terminal (cmd/console) no Ubuntu e quer **salvar todas as saídas do terminal**, incluindo `print()`, logs de treinamento, etc., você pode fazer isso de algumas formas bem simples.

---

### ✅ **1. Redirecionando a saída ao executar o script**

Se você está rodando o script como:

```bash
python meu_script.py
```

Você pode redirecionar a saída para um arquivo com:

```bash
python meu_script.py > saida.txt 2>&1
```

- `>` redireciona a **saída padrão (stdout)** para o arquivo `saida.txt`.
- `2>&1` redireciona também a **saída de erro (stderr)** para o mesmo arquivo.

📌 Isso vai salvar **tudo o que apareceria no terminal** no arquivo `saida.txt`.

---

### ✅ **2. Usando `tee` para salvar e ver ao mesmo tempo**

Se você quer **ver a saída no terminal enquanto salva**:

```bash
python meu_script.py | tee saida.txt
```

Ou, para incluir erros também:

```bash
python meu_script.py 2>&1 | tee saida.txt
```

---

### ✅ **3. Dentro do script Python (opcional)**

Se você quiser capturar a saída diretamente dentro do script (por exemplo, salvar logs programaticamente), pode usar algo como:

```python
import sys
sys.stdout = open('saida.txt', 'w')
```

Coloque isso no início do seu script, e tudo que for `print()` vai para `saida.txt`. Mas atenção: isso **não salva logs do TensorFlow com `tqdm`**, por exemplo, então o método da linha de comando é mais confiável para logs de modelo.

---

Quer que eu adicione ao seu script Python um código que salva log personalizado em arquivo também?