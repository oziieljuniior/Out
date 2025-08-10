import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# dicionário LaTeX das expressões (insira seu expressoes_latex aqui)
expressoes_latex = {
    1:  r'$\frac{4 \times 3}{6 \times 2}$',
    2:  r'$\frac{10}{\frac{55}{11}}$',
    3:  r'$\frac{9}{3}$',
    4:  r'$2 \times 2$',
    5:  r'$\frac{50}{10}$',
    6:  r'$2^2 + 2$',
    7:  r'$16 - 9$',
    8:  r'$2^3$',
    9:  r'$3^2$',
    10: r'$5 \times \left(\frac{8}{4}\right)$',
    11: r'$\frac{30}{3} + 1$',
    12: r'$3 \times 2 \times 2$',
    13: r'$3 + 2 \times 5$',
    14: r'$6 \times 2 + 2$',
    15: r'$8 \times 3 - 9$',
    16: r'$7 \times 3 - 5$',
    17: r'$11 \times 2 - 4$',
    18: r'$3 \times 3 \times 2$',
    19: r'$13 + 6$',
    20: r'$5 \times 4$',
    21: r'$3 \times 7$',
    22: r'$(5 + 6) \times 2$',
    23: r'$4^2 + 7$',
    24: r'$3 \times 2 \times 4$',
    25: r'$(14 - 9) \times 5$',
    26: r'$4^2 + 10$',
    27: r'$13 + (7 \times 2)$',
    28: r'$7 \times 6 - (9 + 5)$',
    29: r'$5 \times 3 + 14$',
    30: r'$\frac{60}{2}$',
    31: r'$8 \times 4 - 1$',
    32: r'$12 \times 3 - 4$',
    33: r'$11 \times 3$',
    34: r'$17 \times 2$',
    35: r'$(4 + 3) \times 5$',
    36: r'$6^2$',
    37: r'$5 \times 4 + 17$',
    38: r'$19 \times 2$',
    39: r'$13 \times 3$',
    40: r'$\frac{20}{4} \times 8$',
    41: r'$6 \times 5 + 11$',
    42: r'$50 - \left(\frac{16}{2}\right)$',
    43: r'$21 + 11 \times 2$',
    44: r'$\frac{55}{5} \times 4$',
    45: r'$(4 + 1) \times 9$',
    46: r'$(13 + 10) \times 2$',
    47: r'$3^3 + 5 \times 4$',
    48: r'$12 \times 2^2$',
    49: r'$13 \times 3 + 10$',
    50: r'$5 \times 9 + 5$',
}


def gerar_cartelas(qtd_cartelas, output_dir="cartelas_bingo"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "cartelas.pdf")
    pdf = PdfPages(pdf_path)

    for n in range(1, qtd_cartelas + 1):
        # sorteio de 9 expressões únicas
        expressoes_sorteadas = random.sample(list(expressoes_latex.values()), 9)
        matriz = [expressoes_sorteadas[i:i+3] for i in range(0, 9, 3)]

        # gerar figura
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")

        # plotar linhas da grade
        for i in range(4):
            ax.plot([0, 3], [i, i], color="black", linewidth=2)
            ax.plot([i, i], [0, 3], color="black", linewidth=2)

        # preencher as células
        for i, linha in enumerate(matriz):
            for j, expr in enumerate(linha):
                ax.text(j + 0.5, 2.5 - i, expr, ha="center", va="center", fontsize=16)

        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_title(f"Cartela {n}", fontsize=18)

        # salvar como imagem
        img_path = os.path.join(output_dir, f"cartela_{n:02d}.png")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")

        # adicionar ao PDF
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"✅ {qtd_cartelas} cartelas geradas em {output_dir}/cartelas.pdf e em PNG separadas.")

# exemplo de uso:
gerar_cartelas(qtd_cartelas=20)
