import tkinter as tk
from tkinter import filedialog

class FileSelector:
    def __init__(self):
        """Inicializa o seletor de arquivos."""
        self.file_path = None
    
    def open_file_dialog(self):
        """Abre uma janela para selecionar um arquivo e salva o caminho escolhido."""
        root = tk.Tk()
        root.withdraw()  # Oculta a janela principal
        self.file_path = filedialog.askopenfilename(title="Selecione um arquivo")
        return self.file_path
    
    def save_selected_path(self, save_path="/home/ozielramos/Documentos/Out/python_project/cake/Oz/Caminhos/selected_file.txt"):
        """Salva o caminho do arquivo selecionado em um arquivo de texto."""
        if self.file_path:
            with open(save_path, "w") as f:
                f.write(self.file_path)
            print(f"Caminho salvo em {save_path}")
        else:
            print("Nenhum arquivo foi selecionado.")

if __name__ == "__main__":
    selector = FileSelector()
    file_path = selector.open_file_dialog()
    if file_path:
        print(f"Arquivo selecionado: {file_path}")
        selector.save_selected_path()
