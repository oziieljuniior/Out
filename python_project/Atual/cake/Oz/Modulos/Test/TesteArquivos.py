import unittest
from unittest.mock import patch, MagicMock
from tkinter import filedialog
import sys
sys.path.append("/home/ozielramos/Documentos/Out/python_project/cake/Oz/Modulos")

from Arquivos import FileSelector  # Importa a classe FileSelector do módulo Arquivos

class TestFileSelector(unittest.TestCase):
    @patch('tkinter.Tk')
    @patch('tkinter.filedialog.askopenfilename')
    def test_open_file_dialog(self, mock_askopenfilename, mock_tk):
        # Configura o mock para retornar um caminho fictício
        mock_askopenfilename.return_value = "/caminho/ficticio/arquivo.txt"
        
        selector = FileSelector()
        file_path = selector.open_file_dialog()
        
        # Verifica se o método retornou o caminho esperado
        self.assertEqual(file_path, "/caminho/ficticio/arquivo.txt")
        self.assertEqual(selector.file_path, "/caminho/ficticio/arquivo.txt")
        
        # Verifica se a janela foi criada e ocultada
        mock_tk.return_value.withdraw.assert_called_once()
        mock_askopenfilename.assert_called_once_with(title="Selecione um arquivo")

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('builtins.print')
    def test_save_selected_path(self, mock_print, mock_open):
        selector = FileSelector()
        
        # Caso 1: Arquivo selecionado (file_path definido)
        selector.file_path = "/caminho/salvo/arquivo.txt"
        selector.save_selected_path("/caminho/salvo/selected_file.txt")
        
        # Verifica se o arquivo foi escrito corretamente
        mock_open.assert_called_once_with("/caminho/salvo/selected_file.txt", "w")
        mock_open().write.assert_called_once_with("/caminho/salvo/arquivo.txt")
        mock_print.assert_called_once_with("Caminho salvo em /caminho/salvo/selected_file.txt")
        
        # Reseta os mocks para o próximo caso
        mock_open.reset_mock()
        mock_print.reset_mock()
        
        # Caso 2: Nenhum arquivo selecionado (file_path = None)
        selector.file_path = None
        selector.save_selected_path()
        
        # Verifica se a mensagem de erro foi exibida
        mock_print.assert_called_once_with("Nenhum arquivo foi selecionado.")
        mock_open.assert_not_called()  # Não deve tentar abrir o arquivo

if __name__ == "__main__":
    unittest.main()