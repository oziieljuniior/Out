import pyscreenshot as ImageGrab
from PIL import Image
from datetime import datetime
import time
import pyautogui

#(997, 353)
#(521, 203)
i = 0

while True:
    print("*-"*24)
    print(f'Realizar o print das odds, entrada {i}')
    #print da tela
    imagem = ImageGrab.grab()
    #salvar imagem com um nome
    imagem.save('/home/darkcover/Documentos/Out/python_project/Falling/Variacao3_LagAjustado/variacao_data_ajuste/Carolina/tela.jpeg','jpeg')
    #abrir imagem com a biblioteca Pil, para realizar corte
    img = Image.open('/home/darkcover/Documentos/Out/python_project/Falling/Variacao3_LagAjustado/variacao_data_ajuste/Carolina/tela.jpeg')
    #area de corte pretendida
    area0 = (525, 254, 999, 530) #area com numeros 
    #comando para cortar area da imagem 
    corte = img.crop(area0)

    # Obter a data e o horário atual
    data_horario_atual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    name = '/home/darkcover/Documentos/Out/python_project/Falling/Variacao3_LagAjustado/variacao_data_ajuste/Carolina/IMG/' + str(i) + '_' + str(data_horario_atual) + '.jpeg'


    #salvamento e carregamento da nova imagem de corte para imagem de 3 digitos
    corte.save(name,'jpeg')

    i += 1

    print(f'Odds capturadas ...')
    j = 1
    while j <= 6:
        time.sleep(50)
        pyautogui.click((637, 252), interval=1)
        time.sleep(10)
        pyautogui.click((871, 252), interval=1)


        j += 1