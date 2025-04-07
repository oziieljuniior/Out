import pyautogui
import pyperclip
import time
import pandas as pd
from PIL import Image
import pyscreenshot as ImageGrab
import os
import shutil

#funções
def Cores(area):
    ImageGrab.grab().save("img1.png")
    img = Image.open("img1.png")
    img_corte = img.crop(area)
    img_corte.save('foto01.jpeg','jpeg')
    img1 = Image.open('foto01.jpeg')
    convert_img1 = img1.convert('RGB').getcolors()
    print(convert_img1)
    return convert_img1


#E:\Python_Project\CNFB3.xlsx
data = pd.read_excel('E:\Python_Project\CNFB3.xlsx')
consulta = data['TICKER'].to_list()
t1 = len(consulta)
#RRP3 ~ não consta]
text = 'Formulário de Referência'
text1 = 'Comentários dos diretores'
text2 = 'Gerenciamento de riscos e controles internos'
text3 = 'Todos'

#Contar o valor inicial de (2,t1). Isso é determinado pela quantidade de arquivos que está na pasta Data\Empresas
for emp in range(57,t1):
    #E:\Python_Project\Data\Empresas
    path = 'E:\Python_Project\Data\Empresas\ ' + consulta[emp]
    print(consulta[emp])
    os.makedirs(path)
    
    i = 0
    while i == 0:
        print("Fase 1")
        area1 = (1027, 21, 1028, 22)
        area2 = (1000, 64, 1001, 65)
        
        convert_im1 = Cores(area1)
        convert_im2 = Cores(area2)

        if convert_im1 == [(1, (211, 227, 253))] and convert_im2 == [(1, (237, 241, 250))]:
            print("Fase 2 \nEscrita URL")
            j = 0
            pyautogui.hotkey('ctrl','t')
            time.sleep(3)
            #https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/empresas-listadas.htm
            pyautogui.write('https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/empresas-listadas.htm', interval = 0.10)
            pyautogui.press('enter')
            time.sleep(10)
            while j == 0:
                print("Fase 3")
                #[(1, (1, 44, 99))] [(1, (247, 247, 247))]
                area3 = (575, 775, 576, 776)
                area4 = (1245, 775, 1246, 776)
                convert_im3 = Cores(area3)
                convert_im4 = Cores(area4)
                
                #[(1, (88, 204, 241))] [(1, (255, 255, 255))]
                #[(1, (255, 255, 255))] [(1, (88, 204, 241))]
                if convert_im3 == [(1, (255, 255, 255))] and convert_im4 == [(1, (88, 204, 241))]:
                    print("Fase 4 \nEscrita da empresa para pesquisa")
                    k = 0
                    time.sleep(10)
                    pyautogui.doubleClick(455,775)
                    pyautogui.write(consulta[emp], interval = 0.5)
                    pyautogui.press('enter')
                    while k == 0:
                        ##Adicionar um método aonde o código pule caso não exista empresa.
                        print("Fase 5")
                        #[(1, (1, 44, 99))] [(1, (247, 247, 247))]
                        area5 = (770, 360, 771, 361)
                        area6 = (455, 775, 456, 776)
                        convert_im5 = Cores(area5)
                        convert_im6 = Cores(area6)
                        
                        if convert_im5 == [(1, (1, 44, 99))] and convert_im6 == [(1, (1, 176, 231))]:
                            time.sleep(3)
                            l = 0
                            print("Fase 6 \nClick em Relatórios Estruturados")
                            time.sleep(10)
                            pyautogui.doubleClick(455,775)
                            #possivel colocar outra captura
                            time.sleep(3)
                            pyautogui.click(1630,575)
                            time.sleep(3)
                            pyautogui.click(1630,650)
                            while l == 0:
                                print("Fase 7")
                                #[(1, (1, 44, 99))] [(1, (247, 247, 247))]
                                #[(1, (1, 44, 99))] [(1, (247, 247, 247))]
                                area7 = (1200,310,1201,311)
                                area8 = (300,680,301,681)
                                convert_im7 = Cores(area7)
                                convert_im8 = Cores(area8)
                                if convert_im7 == [(1, (1, 44, 99))] and convert_im8 == [(1, (255, 255, 255))]:
                                    print("Fase 8 \nPesquisa das empresas por ano")
                                    time.sleep(10)
                                    #
                                    for ano in range(2010,2023):
                                        print('Ano - ', ano)
                                        pyautogui.click((300,680), interval=1)
                                        time.sleep(1)
                                        pyautogui.write(str(ano), interval = 0.5)
                                        pyautogui.press('enter')    
                                        time.sleep(2)
                                        
                                        pyautogui.hotkey('ctrl','f')
                                        pyperclip.copy(text)
                                        pyautogui.hotkey('ctrl', 'v')
                                        pyautogui.press('enter')
                                        n = 0
                                        while n == 0:
                                            #[(1, (1, 44, 99))] [(1, (247, 255, 255))]
                                            area9 = (358, 512, 359, 513)
                                            area10 = (405, 574, 406, 575)
                                            #[(1, (236, 76, 44))] [(1, (232, 85, 33))]
                                            #[(1, (71, 209, 124))] [(1, (54, 213, 121))]
                                            #[(1, (255, 149, 51))] [(1, (255, 255, 0))]
                                            convert_im9 = Cores(area9)
                                            convert_im10 = Cores(area10)
                                            if (convert_im9 == [(1, (255, 149, 51))] and convert_im10 == [(1, (255, 255, 0))]) or (convert_im9 == [(1, (254, 151, 46))] and convert_im10 == [(1, (250, 255, 3))]):
                                                print("encontrei ouro")
                                                time.sleep(5)
                                                    
                                                pyautogui.click((405, 574), interval = 1)
                                                
                                                time.sleep(20)
                                                a1 = 0
                                                while a1 == 0:
                                                    ##adicionar metodo de espera de carregamento do site
                                                    if pyautogui.locateOnScreen('E:\\Python_Project\\Códigos\\Parte2\\Pictures\\save_pdf.png') != None:
                                                        pyautogui.click('E:\\Python_Project\\Códigos\\Parte2\\Pictures\\save_pdf.png', interval = 2)
                                                        a1 += 1
                                            
                                                time.sleep(20)
                                                a1 = 0
                                                while a1 == 0:
                                                    ##adicionar metodo de espera de carregamento do site
                                                    if pyautogui.locateOnScreen('E:\\Python_Project\\Códigos\\Parte2\\Pictures\\todos.png') != None:
                                                        pyautogui.click('E:\\Python_Project\\Códigos\\Parte2\\Pictures\\todos.png', interval = 2)
                                                        a1 += 1
                                                time.sleep(2)
                                                
                                                pyautogui.hotkey('ctrl','f')
                                                pyperclip.copy(text1)
                                                pyautogui.hotkey('ctrl', 'v')
                                                pyautogui.press('enter')

                                                time.sleep(2)
                                                a2 = 0
                                                while a2 <= 5:
                                                    if pyautogui.locateOnScreen('E:\\Python_Project\\Códigos\\Parte2\\Pictures\\comentarios_dir.png') != None:
                                                        pyautogui.click('E:\\Python_Project\\Códigos\\Parte2\\Pictures\\comentarios_dir.png', interval = 2)
                                                        a2 = 6
                                                    time.sleep(2.5)
                                                    a2 += 1

                                                pyautogui.hotkey('ctrl','f')
                                                pyperclip.copy(text2)
                                                pyautogui.hotkey('ctrl', 'v')
                                                pyautogui.press('enter')
                                                time.sleep(2)

                                                pyautogui.click((808, 651), interval=0.5)
                                                

                                                time.sleep(2)

                                                pyautogui.click((883, 971), interval=0.5)
                                                ##Adicionar metodo de espera de carregamento de janela
                                                time.sleep(5)
                                                a2 = 0
                                                while a2 == 0:
                                                    ##adicionar metodo de espera de carregamento do site
                                                    area12 = (416, 411, 417, 412)
                                                    convert_im12 = Cores(area12)
                                                    if convert_im12 == [(1, (240, 240, 240))]:
                                                        print("click on the bellow part 2")
                                                        pyautogui.click((1785, 180), interval = 1)
                                                        a2 = 1 
                                                
                                                text3 = consulta[emp] + "_" + str(ano)
                                                pyperclip.copy(text3)
                                                pyautogui.hotkey('ctrl', 'v')
                                                time.sleep(1)
                                                pyautogui.press('enter')
                                                time.sleep(5)

                                                pyautogui.hotkey('ctrl', 'w')
                                                pyautogui.press('pageup')
                                                pyautogui.press('pageup')
                                                pyautogui.press('pageup')
                                                
                                                n = n + 1
                                            else:
                                                time.sleep(3)
                                                pyautogui.press('pageup')
                                                pyautogui.press('pageup')
                                                pyautogui.press('pageup')
                                                n += 1
                                                time.sleep(3)

                                        #/html/body/app-root/app-companies-menu-select/div/app-companies-structured-reports/form/div[2]/div/div[2]/div/div[1]/div[1]/div[2]/p/a
                                        #/html/body/app-root/app-companies-menu-select/div/app-companies-structured-reports/form/div[2]/div/div[2]/div/div[1]/div[1]/div[2]/p/a
                                               
                                                                            
                                print("Conversão realizada")
                                pyautogui.hotkey('ctrl','w')
                                p = 0
                                while p == 0:
                                    path1 = os.listdir("E:\\Python_Project\\Data\\Downloads")
                                    for arquivo in path1:
                                        source = "E:\\Python_Project\\Data\\Downloads\\" + arquivo
                                        shutil.move(source, path)
                                    p = p + 1
                                i += 1
                                j += 1
                                k += 1
                                l += 1
                                
                                
        
