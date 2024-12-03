import pandas as pd 
import time

##PRODUCAO##

data = pd.read_csv('/home/darkcover1/Documentos/Out/dados/Saidas/FUNCOES/DOUBLE - 17_09_s1.csv')

############


array1, array2, array3 = [], [], []
i, i0, i1 = 0, 0, 0
contador = False
inteiro = int(input("Insera a entrada até onde o modelo deve ser carregado --> "))

while i <= 200000:
    print(24*'***')
    print(f'Número da Entrada - {i}')
    if i <= inteiro:
        odd = float(data['Entrada'][i].replace(",",'.'))
        print(f'Entrada -> {odd}')
    else:
        odd = float(input("Entrada -> ").replace(",",'.'))

    if contador is True:
        if odd >= 2:
            i0 += 1  

    if odd == 0:
        break
    if odd == -1:
        if contador == False:
            contador = True
            i0 = 0
            i1 = 0
            odd = float(input("Entrada -> ").replace(",",'.'))
            if odd >= 2:
                i0 += 1
            i1 += 1
        else:
            contador = False
            i0 = 0
            i1 = 0
            odd = float(input("Entrada -> ").replace(",",'.'))
    #Array1 responsavel por guardar as últimas entradas
    array1.append(odd)

    if odd >= 2:
        att1 = 1
    else:
        att1 = 0
        
    #Array2 responsavel por guardar as entradas 0 ou 1
    array2.append(att1)


    ## Visualização
    if i >= 60 and contador == False:
        print(18*"---")
        print("Contador de apostas -> OFF")
        array3 = array2[-60:]
        
        media = sum(array3)/60

        print(f'Media -> {media} \nPrimeiro Elemento -> {array3[0]} \nQt. de 1s -> {sum(array3)} \nQt. de 0s -> {60 - sum(array3)}')

        print(18*"---")

    if i >= 60 and contador == True:
        print(18*"---")
        print("Contador de apostas -> ON")
        array3 = array2[-60:]
        
        media = sum(array3)/60

        print(f'Media -> {media} \nPrimeiro Elemento -> {array3[0]} \nQt. de 1s -> {sum(array3)} \nQt. de 0s -> {60 - sum(array3)}')

        media_pontual = i0 / i1 
        print(f'Acuracia -> {media_pontual}')

        i1 += 1
        
        print(18*"---")
    ################
    i += 1



