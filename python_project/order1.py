#Import de bibliotecas
import pandas as pd
import numpy as np

#Entrada i_k, consta como 1 pois ela controla nossas entradas. Ele deve ser um interio determinado do intervalo 0 ~ 'A verificar'
i = 1
#Lista de entradas realiza o controla das entradas. Toda vez que uma nova entrada i_k é gerada, o i_k é salvo na lista de entrada junto com seu histórico de entradas anteriores.
lista_entradas = []

#Controladores do jogo.
apostar = 0
contagem = 0
level = 1

#Informação inicial que deve ser salva em uma planilha, considera-se como:
#Rodada ~ Em que entrada o jogo está; level ~ Em qual level o jogador está; apostar ~ condição para apostar(0 - Não apostar, 1 - apostar)
#Acerto ~ Houve acerto ao apostar (0 ~ não houve aposta, 1 ~ houve aposta e acerto, 2 - houve aposta e erro)
#contagem ~ contagem de level, se a contagem chegar a 15 o jogador sobe de level, caso a contagem chegue a -10 o jogo é perdido e level reiniciado
#i ~ entradas
#media5, media10, media20, media40, media80, media160, media320, media640 ~ medias das ultimas entradas. 
entrada_inicial = {
                    'Rodada': [0], 'level': [1], 'apostar': [0], 
                    'acerto': [0], 'contagem': [0], 'i': [0], 
                    'media5': [0], 'media10': [0], 'media20': [0], 
                    'media40': [0], 'media80': [0], 'media160': [0], 
                    'media320': [0], 'media640': [0]
                    }
data = pd.DataFrame(entrada_inicial)
print(data)

#Começamos o jogo aqui, com a primeira entrada gerada.
while i != 0:
    
    print(24*'*-')
             
    i = float(input("Insira a última entrada determinada: "))
    acerto = 0
#Ajustar o i_k de acordo com nossas entradas, ele deve obedecer o nosso intervalo pré-determinado.
    if apostar == 1 and i > 1.5:
        print(24*'$')
        acerto = 1
        contagem += 1
        if contagem == 15:
            level += 1
            contagem = 0
        print(f"Houve acerto! \nA ultima entrada determinada foi: {lista_entradas[-1]} \nA condição de vitoria está em {contagem} \nLevel: {level} \n{24*'*-'}")
    if apostar == 1 and i <= 1.5:
        print(24*'$')
        acerto = 2
        contagem = contagem - 2
        if contagem <= -10:
            level = 1
            contagem = 0 
            print(f"Houve erro! \nA ultima entrada determinada foi: {lista_entradas[-1]} \nA condição de vitoria está em {contagem}, dessa maneira GAME OVER. \n{24*'*-'}")
        else:
            print(f"Houve erro! \nA ultima entrada determinada foi: {lista_entradas[-1]} \nA condição de vitoria está em {contagem} \nLevel: {level} \n{24*'*-'}")
    
    if i == 0:
        break
    lista_entradas.append(i)

    if len(lista_entradas) < 6:
        data.loc[len(data.index)] = [len(lista_entradas), level, apostar, acerto, contagem, i, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        data.loc[len(data.index)] = [len(lista_entradas), level, apostar, acerto, contagem, i, media5, media10, media20, media40, media80, media160, media320, media640]
        apostar = 0

#A partir da quinta entrada, essa parte do código começa a fazer o acompanhamento das médias.
    if len(lista_entradas) >= 5:

        ultimas5 = lista_entradas[-5:]
        ultimas10 = [0] 
        ultimas20 = [0]
        ultimas40 = [0]
        ultimas80 = [0]
        ultimas160 = [0]
        ultimas320 = [0]
        ultimas640 = [0]

        if len(lista_entradas) >= 10:
            ultimas10 = lista_entradas[-10:]
            if len(lista_entradas) >= 20:
                ultimas20 = lista_entradas[-20:]
                if len(lista_entradas) >= 40:
                    ultimas40 = lista_entradas[-40:]
                    if len(lista_entradas) >= 80:
                        ultimas80 = lista_entradas[-80:]
                        if len(lista_entradas) >= 160:
                            ultimas160 = lista_entradas[-160:]
                            if len(lista_entradas) >= 320:
                                ultimas320 = lista_entradas[-320:]
                                if len(lista_entradas) >= 640:
                                    ultimas640 = lista_entradas[-640:]
                                
        media5 = sum(ultimas5) / 5
        media10 = sum(ultimas10) / 10
        media20 = sum(ultimas20) / 20
        media40 = sum(ultimas40) / 40
        media80 = sum(ultimas80) / 80
        media160 = sum(ultimas160) / 160
        media320 = sum(ultimas320) / 320
        media640 = sum(ultimas640) / 640
        
        print(f"Rodada: {len(lista_entradas)} \nMedia 5:  {media5} \nMedia 10: {media10} \nMedia 20: {media20} \nMedia 40: {media40} \nMedia 80: {media80} \nMedia 160: {media160} \nMedia 320: {media320} \nMedia 640: {media640}" )
#Esses intervalos de médias são pré-determidos. Podendo mudar de acordo com nossos estudos.
        if (media5 <= 2.5 and media5 != 0) or (media10 <= 3.5 and media10 != 0) or (media20 <= 4.5 and media20 != 0) or (media40 <= 4.8 and media40 != 0) or (media80 <= 5.44 and media80 != 0) or (media160 <= 5.47 and media160 != 0) or (media320 <= 5.718 and media320 != 0) or (media640 <= 5.889 and media640 != 0):
            apostar = 1
            print("APOSTAR NA PROXIMA RODADA")
        else:
            apostar = 0



print(lista_entradas)
print(data)

data.to_csv('/home/darkcover/Documentos/Out/dados/data_final.csv')