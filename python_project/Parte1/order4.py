## Criar planilha de jogos ~ fase 2
#Import de bibliotecas
import pandas as pd
import numpy as np

data_inicial = pd.read_csv('/home/darkcover/Documentos/Out/dados/odds_200k.csv')
data_inicial = data_inicial.drop(columns=['Unnamed: 0'])
data_inicial = data_inicial.rename(columns={'Odd_Categoria': 'odd_saida'})
print("Data Carregada ...")

#Lista de entradas realiza o controla das entradas. Toda vez que uma nova entrada i_k é gerada, o i_k é salvo na lista de entrada junto com seu histórico de entradas anteriores.
lista_entradas = []
arraymedia5 = []
arraymedia10 = []
arraymedia20 = []
arraymedia40 = []
arraymedia80 = []
arraymedia160 = []
arraymedia320 = []
arraymedia640 = []

#Controladores do jogo.
apostar = 0
contagem = 0
level = 1

#Informação inicial que deve ser salva em uma planilha, considera-se como:
#Rodada ~ Em que entrada o jogo está; level ~ Em qual level o jogador está; apostar ~ condição para apostar(0 - Não apostar, 1 - apostar)
#Acerto ~ Houve acerto ao apostar (0 ~ não houve acerto, 1 ~ acerto)
#contagem ~ contagem de level, se a contagem chegar a 15 o jogador sobe de level, caso a contagem chegue a -10 o jogo é perdido e level reiniciado
#i ~ entradas
#media5, media10, media20, media40, media80, media160, media320, media640 ~ medias das ultimas entradas. 
entrada_inicial = {
                    'Rodada': [0], 'level': [1], 'apostar': [0], 
                    'acerto': [0], 'contagem': [0], 'odd':[0], 'odd_entrada': [0],
                    'odd_saida': [0], 'media5': [0], 'percentil5': [0], 'percentil5geral': [0], 'media10': [0], 'percentil10': [0], 'percentil10geral': [0], 'media20': [0], 'percentil20': [0], 'percentil20geral': [0], 
                    'media40': [0], 'percentil40': [0], 'percentil40geral': [0], 'media80': [0], 'percentil80': [0], 'percentil80geral': [0], 'media160': [0], 'percentil160': [0], 'percentil160geral': [0], 
                    'media320': [0], 'percentil320': [0], 'percentil320geral': [0], 'media640': [0], 'percentil640': [0], 'percentil640geral': [0]
                    }
data_final = pd.DataFrame(entrada_inicial)
print(data_final)

#Começamos o jogo aqui, com a primeira entrada gerada.
for (odd, odd_saida, odd_entrada) in zip(data_inicial['Odd'], data_inicial['odd_saida'], data_inicial['odd_entrada']):
    print("Entrada carregada ...")
    
    print(24*'*-')
             
    i = odd_saida
    acerto = 0
#Ajustar o i_k de acordo com nossas entradas, ele deve obedecer o nosso intervalo pré-determinado.
    if apostar == 1 and odd_saida >= 4:
        print(24*'$')
        acerto = 1
        contagem += 1
        if contagem == 15:
            level += 1
            contagem = 0
        print(f"Houve acerto! \nA ultima entrada determinada foi: {lista_entradas[-1]} \nA condição de vitoria está em {contagem} \nLevel: {level} \n{24*'*-'}")
    if apostar == 1 and odd_saida < 4:
        print(24*'$')
        acerto = 0
        contagem = contagem - 2
        if contagem <= -10:
            level = 1
            contagem = 0 
            print(f"Houve erro! \nA ultima entrada determinada foi: {lista_entradas[-1]} \nA condição de vitoria está em {contagem}, dessa maneira GAME OVER. \n{24*'*-'}")
        else:
            print(f"Houve erro! \nA ultima entrada determinada foi: {lista_entradas[-1]} \nA condição de vitoria está em {contagem} \nLevel: {level} \n{24*'*-'}")
    
    if apostar == 0 and odd_saida >= 4:
        acerto = 1
    if apostar == 0 and odd_saida < 4:
        acerto = 0

    lista_entradas.append(i)

    if len(lista_entradas) < 6:
        data_final.loc[len(data_final.index)] = [len(lista_entradas), level, apostar, acerto, contagem, odd, odd_entrada,odd_saida, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        data_final.loc[len(data_final.index)] = [len(lista_entradas), level, apostar, acerto, contagem, odd, odd_entrada, odd_saida, media5, percentil5, percentil5geral, media10, percentil10, percentil10geral, media20, percentil20, percentil20geral, media40, percentil40, percentil40geral, media80, percentil80, percentil80geral, media160, percentil160, percentil160geral, media320, percentil320, percentil320geral, media640, percentil640, percentil640geral]
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
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#                                
        media5 = sum(ultimas5) / 5
        arraymedia5.append(media5)
        percentil5 = np.percentile(ultimas5, 25)
        percentil5geral = np.percentile(arraymedia5, 25)
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media10 = sum(ultimas10) / 10
        arraymedia10.append(media10)
        percentil10 = np.percentile(ultimas10, 25)
        if len(lista_entradas) < 10:
            percentil10geral = 0
        else:
            percentil10geral = np.percentile(arraymedia10, 25)
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media20 = sum(ultimas20) / 20
        arraymedia20.append(media20)
        percentil20 = np.percentile(ultimas20, 25)
        if len(lista_entradas) < 20:
            percentil20geral = 0
        else:
            percentil20geral = np.percentile(arraymedia20, 25)
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media40 = sum(ultimas40) / 40
        arraymedia40.append(media40)
        percentil40 = np.percentile(ultimas40, 25)
        if len(lista_entradas) < 40:
            percentil40geral = 0
        else:
            percentil40geral = np.percentile(arraymedia40, 25)
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media80 = sum(ultimas80) / 80
        arraymedia80.append(media80)
        percentil80 = np.percentile(ultimas80, 25)
        if len(lista_entradas) < 80:
            percentil80geral = 0
        else:
            percentil80geral = np.percentile(arraymedia80, 25)
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media160 = sum(ultimas160) / 160
        arraymedia160.append(media160)
        percentil160 = np.percentile(ultimas160, 25)
        if len(lista_entradas) < 160:
            percentil160geral = 0
        else:
            percentil160geral = np.percentile(arraymedia160, 25)
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media320 = sum(ultimas320) / 320
        arraymedia320.append(media320)
        percentil320 = np.percentile(ultimas320, 25)
        if len(lista_entradas) < 320:
            percentil320geral = 0
        else:
            percentil320geral = np.percentile(arraymedia320, 25)
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media640 = sum(ultimas640) / 640
        arraymedia640.append(media640)
        percentil640 = np.percentile(ultimas640, 25)
        if len(lista_entradas) < 640:
            percentil640geral = 0
        else:
            percentil640geral = np.percentile(arraymedia640, 25)
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        
        print(f"Rodada: {len(lista_entradas)} \nMedia 5:  {media5} \nPercentil5: {percentil5} \nPercentil5Geral: {percentil5geral} \nMedia 10: {media10} \nPercentil10: {percentil10} \nPercentil10Geral: {percentil10geral} \nMedia 20: {media20} \nPercentil20: {percentil20} \nPercentil20Geral: {percentil20geral} \nMedia 40: {media40} \nPercentil40: {percentil40} \nPercentil40Geral: {percentil40geral} \nMedia 80: {media80} \nPercentil80: {percentil80} \nPercentil80Geral: {percentil80geral} \nMedia 160: {media160} \nPercentil160: {percentil160} \nPercentil160Geral: {percentil160geral} \nMedia 320: {media320} \nPercentil320: {percentil320} \nPercentil320Geral: {percentil320geral} \nMedia 640: {media640} \nPercentil640: {percentil640} \nPercentil640Geral: {percentil640geral}" )
#Esses intervalos de médias são pré-determidos. Podendo mudar de acordo com nossos estudos.
        if (media5 <= percentil5 and media5 != 0) or (media10 <= percentil10 and media10 != 0) or (media20 <= percentil20 and media20 != 0) or (media40 <= percentil40 and media40 != 0) or (media80 <= percentil80 and media80 != 0) or (media160 <= percentil160 and media160 != 0) or (media320 <= percentil320 and media320 != 0) or (media640 <= percentil640 and media640 != 0) or (media5 <= percentil5geral and media5 != 0) or (media10 <= percentil10geral and media10 != 0) or (media20 <= percentil20geral and media20 != 0) or (media40 <= percentil40geral and media40 != 0) or (media80 <= percentil80geral and media80 != 0) or (media160 <= percentil160geral and media160 != 0) or (media320 <= percentil320geral and media320 != 0) or (media640 <= percentil640geral and media640 != 0):
            apostar = 1
            print("APOSTAR NA PROXIMA RODADA")
        else:
            apostar = 0



print(lista_entradas)
print(data_final)

data_final.to_csv('/home/darkcover/Documentos/Out/dados/data_final.csv')