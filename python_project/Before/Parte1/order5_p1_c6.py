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
                    'odd_saida': [0], 'media80': [0], 'desvpad80geral': [0], 'percentil80geral': [0], 'cv80':[0], 'roc80':[0] , 'media160': [0], 'desvpad160geral': [0], 'percentil160geral': [0], 'cv160':[0], 'roc160':[0], 'media320': [0], 'desvpad320geral': [0], 'percentil320geral': [0], 'cv320':[0], 'roc320':[0],'media640': [0], 'desvpad640geral': [0], 'percentil640geral': [0], 'cv640':[0], 'roc640':[0]
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
    if apostar == 1 and odd_saida >= 6:
        print(24*'$')
        acerto = 1
        contagem += 1
        if contagem == 15:
            level += 1
            contagem = 0
        print(f"Houve acerto! \nA ultima entrada determinada foi: {lista_entradas[-1]} \nA condição de vitoria está em {contagem} \nLevel: {level} \n{24*'*-'}")
    if apostar == 1 and odd_saida < 6:
        print(24*'$')
        acerto = 0
        contagem = contagem - 2
        if contagem <= -10:
            level = 1
            contagem = 0 
            print(f"Houve erro! \nA ultima entrada determinada foi: {lista_entradas[-1]} \nA condição de vitoria está em {contagem}, dessa maneira GAME OVER. \n{24*'*-'}")
        else:
            print(f"Houve erro! \nA ultima entrada determinada foi: {lista_entradas[-1]} \nA condição de vitoria está em {contagem} \nLevel: {level} \n{24*'*-'}")
    
    if apostar == 0 and odd_saida >= 6:
        acerto = 1
    if apostar == 0 and odd_saida < 6:
        acerto = 0

    lista_entradas.append(i)

    if len(lista_entradas) < 81:
        data_final.loc[len(data_final.index)] = [len(lista_entradas), level, apostar, acerto, contagem, odd, odd_entrada, odd_saida, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        data_final.loc[len(data_final.index)] = [len(lista_entradas), level, apostar, acerto, contagem, odd, odd_entrada, odd_saida, media80, desvpad80geral, percentil80geral, cv80, roc80, media160, desvpad160geral, percentil160geral, cv160, roc160,media320, desvpad320geral, percentil320geral, cv320, roc320, media640, desvpad640geral, percentil640geral, cv640, roc640]
        apostar = 0

#A partir da quinta entrada, essa parte do código começa a fazer o acompanhamento das médias.
    if len(lista_entradas) >= 80:

        ultimas80 = lista_entradas[-80:]
        ultimas160 = [0]
        ultimas320 = [0]
        ultimas640 = [0]

        if len(lista_entradas) >= 160:
            ultimas160 = lista_entradas[-160:]
            if len(lista_entradas) >= 320:
                ultimas320 = lista_entradas[-320:]
                if len(lista_entradas) >= 640:
                    ultimas640 = lista_entradas[-640:]
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media80 = sum(ultimas80) / 80
        arraymedia80.append(media80)
        percentil80geral = np.percentile(arraymedia80, 25)
        desvpad80geral = np.std(ultimas80)
        cv80 = media80 / desvpad80geral
        media_anterior = np.mean(lista_entradas[-160:-80])
        roc80 = ((media80 - media_anterior) / media_anterior) * 100
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media160 = sum(ultimas160) / 160
        arraymedia160.append(media160)
        if len(lista_entradas) < 160:
            percentil160geral = 0
            desvpad160geral = 0
            cv160 = 0
            roc160 = 0
        else:
            percentil160geral = np.percentile(arraymedia160, 25)
            desvpad160geral = np.std(ultimas160)
            cv160 = desvpad160geral / media160
            media_anterior = np.mean(lista_entradas[-320:-160])
            roc160 = ((media160 - media_anterior) / media_anterior) * 100
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media320 = sum(ultimas320) / 320
        arraymedia320.append(media320)
        if len(lista_entradas) < 320:
            percentil320geral = 0
            desvpad320geral = 0
            cv320 = 0
            roc320 = 0
        else:
            percentil320geral = np.percentile(arraymedia320, 25)
            desvpad320geral = np.std(ultimas320)
            cv320 = desvpad320geral / media320
            media_anterior = np.mean(lista_entradas[-640:-320])
            roc320 = ((media160 - media_anterior) / media_anterior) * 100
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        media640 = sum(ultimas640) / 640
        arraymedia640.append(media640)
        if len(lista_entradas) < 640:
            percentil640geral = 0
            desvpad640geral = 0
            cv640 = 0
            roc640 = 0
        else:
            percentil640geral = np.percentile(arraymedia640, 25)
            desvpad640geral = np.std(ultimas640)
            cv640 = desvpad640geral / media640
            media_anterior = np.mean(lista_entradas[-1280:-640])
            roc640 = ((media640 - media_anterior) / media_anterior) * 100
#*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
        
        print(f"Rodada: {len(lista_entradas)} \nMedia 80: {media80} \nDP80: {desvpad80geral} \nPercentil80Geral: {percentil80geral} \ncv80:{cv80} \nroc80: {roc80}\nMedia 160: {media160} \nDP160: {desvpad160geral} \nPercentil160Geral: {percentil160geral} \ncv160:{cv160} \nroc160: {roc160} \nMedia 320: {media320} \nDP320: {desvpad320geral} \nPercentil320Geral: {percentil320geral} \ncv320:{cv320} \nroc320: {roc320} \nMedia 640: {media640} \nDP640: {desvpad640geral}\nPercentil640Geral: {percentil640geral} \ncv640:{cv640} \nroc640: {roc640}")

#Esses intervalos de médias são pré-determidos. Podendo mudar de acordo com nossos estudos.
        if (media80 <= percentil80geral and cv80 <= 2 and roc80 >= -2 and media80 != 0) or (media160 <= percentil160geral and cv160 <= 0.47 and roc160 >= -1 and media160 != 0) or (media320 <= percentil320geral and cv320 <= 0.47 and roc320 >= -1 and media320 != 0) or (media640 <= percentil640geral and cv640 <= 0.47 and roc640 >= -1 and media640 != 0):
            apostar = 1
            print("APOSTAR NA PROXIMA RODADA")
        else:
            apostar = 0



print(lista_entradas)
print(data_final)

data_final.to_csv('/home/darkcover/Documentos/Out/dados/data_final3.csv')