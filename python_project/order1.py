

i = 1
lista_entradas = []

while i != 0:
    i = float(input("Insira a entrada determinada: "))
    print("A entrada determinada foi: ", i)

    if i == 0:
        break
    lista_entradas.append(i)

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

print(lista_entradas)