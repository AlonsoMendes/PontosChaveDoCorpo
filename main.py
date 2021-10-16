#CRIANDO NO MODELO MPI

#Importando bibliotecas
import cv2
import matplotlib.pyplot as plt
import numpy as np

# EXTRAINDO AS PASTA
#pose_path = "/content/drive/My Drive/pose.zip"
#zip_object = zipfile.ZipFile(file=pose_path, mode="r")
#zip_object.extractall("./")

#imagens_path = "/content/drive/My Drive/imagens.zip"
#zip_object = zipfile.ZipFile(file=imagens_path, mode="r")
#zip_object.extractall("./")
#zip_object.close()

arquivo_proto = "X:/PYTHON/PontosChaveDoCorpo/pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
arquivo_pesos = "X:/PYTHON/PontosChaveDoCorpo/pose/body/mpi/pose_iter_160000.caffemodel"

numero_pontos = 15
pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
               [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]
#print(pares_pontos)

cor_ponto, cor_linha = (255, 128, 0), (7, 62, 248)

imagem = cv2.imread("X:/PYTHON/PontosChaveDoCorpo/imagens/body/single/single_3.jpg")
#cv2.imshow('imagem', imagem)
#cv2.waitKey(0)

#Conversão da imagem para a implementação dos pontos
imagem_copia = np.copy(imagem)

imagem_largura = imagem.shape[1]
imagem_altura = imagem.shape[0]
#print(imagem_largura, imagem_altura)

#Carregar a rede neural
modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)

altura_entrada = 368
largura_entrada = int((altura_entrada / imagem_altura) * imagem_largura)
#print(largura_entrada)

#Convertendo a imagem de openCV para o formato Blob Caffe
blob_entrada = cv2.dnn.blobFromImage(imagem, 1.0 / 255,
                                     (largura_entrada, altura_entrada),
                                     (0, 0, 0), swapRB=False, crop=False)

#Saída
modelo.setInput(blob_entrada)# imagem convertida
saida = modelo.forward()# retornar as previsão da rede neural
#print(saida.shape)

altura = saida.shape[2]
largura = saida.shape[3]
#print(altura, largura)

#Plotando as saídas na imagem
pontos = []
limite = 0.1

for i in range(numero_pontos):
    mapa_confianca = saida[0, i, :, :]# mostra posição horizontal e vertical
    _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)# Buscar o maior ponto de confiabilidade no mapa
    #print(confianca)
    #print(ponto)

    x = (imagem_largura * ponto[0]) / largura
    y = (imagem_altura * ponto[1]) / altura

    if confianca > limite:
        cv2.circle(imagem_copia, (int(x), int(y)), 8, cor_ponto, thickness=-1, lineType=cv2.FILLED) # Desenha um circulo
        cv2.putText(imagem_copia, "{}".format(i), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, lineType=cv2.LINE_AA)

        pontos.append((int(x), int(y)))
    else:
        pontos.append(None)

    #print(len(pontos))
    #print(pontos)
#Criando uma máscara para desenho do esqueleto
tamanho = cv2.resize(imagem, (imagem_largura, imagem_altura))
mapa_suave = cv2.GaussianBlur(tamanho, (3, 3), 0, 0)
mascara_mapa = np.uint8(mapa_suave > limite)

#Desenhar o esqueleto: quando temos os pontos chave, apenas juntamos os pares
for par in pares_pontos:
    # print(par)
    parteA = par[0] #Ponto de origem
    parteB = par[1] #Ponto de destino

    if pontos[parteA] and pontos[parteB]: #Verificando se o valor é existente
       #Colocando um esqueleto na imagem original
        cv2.line(imagem, pontos[parteA], pontos[parteB], cor_linha, 3) #Fazendo as ligações das linhas
        cv2.circle(imagem, pontos[parteA], 8, cor_ponto, thickness=-1,
                   lineType=cv2.LINE_AA) #Fazendo um circulo
        #Colocando uma mascara de fundo preto
        cv2.line(mascara_mapa, pontos[parteA], pontos[parteB], cor_linha, 3)
        cv2.circle(mascara_mapa, pontos[parteA], 8, cor_ponto, thickness=-1,
                   lineType=cv2.LINE_AA)
#Exibindo as saídas
plt.figure(figsize=[14, 10])
plt.imshow(cv2.cvtColor(imagem_copia, cv2.COLOR_BGR2RGB))

plt.figure(figsize=[14, 10])
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))

plt.figure(figsize=[14, 10])
plt.imshow(cv2.cvtColor(mascara_mapa, cv2.COLOR_BGR2RGB))
plt.show()
