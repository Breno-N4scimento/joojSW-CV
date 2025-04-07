from ultralytics import YOLO
import cv2

model = YOLO("../yolov8n.pt")

#caminho para a imagem que você quer analisar
caminho_imagem = "C:\\Users\\elbri\\Downloads\\casaco.jpg"

#carrega a imagem com OpenCV
imagem = cv2.imread(caminho_imagem)

#faz a predição com o modelo YOLO
resultados = model.predict(source=imagem, conf=0.3, show=False)

resultado = resultados[0]

#mostra os resultados
for caixa, classe, conf in zip(resultado.boxes.xyxy, resultado.boxes.cls, resultado.boxes.conf):
    x1, y1, x2, y2 = map(int, caixa)
    nome_classe = model.names[int(classe)]
    confianca = float(conf)

    #desenha a caixa e o nome do objeto
    cv2.rectangle(imagem, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(imagem, f"{nome_classe} ({confianca:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#exibe a imagem com detecções
cv2.imshow("Roupas Detectadas", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
