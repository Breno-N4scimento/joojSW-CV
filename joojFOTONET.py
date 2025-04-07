from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (MobileNetV2, preprocess_input, decode_predictions)
from tensorflow.keras.preprocessing.image import img_to_array

#carrega o modelo YOLO
modelo_yolo = YOLO("../yolov8n.pt")

#carrega o modelo MobileNetV2
modelo_roupa = MobileNetV2(weights='imagenet', include_top=True)

#caminho para a imagem que você quer analisar
caminho_imagem = "C:\\Users\\elbri\\Downloads\\casaco.jpg"

#carrega a imagem
imagem = cv2.imread(caminho_imagem)

#faz a predição com o modelo YOLO
resultados = modelo_yolo.predict(source=imagem, conf=0.3, show=False)
resultado = resultados[0]

#analisa os objetos detectados
for caixa, classe, conf in zip(resultado.boxes.xyxy, resultado.boxes.cls, resultado.boxes.conf):
    nome_classe = modelo_yolo.names[int(classe)]

    #se for pessoa, recorta a área da imagem
    if nome_classe == 'person':
        x1, y1, x2, y2 = map(int, caixa)
        pessoa_crop = imagem[y1:y2, x1:x2]

        #prepara a imagem para o modelo MobileNetV2
        img_resized = cv2.resize(pessoa_crop, (224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        #faz a predição
        preds = modelo_roupa.predict(img_array)
        rotulo = decode_predictions(preds, top=1)[0][0]

        #mostra o rótulo sobre a pessoa detectada
        texto = f"{rotulo[1]} ({rotulo[2]*100:.2f}%)"
        print(f"Predição: {texto}")
        cv2.rectangle(imagem, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(imagem, texto, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#exibe a imagem com detecções
cv2.imshow("Roupas Detectadas", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
