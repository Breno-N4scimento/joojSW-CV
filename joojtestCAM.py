import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (MobileNetV2, preprocess_input, decode_predictions)
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

modelo_yolo = YOLO("../yolov8n.pt")
modelo_roupa = MobileNetV2(weights='imagenet', include_top=True)

#inicia a captura da webcam
cap = cv2.VideoCapture(0)

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #faz a predição com o modelo YOLO
    resultados = modelo_yolo.predict(source=frame, conf=0.3, show=False)
    resultado = resultados[0]

    #analisa as detecções
    for caixa, classe, conf in zip(resultado.boxes.xyxy, resultado.boxes.cls, resultado.boxes.conf):
        nome_classe = modelo_yolo.names[int(classe)]

        if nome_classe == 'person':
            x1, y1, x2, y2 = map(int, caixa)
            pessoa_crop = frame[y1:y2, x1:x2]

            try:
                #prepara a imagem para o modelo MobileNetV2
                img_resized = cv2.resize(pessoa_crop, (224, 224))
                img_array = img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                #faz a predição
                preds = modelo_roupa.predict(img_array)
                rotulo = decode_predictions(preds, top=1)[0][0]

                texto = f"{rotulo[1]} ({rotulo[2]*100:.2f}%)"
            except:
                texto = "Erro na classificação"

            #desenha na tela
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, texto, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    #exibe o resultado em tempo real
    cv2.imshow("Detecção em tempo real", frame)

    #sai do loop se apertar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#libera a câmera e fecha a janela
cap.release()
cv2.destroyAllWindows()
