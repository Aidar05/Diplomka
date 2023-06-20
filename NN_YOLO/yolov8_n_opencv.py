import random

import cv2
import numpy as np
from ultralytics import YOLO

# Открываем и читаем файл с названиями объектов
my_file = open("NN_YOLO/utils/labels.txt", "r")
data = my_file.read()

# Добавляем все в этом файwле в список
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Генерируем рандомные значения для RGB цвета прямоугольника, который окружает объект. Так красивей
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Уже натренированная замороженная модель
model = YOLO("NN_YOLO/weights/yolov8n.pt", "v8")

# Переменные для размеры окна. Делаем меньше экран, чтобы оптимизировать
frame_wid = 600
frame_hyt = 400

# Даем для обработки либо видео, либо веб-камеру
# Это веб-камера
cap = cv2.VideoCapture(0)

# Это для видео   
#cap = cv2.VideoCapture("NN_YOLO/inference/videos/afriq1.MP4")

# Проверка на включенную камеру
if not cap.isOpened():  
    print("Cannot open camera or video")
    exit()

while True:
    # Читаем каждый фрейм видео/вебки
    ret, frame = cap.read()

    # Если не получили фрейм корректно
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  Размер окна уменьшаем
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Обрабатывает фрейм, узнает объекты
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Конвертирует тенсор коллаж в numpy коллаж
    DP = detect_params[0].numpy()
    # print(DP)

    # Если коллаж не пустой, тоесть хотя бы один объект узнался в текущем фрейме
    if len(DP) != 0:
        # Проходиться по всем результатам распознования, показывает для каждой названия класса и прямоугольник
        for i in range(len(detect_params[0])):

            boxes = detect_params[0].boxes  # Это все результаты
            box = boxes[i]  # Берем текущий результат 
            clsID = box.cls.numpy()[0]  # Название объекта
            conf = box.conf.numpy()[0]  # Уверенность. Насколько точно нейнонка уверена в ответе
            bb = box.xyxy.numpy()[0]    # Координаты

            # Рисуем прямогульник вокруг объекта
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Показать название класса и насколько нейронка уверена в ответе
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Показать фрейм который получился
    cv2.imshow("ObjectDetection", frame)

    # Выход на q
    if cv2.waitKey(1) == ord("q"):
        break

# Когда все закончено закрыть видео/вебку
cap.release()
cv2.destroyAllWindows()
