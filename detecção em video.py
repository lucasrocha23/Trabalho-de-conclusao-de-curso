import cv2 as cv
import cv2 as cv_2
import numpy as np
import glob
import random
import time


# Load Yolo
weights = "C:/Users/lucas/Documents/TCC/yolo teste/yolo_custom_detection/training_yolov4_1.5.weights"
config  = "C:/Users/lucas/Documents/TCC/yolo teste/yolo_custom_detection/yolov4.cfg"
# save = "C:/Users/lucas/Documents/TCC/yolo teste/bruto/validacao/yolov3-tiny/real/"

net = cv.dnn.readNet(weights,config)

# Name custom object
classes = ["carro","moto","caminhao","onibus",]

dire = 'C:/Users/lucas/Documents/TCC/yolo teste/predicao/treino.mp4'
cap = cv.VideoCapture(dire)

def predicao(img):
    height, width, channels = img.shape

    # Detecting objects
    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    colors = [(255,255,0),(0,255,255),(255,0,255),(0,0,255)]
    font = cv.FONT_HERSHEY_DUPLEX


    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                # print("ID's >>",class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    # print(indexes)
    carro    = 0
    moto     = 0
    caminhao = 0
    onibus   = 0
    x_ini    = 600
    y_ini    = 30



    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])
            cv.rectangle(img, (x, y), (x + w, y + h),  colors[class_ids[i]], 2)
            cv.putText(img, label, (x+3, y + 8), font, 0.3, colors[class_ids[i]], 1)

            if class_ids[i] == 0:
                carro += 1
            elif class_ids[i] == 1:
                moto += 1
            elif class_ids[i] == 2:
                caminhao += 1
            elif class_ids[i] == 3:
                onibus += 1

    total  = carro + moto + caminhao + onibus
    cv.putText(img, ("carro: " + str(carro)), (x_ini,y_ini), font, 1, (0,0,255), 2) 
    cv.putText(img, ("moto: " + str(moto)), (x_ini,y_ini + 30), font, 1, (0,0,255), 2)
    cv.putText(img, ("caminhao: " + str(caminhao)), (x_ini,y_ini + 60), font, 1, (0,0,255), 2)
    cv.putText(img, ("onibus: " + str(onibus)  ), (x_ini,y_ini + 90), font, 1, (0,0,255), 2)
    cv.putText(img, ("total: " + str(total)  ), (x_ini,y_ini + 120), font, 1, (0,0,255), 2)


layer_names = net.getLayerNames()
output_layers = []
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i - 1])

cont = 0
img_count = 1
while True:
    ret, img = cap.read()

    img = cv.resize(img, None, fx=1.15, fy=1.15)
    # cv.imshow("Image", img)
    

    if cont == 150:
        # save_dir = save + str(img_count) + "_original.png"
        # # print(img_count)
        # arq = open(save_dir, 'wb')
        # cv.imwrite(save_dir,img)
        # arq.close()
        
        # start_time = time.time()
        predicao(img)
        # end_time = time.time()
        # print("--- %s seconds ---" % (end_time - start_time))
        cont = 0
        
        cv_2.imshow("deteccao", img)
        key = cv.waitKey(1)
        if key == 27:
            break
        # save_dir = save + str(img_count) + "_IA.png"
        # arq = open(save_dir, 'wb')
        # cv.imwrite(save_dir,img)
        # arq.close()
        img_count += 1
            
    
    cont += 1
    
        
