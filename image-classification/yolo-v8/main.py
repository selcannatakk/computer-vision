from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# load a pretrained model
model = YOLO('yolov8n-cls.pt')

# runs/classify/train...12 + val saved
model.train(data='./data', epochs=1, imgsz=64)

''' 
start train model 
yolo classify train data='./data' model=yolo8n-cls.pt epochs=100 imgsz=64
'''
