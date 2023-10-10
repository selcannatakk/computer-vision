import glob

from ultralytics import YOLO
import cv2 as cv
import glob
import os

# model_path = './weights/yolov8n-seg.pt'  # will change
# image_path = './data/images/val/image.jpg'  # will change
#
#
# image = cv.imread(image_path)
# H, W, _ = image.shape
#
# model = YOLO(model_path)
#
# results = model(image)
#
# for result in results:
#     for i, mask in enumerate(result.masks.data):
#         mask = mask.numpy() * 255  # mask scale = 0-1 * 255
#         mask = cv.resize(mask, (W, H))
#         cv.imwrite('./output.jpg', mask)


model_path = './weights/yolov8n-seg.pt'  # will change
images_path = './data/images/train/'  # will change

count = 0
for file in os.listdir(images_path):
    # for filename in glob.glob(images_path + "*.jpg"):
    image = cv.imread(images_path+file)
    H, W, _ = image.shape

    model = YOLO(model_path)

    results = model(image)

    for result in results:
        for i, mask in enumerate(result.masks.data):
            mask = mask.numpy() * 255  # mask scale = 0-1 * 255
            mask = cv.resize(mask, (W, H))
            cv.imwrite(f'././data/images/val_segmentation_output/{file}_{count}.png', mask)

    count+1
