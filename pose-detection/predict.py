from ultralytics import YOLO
import cv2 as cv

model_path = 'runs/pose/train/weights/last.pt'  # will change

image_path = './data/images/val/image.jpg'  # will change
img = cv.imread(image_path)

model = YOLO(model_path)

results = model(image_path)[0]

for result in results:
    for keypoint_indx, keypoint in enumerate(result.keypoints.tolist()):
        cv.putText(img, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv.imshow('img', img)
cv.waitKey(0)
