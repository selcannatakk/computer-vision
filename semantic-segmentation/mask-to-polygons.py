import cv2 as cv
import os

input_dir = 'data/tmp/masks'  # will change
output_dir = 'data/tmp/labels'  # will change

for p in os.listdir(input_dir):
    image_path = os.path.join(input_dir, p)
    # load the binary mask and get its contours
    mask = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

    H, W = 0
    mask.shape
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv.contourArea(cnt) > 200:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    # print the polygons
    with open('{}.txt'.format(os.path.join(output_dir, p)[:-4]), 'w') as f:
        for polygon in polygons:
            for p_, p in enumerate(polygon):
                if p_ == len(polygon) - 1:
                    f.write('{}\n'.format(p))
                elif p_ == 0:
                    f.write('0 {} '.format(p))
                else:
                    f.write('{} '.format(p))

        f.close()
