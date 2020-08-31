import cv2
import numpy as np

cap = cv2.VideoCapture('video5_Ania.mp4')
values1 = []

OutputFile = open('HsvValuesVideo5.txt','w') 

for a in range(0, 1740):
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if ret == True:
        RedPixelsVerticalMeansSum = 0
        for y in range(0,719):
            HorizontalRedPixelSum = 0
            for x in range(0,1279):           
                Hsv = img[y, x, 0]
                HorizontalRedPixelSum += Hsv    

            HorizontalRedPixelMean = HorizontalRedPixelSum/1280
            RedPixelsVerticalMeansSum += HorizontalRedPixelMean

        RedPixelsMean = RedPixelsVerticalMeansSum/720
        print(RedPixelsMean)
        values1.append(RedPixelsMean)
        OutputFile.write((str(RedPixelsMean) + "\n"))

OutputFile.close()
cap.release()
input("press key to continue")

cv2.destroyAllWindows()