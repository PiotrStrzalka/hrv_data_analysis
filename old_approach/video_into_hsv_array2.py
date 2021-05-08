import cv2
import numpy as np
from numpy import array

cap = cv2.VideoCapture('video4_1min.mp4')
values1 = []

OutputFile = open('HsvValuesVideo52.txt','w') 

H_values = []

for a in range(0, 100):   #1740
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    numpyArray = array(img)[0:719,0:1279,0] 
    x = numpyArray.mean()
    H_values.append(x)
    print(x)
    print('Frame number: ' + str(a))

    OutputFile.write((str(x) + "\n"))

    # if ret == True:
    #     RedPixelsVerticalMeansSum = 0
    #     for y in range(0,719):
    #         HorizontalRedPixelSum = 0
    #         for x in range(0,1279):           
    #             Hsv = img[y, x, 0]
    #             HorizontalRedPixelSum += Hsv    

    #         HorizontalRedPixelMean = HorizontalRedPixelSum/1280
    #         RedPixelsVerticalMeansSum += HorizontalRedPixelMean

    #     RedPixelsMean = RedPixelsVerticalMeansSum/720
    #     print(RedPixelsMean)
    #     values1.append(RedPixelsMean)
    #     OutputFile.write((str(RedPixelsMean) + "\n"))

OutputFile.close()
cap.release()
input("press key to continue")

cv2.destroyAllWindows()