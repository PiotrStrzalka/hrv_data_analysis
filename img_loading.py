import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('szwajcaria.jpg',cv2.IMREAD_COLOR)

# px = img[640, 320, 2]
# print(px)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture('VID_20180102_140825.mp4')
values1 = []

for a in range(0, 87):
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if ret == True:
        RedPixelsVerticalMeansSum = 0
        for x in range(0,719):
            VerticalRedPixelSum = 0
            for y in range(0,1279):           
                RedPixelValue = img[x, y, 0]
                VerticalRedPixelSum += RedPixelValue    

            VerticalRedPixelMean = VerticalRedPixelSum/1280
            RedPixelsVerticalMeansSum += VerticalRedPixelMean

        RedPixelsMean = RedPixelsVerticalMeansSum/720
        print(RedPixelsMean)

        values1.append(RedPixelsMean)

plt.plot(values1, "r-", label=str(values1[-1]))   
plt.grid(True)
plt.title('Wartość R piksela 640x320') 
plt.show()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
cap.release()
input("press key to continue")

cv2.destroyAllWindows()











# import cv2
# import numpy as np

# def draw_circle(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)

# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)

# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()


# import numpy as np
# from matplotlib import pyplot as plt
# import cv2

# img = np.zeros((512,512,3), np.uint8)

# cv2.line(img,(0,0),(511,511),(255,0,0),5)

# img2 = cv2.imshow('image', img)
# # plt.imshow(img, cmap ='gray', interpolation = 'bicubic')
# # img.show('image', img)
# cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()


# img.show('image', img)
# cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()


# import cv2

# cap = cv2.VideoCapture('TI_OPTI3101_Proximity_Sensor_Test.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.avi', fourcc, 29.0, (1280,720))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         # frame = cv2.flip(frame,0)

#         out.write(frame)

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()









# cap = cv2.VideoCapture('TI_OPTI3101_Proximity_Sensor_Test.mp4')

# while(True):
#     ret, frame = cap.read()

#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()






# from matplotlib import pyplot as plt

# img = cv2.imread('szwajcaria.jpg',cv2.IMREAD_COLOR)

# plt.imshow(img, cmap ='gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])
# plt.show()

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()