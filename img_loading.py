import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.interpolate import CubicSpline


def butter_bandpass_filter(data, lowcut, highcut, fs, order =4):
    nyq = 0.5 * fs #fs- sampling frequency
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low,high], btype='band')
    y = lfilter(b, a, data)
    return y

H_values = []
InputFile = open('HsvValuesVideo4.txt','r')

for line in InputFile.readlines():
    H_values.append(-(float(line)))

# for each in InputFile:
#     H_values.append(float(InputFile.read()))


InputFile.close()

print(H_values[0])
print(H_values[1])
values_after_butter = butter_bandpass_filter(H_values, 0.33, 4, 29)

print(len(values_after_butter))

x_temporary_values = np.arange(1840)
# xs = []
# xs.arange(1000,18410)
xs= np.arange(1000, 18410, 1)
interpolation_forumla = CubicSpline(x_temporary_values,values_after_butter)


output = []
for a in range(1000, 18410):
    output.append(interpolation_forumla(a/10))

peaksX = []
for a in range(25, 17384):       # 1 step is a 1/290sec = 3,5ms
    if (output[a-1] < output[a]) and (output[a-25] < (output[a]-0.15)):
        if (output[a] > output[a+1]) and (output[a]-0.15 > output[a+25]):
            peaksX.append(a)

intervalsArray = []
for c in range(1,len(peaksX)):
    intervalsArray.append(peaksX[c]-peaksX[c-1])

print(peaksX)

mean = np.mean(intervalsArray)
print('mean1= ' + str(mean))

median = np.median(intervalsArray)
print('median = ' + str(median))

peaksToRemove = []

print('Length of peaksX = ' + str(len(peaksX)))

# for c in range(1 , 5):
#     if (peaksX[c]-peaksX[c-1]) < (0.8 * median):
#         peaksX.remove(peaksX[c])
#         # c = c - 1

# plt.subplot(3,1,1)
# plt.plot(H_values, "r-", label=str(H_values[-1]))   
# plt.grid(True)
# plt.ylabel('H from HSV value') 

# plt.subplot(3, 1, 2)
# plt.plot(values_after_butter, "r-", label=str(H_values[-1]))   
# plt.grid(True)
# plt.ylabel('After 4-score butterworth filter') 

plt.subplot(1, 1, 1)
plt.plot(output, "r-", label="Interpolation")   
plt.grid(True)
plt.ylabel('After cubic spline interpolation') 

x = len(peaksX)
for b in range(1, len(peaksX)):
    annotation = (str(round((peaksX[b] - peaksX[b-1])*3.6,1)) + 'ms')
    plt.annotate(annotation,xy=(peaksX[b],output[peaksX[b]]), xytext=(peaksX[b]-170, output[peaksX[b]]+0.1), arrowprops=dict(facecolor='black',shrink=0.01))
    # print(str(peaksX[b]))

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