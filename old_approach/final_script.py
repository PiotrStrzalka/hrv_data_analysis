import cv2
import numpy as np
from numpy import array
from scipy.signal import butter, lfilter
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def get_H_values_from_video(Output_array, videoName, videoLength, videoFPS):
    cap = cv2.VideoCapture(videoName)
    for a in range(0, videoLength*videoFPS):   #1740
        ret, frame = cap.read()
        #cv2.yuv

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # numpyArray = array(img)[0:719,0:1279,0] 


        numpyArray = array(img)[0:719,0:1279,0] 


        x = numpyArray.mean()
        Output_array.append(x)
        if(a == 0):
            Output_array += [x] * 100
        print(x)
        print('Frame number: ' + str(a))
    cap.release()
    cv2.destroyAllWindows()

def butter_bandpass_filter(data, lowcut, highcut, fs, order =4):
    nyq = 0.5 * fs #fs- sampling frequency
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low,high], btype='band')
    y = lfilter(b, a, data)
    return y

def cubic_spline_interpolation(InputList, spline_resolution):
    x_temporary_values = np.arange(len(InputList))
    interpolation_forumla = CubicSpline(x_temporary_values, InputList)

    output = []
    for a in range(100*spline_resolution, len(InputList)*spline_resolution):
        output.append(interpolation_forumla(a/spline_resolution))

    return output

def readPeaksFromPlot(InputList):
    peaksX = []
    for a in range(250, len(InputList)-250):       # 1 step is a 1/290sec = 0,35ms
        if (InputList[a-1] < InputList[a]) and (InputList[a-250] < (InputList[a]-0.15)):
            if (InputList[a] > InputList[a+1]) and (InputList[a]-0.15 > InputList[a+250]):
                peaksX.append(a)

    intervalsArray = []
    for c in range(1,len(peaksX)):
        intervalsArray.append(peaksX[c]-peaksX[c-1])


    mean = np.mean(intervalsArray)
    print('mean1= ' + str(mean))

    median = np.median(intervalsArray)
    print('median = ' + str(median))

    c = 1
    while c < len(peaksX):
        if (peaksX[c]-peaksX[c-1]) < (0.7 * median):
            peaksX.remove(peaksX[c])
            c = c - 1
            
        c = c + 1

    return peaksX

def changeCoordinatesToMs(peaks, resolution):
    RR = []
    oneStepMs = 34,5/resolution
    for b in range(1, len(peaks)):
        RRms = (peaks[b] - peaks[b-1])*0.345
        x = round(RRms,1)
        RR.append(x)
        print(str(x) + '\n')
    return RR


def SDN50(peaks):
    moreThanFiftyMs = 0
    for a in range(1 , len(peaks)):
        if(peaks[a] - peaks[a-1]):
            moreThanFiftyMs = moreThanFiftyMs+1
    return (moreThanFiftyMs/(len(peaks)-1))*100

def SDNN(peaks):
    # SDarray = []
    # for a in range(1 , len(peaks)):
    #     SDarray.append(abs(peaks[a] - peaks[a-1]))

    return round(np.std(peaks),1)


def getFftOfData(InputValues):
    t = np.arange(len(InputValues))
    sp = np.fft.fft(InputValues)
    freq = np.fft.fftfreq(t.shape[-1])


videoLength = 15
videoFPS = 29
videoTitle = 'video4_1min.mp4'
resolution = 100

H_values =[]

get_H_values_from_video(H_values, videoTitle, videoLength, videoFPS)

values_after_butter = butter_bandpass_filter(H_values, 0.1, 8, 29)

values_after_spline = cubic_spline_interpolation(values_after_butter, resolution)

values_after_spline = [ -x for x in values_after_spline]

peaks = readPeaksFromPlot(values_after_spline)

RRIntervals = changeCoordinatesToMs(peaks, resolution)

SDNNx = SDNN(RRIntervals)


print("SDNN in overall " + str(SDNNx) + "ms")

plt.subplot(3, 1, 1)
plt.plot(values_after_spline, "r-", label="Interpolation")   
plt.grid(True)
plt.ylabel('After cubic spline interpolation') 

for b in range(1, len(peaks)):
    annotation = (str(round((peaks[b] - peaks[b-1])*0.345,1)) + 'ms')
    plt.annotate(annotation,xy=(peaks[b],values_after_spline[peaks[b]]), xytext=(peaks[b]-170, values_after_spline[peaks[b]]+0.1), arrowprops=dict(facecolor='black',shrink=0.01))


plt.subplot(3, 1, 2)
values_after_butter
plt.plot(values_after_butter, "r-", label="before") 

plt.subplot(3, 1, 3)

plt.plot(H_values, "r-", label="raw")
# plt.subplot(2, 1, 2)
# t = np.arange(len(RRIntervals))
# sp = np.fft.fft(RRIntervals)
# freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, sp.real)

plt.show()




input("press key to continue")