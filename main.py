import cv2 as cv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate

def nothing():
    pass

DEFAULT_VIDEO_PATH = r"C:\PROJEKTY\sandbox\hrv_data_analysis\example_videos\video9_Tata.mp4"
video_path = DEFAULT_VIDEO_PATH

if len(sys.argv) > 1:
    if os.path.isfile(sys.argv[1]):
        video_path = sys.argv[1]        
print("Using video: " + video_path)      

def get_video_mean_data(path):
    video_csv_path = path[:-3] + "txt"
    arr = None
    
    if os.path.isfile(video_csv_path):
        print("Loading from file: " + video_csv_path)
        arr = np.loadtxt(video_csv_path)
    else:
        print("Calculating new video data")
        try:
            cap = cv.VideoCapture(video_path)
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frames_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            print("Video width: " + str(width) + " height: " + str(height) + " frames count: " + str(frames_count))
        except Exception as e:
            print("Cannot open video error occured: " + str(e))
            exit(1)

        frame_counter = 0
        timestamp = 0.0
        while cap.isOpened():
            ret, frame = cap.read()   
            if ret == False:
                break  
            
            mean_brg = cv.mean(frame)
            mean_hsv = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2HSV))            
            timestamp = cap.get(cv.CAP_PROP_POS_MSEC)  
            
            a = np.array((timestamp, mean_brg[0], mean_brg[1],
                mean_brg[2], mean_hsv[0], mean_hsv[1], mean_hsv[2]), dtype = 'f', ndmin=2) 
            
            if arr is None:
                arr = a.copy()
            else:
                arr = np.vstack((arr, a))
                
            frame_counter = frame_counter + 1
            print("Calculating: " + str(round((frame_counter *100 / frames_count), 2)) + "%")
            
        # For some reason last 6 frames does not have correct timestamp values,
        # will be deleted
        
        arr = arr[:-6]
        
        np.savetxt(video_csv_path, arr)
        cap.release() 
    
    return arr     

BGR_B_INDEX = 1
BGR_G_INDEX = 2
BGR_R_INDEX = 3
HSV_H_INDEX = 4
HSV_S_INDEX = 5
HSV_V_INDEX = 6


video_array = get_video_mean_data(video_path)

def subplot(address, index, name, color = "b"):
    plt.subplot(address)
    plt.plot(video_array[30:,0]/1000, video_array[30:,index], color)
    plt.ylabel(name)
    
sos = signal.butter(10, 0.3333, 'hp', fs = 29.85, output='sos')
def subplot_highpass(address, index, name, color = "b", runup = 300):
    plt.subplot(address)
    data : np.ndarray = video_array[:,index]
    mean = np.mean(data[:runup])
    print("Mean of " + name + " is: " + str(mean))
    
    start = np.full((runup), mean)
    then = video_array[:,index].copy()
    connected = np.hstack((start, then))
    filtered = signal.sosfilt(sos, connected)
    filtered = filtered[runup:]
    plt.plot(video_array[30:,0]/1000, filtered[30:], color)
    plt.ylabel(name)

plt.figure()
# numrows, numcols, plot_number where plot_number ranges from 1 to numrows*numcols
subplot(321, HSV_H_INDEX, "R component", "r")
subplot(323, HSV_S_INDEX, "G component", "g")
subplot(325, HSV_V_INDEX, "B component", "b")

subplot_highpass(322, HSV_H_INDEX, "R high pass component", "orange")
subplot_highpass(324, HSV_S_INDEX, "G high pass component", "navy")
subplot_highpass(326, HSV_V_INDEX, "B high pass component", "dimgray")


# subplot(322, HSV_H_INDEX, "H component", "orange")
# subplot(324, HSV_S_INDEX, "S component", "navy")
# subplot(326, HSV_V_INDEX, "V component", "dimgray")

# Heart rate varies from 20 to 200 BPM (approximately) so the frequency differs
# 50 / 60 = 0.3333 Hz
# 200 / 60 = 3.3333 Hz

#  Sampling frequency for now lets say 29.85Hz

# highpass 
sos = signal.butter(5, 0.5, 'hp', fs = 29.85, output='sos')
filtered_R = signal.sosfilt(sos, video_array[:,BGR_R_INDEX])

# plt.subplot(211)
# plt.plot(video_array[:,0]/1000, video_array[:,BGR_R_INDEX], 'r')
# plt.ylabel("R component")

# plt.subplot(212)
# plt.plot(video_array[:,0]/1000, video_array[:,BGR_G_INDEX], 'g')
# plt.ylabel("G component")

plt.show()

























# while cap.isOpened():
#     start = time.time()
#     ret, frame = cap.read()    
#     duration = time.time() - start
#     print("Loading time: " + str(round(duration*1000, 2)) + "ms")
#     if ret == False:
#         break    
    
#     delta_time = cap.get(cv.CAP_PROP_POS_MSEC) - timestamp
#     timestamp = cap.get(cv.CAP_PROP_POS_MSEC)    
#     frame_counter = frame_counter + 1;
    
#     # cv.line(frame,(0,0),(width-1,height-1), (255,0,0), 5)
#     # r = cv.getTrackbarPos('R', 'image')
#     # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     mean, duration = calculate_mean(frame)
    
#     # font = cv.FONT_HERSHEY_SIMPLEX
#     # cv.putText(hsv, 'H' + str(round(mean[0],2)),(10,100), font, 1,(255,255,255),2,cv.LINE_AA)
#     # cv.putText(hsv, 'S' + str(round(mean[1],2)),(10,200), font, 1,(255,255,255),2,cv.LINE_AA)
#     # cv.putText(hsv, 'V' + str(round(mean[2],2)),(10,300), font, 1,(255,255,255),2,cv.LINE_AA)
    
#     # timeDelta = timeDelta * 1000
#     # cv.putText(hsv, 'Avg calculation time: ' + str(round(timeDelta,2)) + "ms",(10,400), font, 1,(255,255,255),2,cv.LINE_AA)
#     print('Calculation time: ' + str(duration) + "ms, " +
#           "position in video: " + str(frame_counter) + " / " + str(frames_count))
    
    
#     # cv.imshow("frame", hsv)
    
    
#     # if cv.waitKey(1) & 0xFF == ord('q'):
#     #     break

#     if frame_counter == 100:
#         break

# fieldnames = ["time_delta", "mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v"]

# cap.release();
# cv.destroyAllWindows()
    
