from biosppy import storage
from biosppy import plotting
import numpy as np
from biosppy.signals import emg
import fileinput
import time
import serial
import matplotlib.pyplot as plt

## Setup for serial read/write
serial_port = 'COM9';
baud_rate = 9600;
ser = serial.Serial(serial_port, baud_rate)
testList = []
counter = 0
currentTime = time.time()
resetTime = currentTime + 2

while(ser):
    currentTime = time.time()
    value = ser.readline()
    decoded = value.decode('utf-8')
    testList.append(float(decoded))
    counter += 1
    if currentTime > resetTime:
        for i in range(len(testList)):
            if testList[i] >= 4.0:
                print('           Onset Detected!')
                break
        print(counter)
        currentTime = time.time()
        resetTime = currentTime + 1
        testList.clear()
        counter = 0
        print("Serial working...\r")



# #This section does various biosspy operations to get onsets and output it to proper places
# output_ts = emg.emg(signal=forcesignal, sampling_rate=308, show=False)
# ts = output_ts['ts']
# londrelsignal = emg.londral_onset_detector(signal=forcesignal, rest=flatsignal, sampling_rate=308,size=50, threshold=6.33, active_state_duration=80)
# onsets = londrelsignal['onsets']
# filtered = output_ts['filtered']
# lasttime=0
# print('Start:')
# for i in range(len(ts)):
#     for j in range(len(onsets)):
#         if i == onsets[j]:
#             currenttime = ts[i]
#             deltatime = currenttime - lasttime
#             print(ts[i])
#             time.sleep(deltatime)
#             lasttime = ts[i]
#             ser.write('l'.encode())

#Random extra stuff, didnt wanna delete it
#plotting.plot_emg(ts=ts, sampling_rate=308, raw=forcesignal, filtered=filtered, onsets=onsets, show=True)
#output_slow = emg.emg(signal=softsignal, sampling_rate=290, show=True)
#output_force = emg.emg(signal=forcesignal, sampling_rate=290, show=True)
#print(output_slow['onsets'])
#print(output_force['onsets'])
#outputz = emg.emg(signal=movesignal, sampling_rate=281., show=True)
#output = emg.emg(signal=rawsignal, sampling_rate=281., show=True)