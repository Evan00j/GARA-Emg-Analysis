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


## Making sure files are in proper format for array conversion
soft = 'soft.txt'
force = 'forceful.txt'
flat = 'newflat.txt'

for line in fileinput.FileInput(soft, inplace=1):
    if line.replace("\n",""):
        print(line)
for line in fileinput.FileInput(flat, inplace=1):
    if line.replace("\n",""):
        print(line)
for line in fileinput.FileInput(force, inplace=1):
    if line.replace("\n",""):
        print(line)

softsignal = np.loadtxt(soft)
forcesignal = np.loadtxt(force)
flatsignal = np.loadtxt(flat)
type

#This section does various biosspy operations to get onsets and output it to proper places
output_ts = emg.emg(signal=forcesignal, sampling_rate=308, show=False)
ts = output_ts['ts']
londrelsignal = emg.londral_onset_detector(signal=forcesignal, rest=flatsignal, sampling_rate=308,size=50, threshold=6.33, active_state_duration=80)
onsets = londrelsignal['onsets']
filtered = output_ts['filtered']
lasttime=0
print('Start:')
for i in range(len(ts)):
    for j in range(len(onsets)):
        if i == onsets[j]:
            currenttime = ts[i]
            deltatime = currenttime - lasttime
            print(ts[i])
            time.sleep(deltatime)
            lasttime = ts[i]
            ser.write('l'.encode())

#Random extra stuff, didnt wanna delete it
#plotting.plot_emg(ts=ts, sampling_rate=308, raw=forcesignal, filtered=filtered, onsets=onsets, show=True)
#output_slow = emg.emg(signal=softsignal, sampling_rate=290, show=True)
#output_force = emg.emg(signal=forcesignal, sampling_rate=290, show=True)
#print(output_slow['onsets'])
#print(output_force['onsets'])
#outputz = emg.emg(signal=movesignal, sampling_rate=281., show=True)
#output = emg.emg(signal=rawsignal, sampling_rate=281., show=True)