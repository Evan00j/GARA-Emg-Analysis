import serial
import numpy as np
from biosppy.signals import emg
import fileinput
import time

### Setup and initiate serial port
serial_port = 'COM3';
baud_rate = 9600;
ser = serial.Serial(serial_port, baud_rate)

###Getting Onset values from data set
for line in fileinput.FileInput("moveout.txt", inplace=1):
    if line.replace("\n",""):
        print(line)
rawSignal = np.loadtxt("moveout.txt")


output = emg.emg(signal=rawSignal, sampling_rate=281., show=False)


for i in range(len(output['ts'])):
    for j in range(len(output['onsets'])):
        if i == output['onsets'][j]:
            print(output['ts'][i])
            ser.write('l'.encode())
            time.sleep(.5)
