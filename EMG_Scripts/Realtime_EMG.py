from biosppy import storage
from biosppy import plotting
import numpy as np
from biosppy.signals import emg
import fileinput
import time
import serial
import matplotlib.pyplot as plt
import datetime
import string

def connectToCOM(com_port='COM9', baud_rate=9600):
    serial_con = serial.Serial(com_port, baud_rate)
    return serial_con


def serBurstRead(ser_line, burst_length):
    readbufferA = []
    readbufferB = []
    currenttime = time.time()

    while time.time() < currenttime + burst_length:
        val_byte = ser_line.readline()
        val_utf8 = val_byte.decode('utf-8')
        if val_utf8[0] == 'a':
            try: readbufferA.append(float(val_utf8[1:-2]))
            except ValueError: False
        if val_utf8[0] == 'b':
            try: readbufferB.append(float(val_utf8[1:-2]))
            except ValueError: False
    return readbufferA, readbufferB

def serBurstReadTest(burst_length):
    readbuffer = []
    currenttime = time.time()

    while time.time() < currenttime + burst_length:
        readbuffer.append(datetime.date.today())
        time.sleep(.5)
    return readbuffer

## Setup for serial read/write
ser_line = connectToCOM('COM9', 9600)

print('Listening')

while (True):
    testListA, testListB = serBurstRead(ser_line,1)
    print('List A: ',testListA[:50], '\n', 'List B: ', testListB[:50])

    for i in range(len(testListA)):
        if testListA[i] > 4.5:
            ser_line.write('closeGrip'.encode())
            print('Close Grip Command Sent')
            while True:
                byte = ser_line.readline()
                mot = byte.decode('utf-8')
                print(mot[:-2])
                if mot[:-2] == 'Done':
                    break
            break
    for i in range(len(testListB)):
        if testListB[i] > 4.5:
            ser_line.write('openGrip'.encode())
            print('Open Grip Command Sent')
            while True:
                byte = ser_line.readline()
                mot = byte.decode('utf-8')
                print(mot[:-2])
                if mot[:-2] == 'Done':
                    break
            break

