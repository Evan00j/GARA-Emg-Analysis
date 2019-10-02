from biosppy import storage
from biosppy import plotting
import numpy as np
from biosppy.signals import emg
import fileinput
import time
import serial
import matplotlib.pyplot as plt
import datetime

def connectToCOM(com_port='COM9', baud_rate=9600):
    serial_con = serial.Serial(com_port, baud_rate)
    return serial_con


def serBurstRead(ser_line, burst_length):
    readbuffer = []
    currenttime = time.time()
    val_byte = ser_line.readline()
    val_utf8 = val_byte.decode('utf-8')

    while time.time() < currenttime + burst_length:
        readbuffer.append(float(val_utf8))
    return readbuffer

def serBurstReadTest(burst_length):
    readbuffer = []
    currenttime = time.time()

    while time.time() < currenttime + burst_length:
        readbuffer.append(datetime.date.today())
        time.sleep(.2)
    return readbuffer

## Setup for serial read/write
##ser_line = connectToCOM('COM9', 9600)

while (True):
    testList = serBurstReadTest(2)
    print(len(testList), ' ', testList, '\n')