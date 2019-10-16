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
import math
from joblib import load,dump
from sklearn import tree

def connectToCOM(com_port='COM9', baud_rate=9600):
    serial_con = serial.Serial(com_port, baud_rate)
    return serial_con

def serFillBuff(ser_line, length):
    readBuffer = []
    val_utf8 = 'a'
    while len(readBuffer) < length:
        val_byte = ser_line.readline()
        try:
            val_utf8 = val_byte.decode('utf-8')
        except ValueError:
            False
        # except ValueError: False
        if val_utf8[0] == 'a':
            try: readBuffer.append(float(val_utf8[1:-2]))
            except ValueError: False
        # if val_utf8[0] == 'b':
        #     try: readbufferB.append(float(val_utf8[1:-2]))
        #     except ValueError: False
    return readBuffer #, readbufferB

def serBurstRead(ser_line, burst_length):
    readbufferA = []
    readbufferB = []
    currenttime = time.time()

    while time.time() < currenttime + burst_length:
        val_byte = ser_line.readline()
        try: val_utf8 = val_byte.decode('utf-8')
        except ValueError: False
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

def waitForMovement(ser_Line):
    while True:
        byte = ser_line.readline()
        utf = byte.decode('utf-8')
        print(utf.strip())
        if utf.strip() == 'Done':
            return

def calculateMAV(testList):
    sum = 0
    for i in testList:
        sum += i
    return sum/len(testList)

def calculateMAVS(testList):
    return testList[1] - testList[0]

def calculateSSI_VAR_RMS(testList):
    sum = 0
    for i in testList:
        sum += abs(i)**2
    return sum, sum/49, math.sqrt(sum/50)

def calculateWL(testList):
    sum = 0
    for i in range(1,len(testList)-1):
        sum += abs((testList[i]-(testList[i-1])))
    return sum


def readFromEMG(ser_line, inputList, letter):
    if len(inputList) < 49:
        return
    else:
        val_byte = ser_line.readline()
        val_utf8 = val_byte.decode('utf-8')
        if val_utf8[0] == letter:
            try:
                inputList.pop(0)
                inputList.append(float(val_utf8[1:-2]))
            except ValueError:
                False
        return inputList

def trigger(value):
    if value > 1.1: return 1
    else: return 0

## Setup for serial read/write
ser_line = connectToCOM('COM9', 19200)
model = load('expMod.joblib')
print('Listening')
burstLen = .50
testListA = serFillBuff(ser_line, 8)
mavList = [0]
counter = 0


print(testListA)

timeout = time.time() + 30
while (time.time() < timeout):

    testListA = readFromEMG(ser_line, testListA, 'a')
    print(testListA)

    if len(mavList) < 2:
        mavList.append(calculateMAV(testListA))
    else:
        mavList.pop(0)
        mavList.append(calculateMAV(testListA))
        lastMAVS = calculateMAVS(mavList)
        # print('MAVS: ', lastMAVS)
    SSI, VAR, RMS  = calculateSSI_VAR_RMS(testListA)
    treeList = [testListA[-1],trigger(testListA[-1]), calculateMAV(testListA), lastMAVS, SSI, VAR, RMS, calculateWL(testListA)]




    # print('SSI: ',SSI)
    # print('VAR: ', VAR)
    # print('RMS: ', RMS)
    # print('WL: ', calculateWL(testListA))

