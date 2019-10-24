import serial
import time
import threading
import queue


def read_com(serialport,baudrate,run_t):
    serial_port = serialport;
    baud_rate = baudrate;

    ser = serial.Serial(serial_port, baud_rate)

    timeout = time.time() + run_t
    counter = 0

    output_list = []
    while True:
        received = ser.readline()
        decoded = received.decode('utf-8')
        print(decoded)
        output_list.append(decoded)
        counter = counter + 1
        if time.time() > timeout:
            break
    q.put(output_list)



def out_com():
    num = q.get()
    print(num)



q = queue.Queue()
thread1 = threading.Thread(target=read_com, args=('COM9',19200,30))
thread2 = threading.Thread(target=out_com)

thread1.start()
thread2.start()