import serial
import time

serial_port ='COM9';
baud_rate = 19200;
read_from_file_path = 'moveout.txt'
write_to_file_path = 'newflat.txt';

input_file = open(read_from_file_path, "r")
output_file = open(write_to_file_path, 'w+')
ser = serial.Serial(serial_port, baud_rate)

timeout = time.time() + 30
counter = 0

while True:
    received = ser.readline()
    decoded = received.decode('utf-8')
    print(decoded)
    output_file.write(decoded)
    counter = counter + 1
    if time.time() > timeout:
        output_file.write(str(counter))
        break
#for lines in input_file:
#    if lines > str(1.86):
#        output_file.write(lines)
#    elif lines == '\n':
#        continue
#    elif lines == '':
#        continue
#    else:
#        output_file.write(str(1.65)+'\n')
###
#for lines in output_file:
#    if type(lines) == str:
#        output_file.write(lines)
