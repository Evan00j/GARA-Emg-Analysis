from biosppy import storage
from biosppy import plotting
import numpy as np
from biosppy.signals import emg
import fileinput
import matplotlib.pyplot as plt

for line in fileinput.FileInput("moveout.txt", inplace=1):
    if line.replace("\n",""):
        print(line)
rawsignal = np.loadtxt("moveout.txt")


output = emg.emg(signal=rawsignal, sampling_rate=281., show=True)


for i in range(len(output['ts'])):
    for j in range(len(output['onsets'])):
        if i == output['onsets'][j]:
            print(output['ts'][i])
#print(output['onsets'][1])


#output = plotting.plot_emg(ts=10, sampling_rate=248., raw=rawsignal, filtered=filteredsignal,