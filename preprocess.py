import os
import numpy as np
import pandas as pd
from datetime import datetime


import matplotlib.pyplot as plt
def load_data(folder='activity', seq_length=None, threshold=60, normalised = False):
    if folder[-1] != '/':
        folder += '/'
    if normalised:
        threshold /= 86400.

    users = []
    data_files = os.listdir(folder)

    for data_file in data_files:
        data = pd.read_csv(folder+data_file)
        activity = np.asarray(data[' activity inference'])

        dt_objects = [datetime.fromtimestamp(t) for t in data['timestamp']]
        if not normalised:
            time = [dt.hour*3600+dt.minute*60+dt.second for dt in dt_objects]
        else:
            time = [(dt.hour*3600+dt.minute*60+dt.second) / 86400. for dt in dt_objects]

        length = 0
        num_days = 1
        sequence = []
        for ind,_ in enumerate(time):
            if ind > 0 and dt_objects[ind].day != dt_objects[ind-1].day:
                num_days += 1

            if ind > 0 and abs(time[ind]-time[ind-1]) > threshold:
                mean_sec = np.mean(time[ind-length:ind])

                act, cnts = np.unique(activity[ind-length:ind], return_counts=True)
                sequence.append([mean_sec, act[np.argmax(cnts)]])

                length = 1
            
            else:
                length += 1
                if seq_length != None and length == seq_length:
                    mean_sec = np.mean(time[0,ind-length+1:ind+1])
                    act, cnts = np.unique(activity[ind-length+1:ind+1], return_counts=True)
                    sequence.append(mean_sec, act[np.argmax(cnts)])
                    length = 0
                if seq_length == None and ind == len(time)-1:
                    mean_sec = np.mean(time[ind-length+1:ind+1])
                    act, cnts = np.unique(activity[ind-length+1:ind+1], return_counts=True)
                    sequence.append([mean_sec, act[np.argmax(cnts)]])

        users.append([sequence, num_days])

    return users

