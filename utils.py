import os
import numpy as np
import pandas as pd
from sliding_window import sliding_window


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    print("Printing the data shapes after the sliding window function")
    print(data_x.shape)
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    print(data_y.shape)
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def opp_sliding_windowX(data_x, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    return data_x.astype(np.float32)


def opp_sliding_windowY(data_y, ws, ss):
    data_y = np.asarray(
        [[i[-1]] for i in sliding_window(data_y, ws, ss)]
    )  # Essentially getting the last label from the time segment
    return data_y.reshape(len(data_y)).astype(np.uint8)


def standardize_classes(x):
    if x == 1:
        return 0
    if x == 2:
        return 1
    elif x == 3:
        return 2
    elif x == 4:
        return 3
    elif x == 5:
        return 4
    elif x == 6:
        return 5
    elif x == 7:
        return 6
    elif x == 8:
        return 7
    elif x == 13:
        return 8
    elif x == 14:
        return 9
    elif x == 130:
        return 10
    elif x == 140:
        return 11
