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
