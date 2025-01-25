import collections
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import uctc.nn as nn

use_graphics = True

def maybe_sleep_and_close(seconds):
    if use_graphics and plt.get_fignums():
        time.sleep(seconds)
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            plt.close(fig)
            try:
                # This raises a TclError on some Windows machines
                fig.canvas.start_event_loop(1e-3)
            except:
                pass

def get_data_path(filename):
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise Exception("Could not find data file: {}".format(filename))
    return path

