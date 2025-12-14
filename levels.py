# ----< Global Imports >----
from AlgorithmImports import *
import numpy as np
import scipy
from scipy.signal import find_peaks
import pandas as pd
import timeit


def find_levels( 
        price: np.array, atr: float, # Log closing price, and log atr 
        first_w: float = 0.1, # The last price point has only 10% of the weight
        atr_mult: float = 1.0, 
        prom_thresh: float = 0.1
        ):

    # Setup weights
    last_w = 1.0 # the newest data has the most weight
    w_step = (last_w - first_w) / len(price)
    weights = first_w + np.arange(len(price)) * w_step
    weights[weights < 0] = 0.0

    # Get kernel of price. 
    kernal = scipy.stats.gaussian_kde(price, bw_method=atr*atr_mult, weights=weights)

    # Construct market profile
    min_v = np.min(price)
    max_v = np.max(price)

    step = (max_v - min_v) / 200
    price_range = np.arange(min_v, max_v, step)
    
    pdf = kernal(price_range) # Market profile

    # Find significant peaks in the market profile
    pdf_max = np.max(pdf)
    prom_min = pdf_max * prom_thresh

    peaks, props = scipy.signal.find_peaks(pdf, prominence=prom_min)
    levels = [] 
    for peak in peaks:
        levels.append(price_range[peak])

    return levels, peaks, props, price_range, pdf, weights


