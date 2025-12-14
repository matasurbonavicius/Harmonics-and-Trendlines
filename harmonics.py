# ----< Global Imports >----
from AlgorithmImports import *
from dataclasses import dataclass
from collections import deque
from typing import Union
import pandas as pd
import numpy as np
import itertools


class HarmonicsIdentification():

    @dataclass
    class XABCD_RATIOS:

        """
        Description:
            A data class for holding information about the pattern
        """

        XB: Union[float, list, None]
        AC: Union[float, list, None]
        DB: Union[float, list, None]
        XD: Union[float, list, None]
        name: str

    @dataclass
    class XABCD:

        """
        Description:
            A data class which holds information of the current pattern, its values
        """
        
        # Index on the chart, that is accepted by mplfinance
        # If there currently is 100 candles on the chart, and D is the
        # last data point, then D will be 100.
        X: float
        D: float
        C: float
        B: float
        A: float

        # Prices of each point
        X_price: float
        D_price: float
        C_price: float
        B_price: float
        A_price: float

        # Retracement ratios between key points
        DB_ratio: float
        AC_ratio: float
        XB_ratio: float
        XD_ratio: float

        def __init__(self):
            
            # Reset Variables with each initialization
            self.X = None
            self.D = None
            self.C = None
            self.B = None
            self.A = None

            self.X_price = None
            self.D_price = None
            self.C_price = None
            self.B_price = None
            self.A_price = None

            self.DB_ratio = None
            self.AC_ratio = None
            self.XB_ratio = None
            self.XD_ratio = None

    @dataclass
    class UTILS:

        """
        Description:
            A data class which holds data that is used for plotting
        """
        
        bottoms_ratios: list # List of indexes for the bottom points
        tops_ratios:    list # List of indexes for the upper points
        
        # Tuples that are accepted by mplfinance for plotting a tline
        indexes_pairs:        tuple
        top_ratios_pairs:     tuple  
        bottoms_ratios_pairs: tuple 
        xd_line:              tuple 

        def __init__(self):

            # Reset Variables with each initialization        
            self.bottoms_ratios = deque(maxlen=3)  
            self.tops_ratios    = deque(maxlen=3) 
            
            self.indexes_pairs        = None  
            self.top_ratios_pairs     = None  
            self.bottoms_ratios_pairs = None 
            self.xd_line              = None 


    def __init__(self, algo, zigzag_sigma: float, err_thresh: float = 1.2) -> None:
        
        """
        Description:
            This module identifies if there currently is a harmonics pattern visible in a given set of data

        Arguments:
            *zigzag_sigma: a percentage change in price which must occur for the zigzag to record a top or a bottom

        """

        self.algo = algo
        self.zigzag_sigma = zigzag_sigma
        self.indexes = deque(maxlen=5)
        self.err_thresh = err_thresh

        # Define Patterns
        self.GARTLEY = self.XABCD_RATIOS(0.618, [0.382, 0.886], [1.13, 1.618], 0.786, "Gartley")
        self.BAT = self.XABCD_RATIOS([0.382, 0.50], [0.382, 0.886], [1.618, 2.618], 0.886, "Bat")
        self.BUTTERFLY = self.XABCD_RATIOS(0.786, [0.382, 0.886], [1.618, 2.24], [1.27, 1.41], "Butterfly")
        self.CRAB = self.XABCD_RATIOS([0.382, 0.618], [0.382, 0.886], [2.618, 3.618], 1.618, "Crab")
        self.DEEP_CRAB = self.XABCD_RATIOS(0.886, [0.382, 0.886], [2.0, 3.618], 1.618, "Deep Crab")
        self.CYPHER = self.XABCD_RATIOS([0.382, 0.618], [1.13, 1.41], [1.27, 2.00], 0.786, "Cypher")
        self.ALL_PATTERNS = [self.GARTLEY, self.BAT, self.BUTTERFLY, self.CRAB, self.DEEP_CRAB, self.CYPHER]

    def zigzag(self, close: np.array, high: np.array, low: np.array):
        
        """ 
        Description:
            Function is a zigzag indicator which draws ups and downs of the market.
            For each turning point to occur price must move at least the % amount of sigma
        
        Arguments:
            *close: np.array of close prices
            *high: np.array of close prices
            *low: np.array of close prices
        
        Returns:
            tops: a list of all identified tops in the given set of data
            bottoms: a list of all identified tops in the given set of data
        """

        up_zig = True # Last extreme is a bottom. Next is a top. 
        tmp_max = high[0]
        tmp_min = low[0]
        tmp_max_i = 0
        tmp_min_i = 0

        tops = []
        bottoms = []

        for i in range(len(close)):
            if up_zig: # Last extreme is a bottom
                if high[i] > tmp_max:
                    # New high, update 
                    tmp_max = high[i]
                    tmp_max_i = i
                elif close[i] < tmp_max - tmp_max * self.zigzag_sigma: 
                    # Price retraced by sigma %. Top confirmed, record it
                    # top[0] = confirmation index
                    # top[1] = index of top
                    # top[2] = price of top
                    top = [i, tmp_max_i, tmp_max]
                    tops.append(top)

                    # Setup for next bottom
                    up_zig = False
                    tmp_min = low[i]
                    tmp_min_i = i
            else: # Last extreme is a top
                if low[i] < tmp_min:
                    # New low, update 
                    tmp_min = low[i]
                    tmp_min_i = i
                elif close[i] > tmp_min + tmp_min * self.zigzag_sigma: 
                    # Price retraced by sigma %. Bottom confirmed, record it
                    # bottom[0] = confirmation index
                    # bottom[1] = index of bottom
                    # bottom[2] = price of bottom
                    bottom = [i, tmp_min_i, tmp_min]
                    bottoms.append(bottom)

                    # Setup for next top
                    up_zig = True
                    tmp_max = high[i]
                    tmp_max_i = i

        return tops, bottoms


    def get_error(self, required_ratio: Union[float, list], current_ratio: float) -> bool:

        """
        Description:
            Function checks the error between the required ratio, which is found in
            pattern definitions and the current ratio, found in the market from one zigzag
            point to another.

        Arguments:
            *required_ratio: Union[float, list] Is a ratio that is required for the given pattern
                if required ratio is a list, the lenght must be equal to 2
            *current_ratio: float Is a ratio that is currently observed in the market between given zigzag points

        Returns:
            The absolute difference between two ratios. If required ratio is a list, then returns the smaller error from them
        """

        if isinstance(required_ratio, list):
            
            # If ratio is a list where one of the values are None, then only compare the value thats not None
            if required_ratio[0] is None:
                return abs(required_ratio[0]-current_ratio)
            elif required_ratio[1] is None:
                return abs(required_ratio[1]-current_ratio)
            else:
                ValueError('Required Ratio in get_error function is all None')
            
            # If current ratio falls between the bounds, the error is 0
            if min([required_ratio[0], required_ratio[1]]) < current_ratio and current_ratio < max([required_ratio[0], required_ratio[1]]):
                return 0
            # Otherwise return the minimum error from the difference of the closest bound
            else:
                return min([abs(required_ratio[0]-current_ratio), abs(required_ratio[1]-current_ratio)])
        
        # If required ratio for that data point is None, then there is no error, thus return 0
        if required_ratio is None:
            return 0
        
        # In all other cases return the absolute difference between ratios
        return abs(current_ratio-required_ratio)

    def call_after_plotting(self, added_last_value):
        if added_last_value:
            self.indexes.pop()
            added_last_value = False


    def harmonics(self, subset: pd.DataFrame, return_tuples: bool = False):
        
        XABCD_ = self.XABCD()
        UTILS_ = self.UTILS()

        close = subset['Close']
        high = subset['High']
        low = subset['Low']

        ad = False

        tops, bottoms = self.zigzag(np.array(close), np.array(high), np.array(low))

        if len(tops) > 0 and len(bottoms) > 0 and len(subset) > 0:

            # Index of the last top and the last bottom
            tops_date = subset.iloc[tops[-1][1]].name
            bottoms_date = subset.iloc[bottoms[-1][1]].name

            # if this last top/bottom is currently not recorded- record it
            self.indexes.append(tops_date) if tops_date not in self.indexes else None
            self.indexes.append(bottoms_date) if bottoms_date not in self.indexes else None
            
            # Also add the last bar
            if (subset.iloc[-1].name) not in self.indexes:
                self.indexes.append((subset.iloc[-1].name))
                ad = True

            # Show ratio (retracement) lines from peaks of zigzag
            for l, top in enumerate(tops):
                top_date = subset.iloc[tops[l][1]].name
                if top_date not in UTILS_.tops_ratios:
                    if len(UTILS_.tops_ratios) > 0 and top_date > UTILS_.tops_ratios[-1]:
                        UTILS_.tops_ratios.append(top_date)
                    elif len(UTILS_.tops_ratios) == 0:
                        UTILS_.tops_ratios.append(top_date)
        
            for l, bottom in enumerate(bottoms):
                bottom_date = subset.iloc[bottoms[l][1]].name
                if bottom_date not in UTILS_.bottoms_ratios:
                    if len(UTILS_.bottoms_ratios) > 0 and bottom_date > UTILS_.bottoms_ratios[-1]:
                        UTILS_.bottoms_ratios.append(bottom_date)
                    elif len(UTILS_.bottoms_ratios) == 0:
                        UTILS_.bottoms_ratios.append(bottom_date)
                
            # Extent ratio line to the very last candle
            if UTILS_.tops_ratios[-1] < UTILS_.bottoms_ratios[-1]:
                if subset.iloc[-1].name not in UTILS_.tops_ratios:
                    UTILS_.tops_ratios.append(subset.iloc[-1].name)
                UTILS_.bottoms_ratios.popleft()
            else:
                if subset.iloc[-1].name not in UTILS_.bottoms_ratios:
                    UTILS_.bottoms_ratios.append(subset.iloc[-1].name)
                UTILS_.tops_ratios.popleft()

            all_candles = len(subset)

            # Get the index of each point (DateTime)
            D_index = self.indexes[-1] if len(self.indexes) > 0 else None
            C_index = self.indexes[-2] if len(self.indexes) > 1 else None
            B_index = self.indexes[-3] if len(self.indexes) > 2 else None
            A_index = self.indexes[-4] if len(self.indexes) > 3 else None
            X_index = self.indexes[-5] if len(self.indexes) > 4 else None

            XABCD_.D = subset.loc[D_index]
            XABCD_.C = (subset.index.get_loc(C_index)+1) if (C_index and C_index in subset.index) else None
            XABCD_.B = (subset.index.get_loc(B_index)+1) if (B_index and B_index in subset.index) else None
            XABCD_.A = (subset.index.get_loc(A_index)+1) if (A_index and A_index in subset.index) else None
            XABCD_.X = (subset.index.get_loc(X_index)+1) if (X_index and X_index in subset.index) else None

            # Fetch its price
            XABCD_.D_price = close[-1]
            if XABCD_.C:
                XABCD_.C_price = subset['Low' if (XABCD_.D_price > close[-int(all_candles-XABCD_.C+1)]) else 'High'][-int(all_candles-XABCD_.C+1)]
            else:
                XABCD_.C_price = None
            
            if XABCD_.B and XABCD_.C:
                XABCD_.B_price = subset['Low' if (XABCD_.D_price < close[-int(all_candles-XABCD_.C+1)]) else 'High'][-int(all_candles-XABCD_.B+1)]
            else:
                XABCD_.B_price = None
            
            if XABCD_.A and XABCD_.C:
                XABCD_.A_price = subset['Low' if (XABCD_.D_price > close[-int(all_candles-XABCD_.C+1)]) else 'High'][-int(all_candles-XABCD_.A+1)]
            else:
                None
            
            if XABCD_.X and XABCD_.C:
                XABCD_.X_price = subset['Low' if (XABCD_.D_price < close[-int(all_candles-XABCD_.C+1)]) else 'High'][-int(all_candles-XABCD_.X+1)]
            else:
                XABCD_.X_price = None

            DC_h = abs(XABCD_.D_price - XABCD_.C_price) if (XABCD_.D_price and XABCD_.C_price) else None
            CB_h = abs(XABCD_.C_price - XABCD_.B_price) if (XABCD_.C_price and XABCD_.B_price) else None
            BA_h = abs(XABCD_.B_price - XABCD_.A_price) if (XABCD_.B_price and XABCD_.A_price) else None
            AX_h = abs(XABCD_.A_price - XABCD_.X_price) if (XABCD_.A_price and XABCD_.X_price) else None

            XABCD_.DB_ratio = (DC_h / CB_h) if CB_h else None
            XABCD_.AC_ratio = (CB_h / BA_h) if BA_h else None
            XABCD_.XB_ratio = (BA_h / AX_h) if AX_h else None
            try:
                XABCD_.XD_ratio = (AX_h / DC_h) if AX_h else None
            except:
                XABCD_.XD_ratio = 0
            
            best_err = 1e30
            best_pat = None
            if XABCD_.DB_ratio and XABCD_.AC_ratio and XABCD_.XB_ratio and XABCD_.XD_ratio:

                for pat in self.ALL_PATTERNS:
                    err = 0.0
                    err += self.get_error(pat.DB, XABCD_.DB_ratio)
                    err += self.get_error(pat.AC, XABCD_.AC_ratio)
                    err += self.get_error(pat.XB, XABCD_.XB_ratio)
                    err += self.get_error(pat.XD, XABCD_.XD_ratio)
    
                    if err < best_err:
                        best_err = err
                        best_pat = pat.name
                
                if best_err <= self.err_thresh:
                    if XABCD_.D_price > XABCD_.C_price:
                        best_pat = f'Bearish {best_pat}'
                    else:
                        best_pat = f'Bullish {best_pat}'
                else:
                    best_pat = None
            
            if not return_tuples:
                return UTILS_, XABCD_, best_pat, ad
            else:
                if len(self.indexes) > 1:

                    UTILS_.indexes_pairs = list(zip(self.indexes, itertools.islice(self.indexes, 1, None)))
                    UTILS_.top_ratios_pairs = list(zip(UTILS_.tops_ratios, itertools.islice(UTILS_.tops_ratios, 1, None)))
                    UTILS_.bottoms_ratios_pairs = list(zip(UTILS_.bottoms_ratios, itertools.islice(UTILS_.bottoms_ratios, 1, None)))
                    UTILS_.xd_line = list(zip([self.indexes[0]], [self.indexes[-1]]))
                
                    return UTILS_, XABCD_, best_pat, ad
                else:
                    return UTILS_, XABCD_, best_pat, ad
        
        if not return_tuples:
            return UTILS_, XABCD_, None, ad
        else:
            return UTILS_, XABCD_, None, ad