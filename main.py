# ----< Global Imports >----
from AlgorithmImports import *
from harmonics import *
from levels import *
from trendlines import *


class JumpingBlackWhale(QCAlgorithm):

    def Initialize(self):

        # Backtest Settings
        self.SetStartDate(2008, 1, 1)
        self.SetEndDate(2010, 1, 1)
        self.SetCash(10000)
        self.lookback = 30

        # Instrument
        self.symbol = 'SPY'
        self.AddEquity(self.symbol, Resolution.Minute)

        # Indicators
        self.SMA = SimpleMovingAverage(self.symbol, 21)
        self.SMA_fast = SimpleMovingAverage(self.symbol, 20)
        self.STR = SuperTrend(20, 2, MovingAverageType.Wilders)
        self.ATR = AverageTrueRange(self.symbol, 20)

        # Consolidator
        self.cons = TradeBarConsolidator(timedelta(days=1))
        self.cons.DataConsolidated += self.consolidation_update
        self.SubscriptionManager.AddConsolidator(self.symbol, self.cons)

        # Register Indicator
        self.RegisterIndicator(self.symbol, self.SMA_fast, self.cons)
        self.RegisterIndicator(self.symbol, self.STR, self.cons)
        self.RegisterIndicator(self.symbol, self.ATR, self.cons)

        # Rolling Windows
        self.close_prices_window = RollingWindow[float](self.lookback)
        self.high_prices_window = RollingWindow[float](self.lookback)
        self.low_prices_window = RollingWindow[float](self.lookback)
        self.index_window = RollingWindow[DateTime](self.lookback)
        self.closest_levels = RollingWindow[float](self.lookback)

        # Initialize Modules
        self.trend = Trendline_Identification(self.lookback)
        self.harm = HarmonicsIdentification(self, 0.01)

        # Helpers
        self.series = Chart("Chart")
        self.circle = ScatterMarkerSymbol.Circle

        # Adding formatted series
        self.series.AddSeries(
            Series("Buy", SeriesType.Scatter, "$", Color.Green, self.circle)
            )
        self.series.AddSeries(
            Series("Sell", SeriesType.Scatter, "$", Color.Red, self.circle)
            )
        self.series.AddSeries(
            Series("Harmonic Sell", SeriesType.Scatter, "$", Color.Yellow, self.circle)
            )
        
        self.AddChart(self.series)

        
        self.warmup = 0

        self.quantity = 0
        self.stop = None
        self.bought = None
        self.sell_allowed = False

    
    def find_closest_number(self, array, x, atr):
        closest_number = float('-inf')

        new_array = []
        for i in array:
            new_array.append(i+atr)
        array = new_array
        
        for num in array:
            if num < x and num > closest_number:
                closest_number = num
        if closest_number == float('-inf'):
            return x
        return closest_number

    def OnData(self, data: Slice):
        pass

    def OnOrderEvent(self, orderEvent: object):
        self.quantity += orderEvent.FillQuantity
        if orderEvent.FillQuantity < 0:
            self.Plot("Chart", "Sell", orderEvent.FillPrice)

    def consolidation_update(self, sender: Any, bar: object) -> None:
        
        # Populate rolling windows
        self.index_window.Add(bar.EndTime)
        self.close_prices_window.Add(bar.Close)
        self.high_prices_window.Add(bar.High)
        self.low_prices_window.Add(bar.Low)

        self.Plot("Chart", "Price", bar.Close)

        if self.warmup > self.lookback:
        
            # ----< Initialize Market-View Modules >----
            list_index = []
            list_close = []
            list_low   = []
            list_high  = []

            # Convert rolling windows to lists
            for x in self.index_window:
                list_index.append(x)
            for x in self.close_prices_window:
                list_close.append(x)
            for x in self.low_prices_window:
                list_low.append(x)
            for x in self.high_prices_window:    
                list_high.append(x)
            
            # Convert all lists to pd.Series
            df_index = pd.Series(list_index[::-1])
            df_close = pd.Series(list_close[::-1])
            df_low   = pd.Series(list_low[::-1])
            df_high  = pd.Series(list_high[::-1])

            # Convert to one dataframe
            subset = pd.DataFrame({'High': df_high, 'Close': df_close, 'Low': df_low})
            subset.index = df_index

            # Modules
            pattern, best_fit_coefs, support_coefs, resist_coefs = self.trend.fit_trendlines(df_close)
            UTILS_, XABCD_, best_pat, ad = self.harm.harmonics(subset)
            levels, peaks, props, price_range, pdf, weights = find_levels(np.array(subset['Close']), self.ATR.Current.Value)
            self.harm.call_after_plotting(ad)

            self.closest_levels.Add(self.find_closest_number(levels, bar.Close, self.ATR.Current.Value))

            if best_pat and "Bullish" in best_pat:
                self.Plot("Chart", "Buy", bar.Close)
                self.SetHoldings(self.symbol, 1)
                self.sell_allowed = False
            
            if best_pat and "Bearish" in best_pat and "bullish" in pattern or "rising" in pattern:
                self.sell_allowed = True
                self.Plot("Chart", "Harmonic Sell", bar.Close)
                # self.SetHoldings(self.symbol, 0)
                
            if self.sell_allowed and self.stop and bar.Close < self.stop:
                self.SetHoldings(self.symbol, 0)
                self.Plot("Chart", "Sell", bar.Close)
                self.sell_allowed = False
            
            if self.stop:
                self.Plot("Chart", "stop", self.stop)
            
            if self.closest_levels.IsReady:
                self.stop = self.closest_levels[0]
                if self.closest_levels[1] < self.closest_levels[0]:
                    self.stop = self.closest_levels[1]
                if bar.Close < self.stop:
                    self.stop = self.ATR.Current.Value

        self.warmup = self.warmup + 1