"""
Trendline Identification Module

Automatically detects and draws optimal support/resistance trendlines,
and classifies the resulting chart pattern.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import numpy as np


class ChartPattern(Enum):
    """Recognized chart patterns from trendline analysis."""
    RISING_WEDGE = "Rising Wedge"
    FALLING_WEDGE = "Falling Wedge"
    BULLISH_CHANNEL = "Bullish Channel"
    BEARISH_CHANNEL = "Bearish Channel"
    TRIANGLE = "Triangle"
    EXPANDING_RANGE = "Expanding Range"
    RANGE = "Range"
    UNKNOWN = "Unknown"


@dataclass
class Trendline:
    """A trendline defined by slope and intercept."""
    slope: float
    intercept: float

    def get_value(self, x: int) -> float:
        """Get y value at position x."""
        return self.slope * x + self.intercept

    def get_line(self, length: int) -> np.ndarray:
        """Get array of y values for plotting."""
        return np.array([self.get_value(i) for i in range(length)])


@dataclass
class TrendlineResult:
    """Result of trendline detection."""
    pattern: ChartPattern
    support: Trendline
    resistance: Trendline
    best_fit: Trendline
    intercept_x: Optional[float] = None  # Where lines meet (if they do)


class TrendlineDetector:
    """Detects support and resistance trendlines in price data."""

    def __init__(self, lookback: int = 20):
        """
        Initialize detector.

        Args:
            lookback: Number of candles to analyze for trendlines
        """
        self.lookback = lookback

    def detect(
        self,
        close: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None
    ) -> TrendlineResult:
        """
        Detect trendlines in price data.

        Args:
            close: Array of closing prices
            high: Array of high prices (optional, uses close if None)
            low: Array of low prices (optional, uses close if None)

        Returns:
            TrendlineResult with support, resistance, and pattern classification
        """
        # Use close prices if high/low not provided
        high = high if high is not None else close
        low = low if low is not None else close

        # Ensure numpy arrays
        close = np.array(close)
        high = np.array(high)
        low = np.array(low)

        # Calculate line of best fit
        x = np.arange(len(close))
        best_fit_coefs = np.polyfit(x, close, 1)
        best_fit = Trendline(slope=best_fit_coefs[0], intercept=best_fit_coefs[1])

        # Find pivot points (furthest from best fit line)
        line_points = best_fit.get_line(len(close))
        upper_pivot = (high - line_points).argmax()
        lower_pivot = (low - line_points).argmin()

        # Optimize trendlines
        support = self._optimize_trendline(
            is_support=True,
            pivot=lower_pivot,
            init_slope=best_fit.slope,
            prices=low
        )

        resistance = self._optimize_trendline(
            is_support=False,
            pivot=upper_pivot,
            init_slope=best_fit.slope,
            prices=high
        )

        # Calculate where lines intercept
        slope_diff = support.slope - resistance.slope
        if abs(slope_diff) > 1e-10:
            intercept_x = (resistance.intercept - support.intercept) / slope_diff
        else:
            intercept_x = None

        # Classify pattern
        pattern = self._classify_pattern(support, resistance, intercept_x, len(close))

        return TrendlineResult(
            pattern=pattern,
            support=support,
            resistance=resistance,
            best_fit=best_fit,
            intercept_x=intercept_x
        )

    def _optimize_trendline(
        self,
        is_support: bool,
        pivot: int,
        init_slope: float,
        prices: np.ndarray
    ) -> Trendline:
        """
        Optimize trendline slope to minimize error while staying valid.

        A valid support line must be below all prices.
        A valid resistance line must be above all prices.
        """
        slope_unit = (prices.max() - prices.min()) / len(prices)

        opt_step = 1.0
        min_step = 0.001
        curr_step = opt_step

        best_slope = init_slope
        best_err = self._check_trendline(is_support, pivot, init_slope, prices)

        if best_err < 0:
            # Initial slope invalid, return basic line
            return Trendline(slope=init_slope, intercept=-init_slope * pivot + prices[pivot])

        get_derivative = True
        derivative = 0.0

        while curr_step > min_step:
            if get_derivative:
                # Numerical differentiation
                test_slope = best_slope + slope_unit * min_step
                test_err = self._check_trendline(is_support, pivot, test_slope, prices)
                derivative = test_err - best_err

                if test_err < 0:
                    test_slope = best_slope - slope_unit * min_step
                    test_err = self._check_trendline(is_support, pivot, test_slope, prices)
                    derivative = best_err - test_err

                if test_err < 0:
                    break  # Can't find valid derivative

                get_derivative = False

            # Move in direction that reduces error
            if derivative > 0:
                test_slope = best_slope - slope_unit * curr_step
            else:
                test_slope = best_slope + slope_unit * curr_step

            test_err = self._check_trendline(is_support, pivot, test_slope, prices)

            if test_err < 0 or test_err >= best_err:
                curr_step *= 0.5
            else:
                best_err = test_err
                best_slope = test_slope
                get_derivative = True

        intercept = -best_slope * pivot + prices[pivot]
        return Trendline(slope=best_slope, intercept=intercept)

    def _check_trendline(
        self,
        is_support: bool,
        pivot: int,
        slope: float,
        prices: np.ndarray
    ) -> float:
        """
        Check if trendline is valid and return error.

        Returns:
            Squared error sum if valid, -1.0 if invalid
        """
        intercept = -slope * pivot + prices[pivot]
        line_vals = slope * np.arange(len(prices)) + intercept
        diffs = line_vals - prices

        # Support must be below prices, resistance must be above
        if is_support and diffs.max() > 1e-5:
            return -1.0
        if not is_support and diffs.min() < -1e-5:
            return -1.0

        return (diffs ** 2).sum()

    def _classify_pattern(
        self,
        support: Trendline,
        resistance: Trendline,
        intercept_x: Optional[float],
        data_length: int
    ) -> ChartPattern:
        """Classify the chart pattern based on trendline slopes."""
        sup_slope = support.slope
        res_slope = resistance.slope

        # Both trending up
        if sup_slope > 0 and res_slope > 0:
            if intercept_x is not None and 0 < intercept_x < data_length:
                return ChartPattern.RISING_WEDGE
            elif intercept_x is not None and intercept_x < 0 and abs(intercept_x) < data_length:
                return ChartPattern.EXPANDING_RANGE
            return ChartPattern.BULLISH_CHANNEL

        # Both trending down
        if sup_slope < 0 and res_slope < 0:
            if intercept_x is not None and 0 < intercept_x < data_length:
                return ChartPattern.FALLING_WEDGE
            elif intercept_x is not None and intercept_x < 0 and abs(intercept_x) < data_length:
                return ChartPattern.EXPANDING_RANGE
            return ChartPattern.BEARISH_CHANNEL

        # Converging (support up, resistance down)
        if sup_slope > 0 and res_slope < 0:
            if intercept_x is not None and 0 < intercept_x < data_length * 2:
                return ChartPattern.TRIANGLE
            return ChartPattern.RANGE

        # Diverging (support down, resistance up)
        if sup_slope < 0 and res_slope > 0:
            return ChartPattern.EXPANDING_RANGE

        return ChartPattern.UNKNOWN
