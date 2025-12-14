"""
Harmonic Pattern Identification Module

Identifies classic harmonic patterns (Gartley, Bat, Butterfly, Crab, etc.)
using zigzag pivot points and Fibonacci ratios.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from collections import deque
import numpy as np
import pandas as pd
import itertools


@dataclass
class HarmonicPattern:
    """Definition of a harmonic pattern with required Fibonacci ratios."""
    name: str
    xb: Union[float, List[float]]  # X to B retracement
    ac: Union[float, List[float]]  # A to C retracement
    bd: Union[float, List[float]]  # B to D extension
    xd: Union[float, List[float]]  # X to D retracement


@dataclass
class PatternPoints:
    """XABCD pattern points with prices and chart positions."""
    x_idx: Optional[int] = None
    a_idx: Optional[int] = None
    b_idx: Optional[int] = None
    c_idx: Optional[int] = None
    d_idx: Optional[int] = None

    x_price: Optional[float] = None
    a_price: Optional[float] = None
    b_price: Optional[float] = None
    c_price: Optional[float] = None
    d_price: Optional[float] = None

    xb_ratio: Optional[float] = None
    ac_ratio: Optional[float] = None
    bd_ratio: Optional[float] = None
    xd_ratio: Optional[float] = None


@dataclass
class PatternResult:
    """Result of pattern detection."""
    name: Optional[str] = None
    direction: Optional[str] = None  # "Bullish" or "Bearish"
    error: float = float('inf')
    points: PatternPoints = field(default_factory=PatternPoints)
    lines: List[Tuple] = field(default_factory=list)

    @property
    def full_name(self) -> Optional[str]:
        if self.name and self.direction:
            return f"{self.direction} {self.name}"
        return None


class HarmonicDetector:
    """Detects harmonic patterns in price data."""

    # Classic harmonic patterns
    PATTERNS = [
        HarmonicPattern("Gartley",    0.618,         [0.382, 0.886], [1.13, 1.618],  0.786),
        HarmonicPattern("Bat",        [0.382, 0.50], [0.382, 0.886], [1.618, 2.618], 0.886),
        HarmonicPattern("Butterfly",  0.786,         [0.382, 0.886], [1.618, 2.24],  [1.27, 1.41]),
        HarmonicPattern("Crab",       [0.382, 0.618],[0.382, 0.886], [2.618, 3.618], 1.618),
        HarmonicPattern("Deep Crab",  0.886,         [0.382, 0.886], [2.0, 3.618],   1.618),
        HarmonicPattern("Cypher",     [0.382, 0.618],[1.13, 1.41],   [1.27, 2.00],   0.786),
    ]

    def __init__(self, ratio_tolerance: float = 0.15):
        """
        Initialize detector.

        Args:
            ratio_tolerance: Maximum allowed deviation from ideal Fibonacci ratios
        """
        self.ratio_tolerance = ratio_tolerance
        self._pivots = deque(maxlen=6)

    def detect(self, data: pd.DataFrame, sigma: Optional[float] = None) -> PatternResult:
        """
        Detect harmonic patterns in OHLC data.

        Args:
            data: DataFrame with 'Open', 'High', 'Low', 'Close' columns
            sigma: Zigzag sensitivity (auto-calculated if None)

        Returns:
            PatternResult with detected pattern info
        """
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values

        # Calculate optimal zigzag sensitivity
        if sigma is None:
            sigma = self._calculate_sigma(close, high, low)

        # Find zigzag pivots
        tops, bottoms = self._zigzag(sigma, close, high, low)

        if not tops or not bottoms:
            return PatternResult()

        # Build pivot sequence
        pivots = self._build_pivot_sequence(data, tops, bottoms)

        if len(pivots) < 5:
            return PatternResult()

        # Extract XABCD points
        points = self._extract_points(data, pivots)

        if not self._has_valid_points(points):
            return PatternResult()

        # Calculate ratios
        self._calculate_ratios(points)

        # Find best matching pattern
        result = self._match_pattern(points)
        result.lines = self._build_lines(pivots)

        return result

    def _calculate_sigma(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate optimal zigzag sigma based on ATR."""
        atr = np.mean(np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )[1:])

        base_sigma = atr / close[-1]

        # Find optimal sigma through error minimization
        best_sigma = base_sigma
        best_score = float('inf')

        for mult in np.arange(1, 15, 0.5):
            test_sigma = base_sigma * mult
            tops, bottoms, _, error = self._zigzag_with_error(test_sigma, close, high, low)
            pivot_count = len(tops) + len(bottoms)

            if pivot_count >= 4:
                score = error / (pivot_count ** 0.5)
                if score < best_score:
                    best_score = score
                    best_sigma = test_sigma

        return best_sigma

    def _zigzag(self, sigma: float, close: np.ndarray, high: np.ndarray, low: np.ndarray):
        """
        Calculate zigzag indicator.

        Returns:
            tops: List of (confirmation_idx, peak_idx, price)
            bottoms: List of (confirmation_idx, trough_idx, price)
        """
        tops, bottoms, _, _ = self._zigzag_with_error(sigma, close, high, low)
        return tops, bottoms

    def _zigzag_with_error(self, sigma: float, close: np.ndarray, high: np.ndarray, low: np.ndarray):
        """Zigzag calculation with error tracking."""
        looking_for_top = True
        tmp_max, tmp_min = high[0], low[0]
        tmp_max_i, tmp_min_i = 0, 0

        tops, bottoms = [], []
        pivots = [(0, 0, close[0])]

        for i in range(len(close)):
            if looking_for_top:
                if high[i] > tmp_max:
                    tmp_max, tmp_max_i = high[i], i
                elif close[i] < tmp_max * (1 - sigma):
                    tops.append((i, tmp_max_i, tmp_max))
                    pivots.append((i, tmp_max_i, tmp_max))
                    looking_for_top = False
                    tmp_min, tmp_min_i = low[i], i
            else:
                if low[i] < tmp_min:
                    tmp_min, tmp_min_i = low[i], i
                elif close[i] > tmp_min * (1 + sigma):
                    bottoms.append((i, tmp_min_i, tmp_min))
                    pivots.append((i, tmp_min_i, tmp_min))
                    looking_for_top = True
                    tmp_max, tmp_max_i = high[i], i

        pivots.append((len(close)-1, len(close)-1, close[-1]))

        # Calculate error
        pivot_prices = [p[2] for p in pivots]
        pivot_indices = [p[1] for p in pivots]
        zigzag_line = np.interp(np.arange(len(close)), pivot_indices, pivot_prices)
        error = np.sum(np.abs(close - zigzag_line))

        return tops, bottoms, pivots, error

    def _build_pivot_sequence(self, data: pd.DataFrame, tops: list, bottoms: list) -> deque:
        """Build chronologically sorted pivot sequence."""
        pivots = deque(maxlen=6)

        # Convert to datetime indices
        top_dates = [(data.index[t[1]], 'top', t[2]) for t in tops]
        bottom_dates = [(data.index[b[1]], 'bottom', b[2]) for b in bottoms]

        all_pivots = sorted(top_dates + bottom_dates, key=lambda x: x[0])

        for pivot in all_pivots[-5:]:
            pivots.append(pivot)

        # Add current price as potential D point
        last_pivot_type = all_pivots[-1][1] if all_pivots else 'bottom'
        pivots.append((data.index[-1], 'top' if last_pivot_type == 'bottom' else 'bottom', data['Close'].iloc[-1]))

        return pivots

    def _extract_points(self, data: pd.DataFrame, pivots: deque) -> PatternPoints:
        """Extract XABCD points from pivots."""
        points = PatternPoints()
        pivot_list = list(pivots)

        if len(pivot_list) >= 5:
            # Get indices relative to data
            points.x_idx = data.index.get_loc(pivot_list[-5][0]) if pivot_list[-5][0] in data.index else None
            points.a_idx = data.index.get_loc(pivot_list[-4][0]) if pivot_list[-4][0] in data.index else None
            points.b_idx = data.index.get_loc(pivot_list[-3][0]) if pivot_list[-3][0] in data.index else None
            points.c_idx = data.index.get_loc(pivot_list[-2][0]) if pivot_list[-2][0] in data.index else None
            points.d_idx = data.index.get_loc(pivot_list[-1][0]) if pivot_list[-1][0] in data.index else None

            # Get prices
            points.x_price = pivot_list[-5][2]
            points.a_price = pivot_list[-4][2]
            points.b_price = pivot_list[-3][2]
            points.c_price = pivot_list[-2][2]
            points.d_price = pivot_list[-1][2]

        return points

    def _has_valid_points(self, points: PatternPoints) -> bool:
        """Check if all XABCD points are valid."""
        return all([
            points.x_price is not None,
            points.a_price is not None,
            points.b_price is not None,
            points.c_price is not None,
            points.d_price is not None,
        ])

    def _calculate_ratios(self, points: PatternPoints):
        """Calculate Fibonacci ratios between points."""
        xa = abs(points.a_price - points.x_price)
        ab = abs(points.b_price - points.a_price)
        bc = abs(points.c_price - points.b_price)
        cd = abs(points.d_price - points.c_price)

        points.xb_ratio = ab / xa if xa != 0 else float('inf')
        points.ac_ratio = bc / ab if ab != 0 else float('inf')
        points.bd_ratio = cd / bc if bc != 0 else float('inf')
        points.xd_ratio = cd / xa if xa != 0 else float('inf')

    def _ratio_error(self, required: Union[float, List[float]], actual: float) -> float:
        """Calculate error between required and actual ratio."""
        if required is None:
            return 0

        if isinstance(required, list):
            if min(required) <= actual <= max(required):
                return 0
            return min(abs(r - actual) for r in required)

        return abs(required - actual)

    def _match_pattern(self, points: PatternPoints) -> PatternResult:
        """Find best matching harmonic pattern."""
        best_result = PatternResult(points=points)

        for pattern in self.PATTERNS:
            error = 0

            # Check each ratio
            for req, act in [
                (pattern.xb, points.xb_ratio),
                (pattern.ac, points.ac_ratio),
                (pattern.bd, points.bd_ratio),
                (pattern.xd, points.xd_ratio),
            ]:
                ratio_err = self._ratio_error(req, act)
                if ratio_err > self.ratio_tolerance:
                    error = float('inf')
                    break
                error += ratio_err

            if error < best_result.error:
                direction = "Bearish" if points.d_price > points.c_price else "Bullish"
                best_result = PatternResult(
                    name=pattern.name,
                    direction=direction,
                    error=error,
                    points=points
                )

        return best_result

    def _build_lines(self, pivots: deque) -> List[Tuple]:
        """Build line segments for plotting."""
        pivot_list = list(pivots)
        lines = []

        for i in range(len(pivot_list) - 1):
            lines.append((pivot_list[i][0], pivot_list[i+1][0]))

        # Add XD line
        if len(pivot_list) >= 5:
            lines.append((pivot_list[-5][0], pivot_list[-1][0]))

        return lines
