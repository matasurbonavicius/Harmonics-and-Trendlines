"""
Harmonic Pattern & Trendline Visualization

An animated candlestick chart that identifies harmonic patterns
(Gartley, Bat, Butterfly, etc.) and draws support/resistance trendlines.

Usage:
    python main.py

Controls:
    - Click anywhere to pause/resume animation
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mplfinance as mpf
import yfinance as yf
import pandas as pd
import numpy as np

from harmonics import HarmonicDetector, PatternResult
from trendlines import TrendlineDetector, TrendlineResult, ChartPattern


class HarmonicVisualizer:
    """Animated visualization of harmonic patterns and trendlines."""

    def __init__(
        self,
        symbol: str = "SPY",
        start_date: str = "2021-01-04",
        end_date: str = "2022-01-05",
        lookback: int = 150,
        trendline_lookback: int = 20,
        interval_ms: int = 100,
    ):
        """
        Initialize the visualizer.

        Args:
            symbol: Stock ticker symbol
            start_date: Data start date (YYYY-MM-DD)
            end_date: Data end date (YYYY-MM-DD)
            lookback: Number of candles to display
            trendline_lookback: Candles to use for trendline calculation
            interval_ms: Animation interval in milliseconds
        """
        self.symbol = symbol
        self.lookback = lookback
        self.trendline_lookback = trendline_lookback
        self.interval_ms = interval_ms

        # Load data
        print(f"Downloading {symbol} data...")
        self.data = self._load_data(symbol, start_date, end_date)
        print(f"Loaded {len(self.data)} candles")

        # Initialize detectors
        self.harmonic_detector = HarmonicDetector()
        self.trendline_detector = TrendlineDetector(lookback=trendline_lookback)

        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        plt.style.use("seaborn-v0_8-whitegrid")

        # Animation state
        self.paused = False
        self.animation = None

    def _load_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download and prepare OHLC data."""
        data = yf.download(symbol, start=start, end=end, progress=False)

        # Handle multi-index columns from newer yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Ensure we have the right columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'

        return data

    def _update(self, frame: int):
        """Update function for animation."""
        self.ax.clear()

        # Skip initial frames while building up data
        if frame <= self.lookback:
            self._show_loading(frame)
            return

        # Get data subset
        start = max(0, frame - self.lookback)
        subset = self.data.iloc[start:frame + 1].copy()

        # Detect patterns
        harmonic_result = self.harmonic_detector.detect(subset)

        # Detect trendlines (on recent data)
        trendline_subset = subset.tail(self.trendline_lookback)
        trendline_result = self.trendline_detector.detect(
            close=trendline_subset['Close'].values,
            high=trendline_subset['High'].values,
            low=trendline_subset['Low'].values
        )

        # Plot chart
        self._plot_chart(subset, harmonic_result, trendline_result)

    def _show_loading(self, frame: int):
        """Show loading progress."""
        progress = frame / self.lookback
        self.ax.text(
            0.5, 0.5,
            f"Loading... {progress:.0%}",
            transform=self.ax.transAxes,
            ha='center', va='center',
            fontsize=20, color='gray'
        )
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

    def _plot_chart(
        self,
        data: pd.DataFrame,
        harmonic: PatternResult,
        trendline: TrendlineResult
    ):
        """Plot candlestick chart with patterns and trendlines."""
        # Build trendlines for mplfinance
        tlines = []

        # Add harmonic pattern lines
        if harmonic.lines:
            pattern_lines = [line for line in harmonic.lines[:-1]] if len(harmonic.lines) > 1 else harmonic.lines
            if pattern_lines:
                tlines.append(dict(
                    tlines=pattern_lines,
                    colors='#2196F3',
                    linewidths=2,
                    alpha=0.8
                ))

            # XD line (dashed)
            if len(harmonic.lines) >= 1:
                xd_line = [harmonic.lines[-1]]
                tlines.append(dict(
                    tlines=xd_line,
                    colors='#FF9800',
                    linestyle='--',
                    linewidths=1.5,
                    alpha=0.7
                ))

        # Add support/resistance trendlines
        trendline_dates = data.tail(self.trendline_lookback).index.tolist()
        if len(trendline_dates) >= 2:
            support_line = trendline.support.get_line(len(trendline_dates))
            resist_line = trendline.resistance.get_line(len(trendline_dates))

            support_points = list(zip(trendline_dates, support_line))
            resist_points = list(zip(trendline_dates, resist_line))

            # Add as alines (anchored lines)
            alines = [support_points, resist_points]
        else:
            alines = None

        # Plot candlesticks
        plot_kwargs = dict(ax=self.ax, type='candle', style='charles')
        if tlines:
            plot_kwargs['tlines'] = tlines
        if alines:
            plot_kwargs['alines'] = dict(
                alines=alines,
                colors=['#4CAF50', '#F44336'],  # Green support, red resistance
                linewidths=1.5,
                alpha=0.8
            )

        mpf.plot(data, **plot_kwargs)

        # Add harmonic point labels
        self._add_point_labels(harmonic)

        # Add ratio labels
        self._add_ratio_labels(harmonic)

        # Add pattern labels
        self._add_pattern_labels(harmonic, trendline)

        # Style
        self.ax.set_title(f'{self.symbol} - Pattern Detection', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('')
        self.ax.tick_params(axis='x', rotation=45, labelsize=9)

    def _add_point_labels(self, result: PatternResult):
        """Add XABCD labels to chart."""
        points = result.points
        labels = [
            ('X', points.x_idx, points.x_price),
            ('A', points.a_idx, points.a_price),
            ('B', points.b_idx, points.b_price),
            ('C', points.c_idx, points.c_price),
            ('D', points.d_idx, points.d_price),
        ]

        for label, idx, price in labels:
            if idx is not None and price is not None:
                self.ax.annotate(
                    label,
                    xy=(idx, price),
                    fontsize=12,
                    fontweight='bold',
                    color='#1565C0',
                    ha='center',
                    va='bottom' if label in ['X', 'B', 'D'] else 'top'
                )

    def _add_ratio_labels(self, result: PatternResult):
        """Add Fibonacci ratio labels."""
        points = result.points

        ratios = [
            (points.x_idx, points.b_idx, points.x_price, points.b_price, points.xb_ratio),
            (points.a_idx, points.c_idx, points.a_price, points.c_price, points.ac_ratio),
            (points.b_idx, points.d_idx, points.b_price, points.d_price, points.bd_ratio),
        ]

        for x1, x2, y1, y2, ratio in ratios:
            if all(v is not None for v in [x1, x2, y1, y2, ratio]) and ratio < 10:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                self.ax.annotate(
                    f'{ratio:.3f}',
                    xy=(mid_x, mid_y),
                    fontsize=9,
                    color='#FF6F00',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
                )

    def _add_pattern_labels(self, harmonic: PatternResult, trendline: TrendlineResult):
        """Add pattern name labels."""
        # Harmonic pattern (top-left)
        harmonic_text = harmonic.full_name or "Scanning..."
        harmonic_color = '#4CAF50' if harmonic.name else '#9E9E9E'

        self.ax.text(
            0.02, 0.98,
            f'Harmonic: {harmonic_text}',
            transform=self.ax.transAxes,
            fontsize=12,
            fontweight='bold',
            color=harmonic_color,
            va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=harmonic_color)
        )

        # Trendline pattern (below harmonic)
        trendline_text = trendline.pattern.value
        trendline_color = self._get_pattern_color(trendline.pattern)

        self.ax.text(
            0.02, 0.90,
            f'Structure: {trendline_text}',
            transform=self.ax.transAxes,
            fontsize=12,
            fontweight='bold',
            color=trendline_color,
            va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=trendline_color)
        )

    def _get_pattern_color(self, pattern: ChartPattern) -> str:
        """Get color for chart pattern."""
        bullish = ['#4CAF50', '#8BC34A']  # Greens
        bearish = ['#F44336', '#FF5722']  # Reds
        neutral = ['#9E9E9E', '#607D8B']  # Grays

        if pattern in [ChartPattern.BULLISH_CHANNEL, ChartPattern.FALLING_WEDGE]:
            return bullish[0]
        elif pattern in [ChartPattern.BEARISH_CHANNEL, ChartPattern.RISING_WEDGE]:
            return bearish[0]
        elif pattern == ChartPattern.TRIANGLE:
            return '#FF9800'  # Orange
        else:
            return neutral[0]

    def _toggle_pause(self, event):
        """Toggle animation pause state."""
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def run(self):
        """Start the visualization."""
        self.animation = FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.data),
            interval=self.interval_ms,
            repeat=True
        )

        self.fig.canvas.mpl_connect('button_press_event', self._toggle_pause)

        plt.tight_layout()
        plt.show()


def main():
    """Entry point."""
    visualizer = HarmonicVisualizer(
        symbol="SPY",
        start_date="2021-01-04",
        end_date="2022-01-05",
        lookback=150,
        trendline_lookback=20,
        interval_ms=100,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
