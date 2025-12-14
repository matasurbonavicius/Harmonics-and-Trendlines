"""
Simple pattern outcome analysis.
Finds harmonic patterns and checks what happens N candles after detection.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from harmonics import HarmonicDetector

# Config
SYMBOL = "SPY"
START = "2018-01-01"
END = "2024-12-01"
LOOKBACK = 100  # candles to look for pattern
HOLD_PERIODS = [3, 6, 9, 12, 15]  # candles to hold after pattern

def main():
    print(f"Downloading {SYMBOL} data from {START} to {END}...")
    data = yf.download(SYMBOL, start=START, end=END, progress=False)
    # Flatten columns if multi-index (yfinance stuff)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    detector = HarmonicDetector(ratio_tolerance=0.12)

    results = {
        'bullish': {h: [] for h in HOLD_PERIODS},
        'bearish': {h: [] for h in HOLD_PERIODS},
    }

    patterns_found = []
    last_pattern_idx = -LOOKBACK  # avoid overlapping detections

    # Slide through data
    for i in range(LOOKBACK, len(data) - max(HOLD_PERIODS)):
        subset = data.iloc[i-LOOKBACK:i+1]
        result = detector.detect(subset)

        if result.name and result.direction:
            # Skip if too close to last pattern
            if i - last_pattern_idx < 10:
                continue

            last_pattern_idx = i
            entry_price = data['Close'].iloc[i]
            direction = result.direction.lower()

            patterns_found.append({
                'date': data.index[i],
                'pattern': result.full_name,
                'entry': entry_price,
                'error': result.error
            })

            # Calculate returns for each hold period
            for h in HOLD_PERIODS:
                exit_price = data['Close'].iloc[i + h]

                if direction == 'bullish':
                    ret = (exit_price - entry_price) / entry_price * 100
                else:  # bearish = short
                    ret = (entry_price - exit_price) / entry_price * 100

                results[direction][h].append(ret)

    # Print results
    print(f"Total patterns found: {len(patterns_found)}")

    bullish_count = sum(1 for p in patterns_found if 'Bullish' in p['pattern'])
    bearish_count = sum(1 for p in patterns_found if 'Bearish' in p['pattern'])
    print(f"  Bullish: {bullish_count}")
    print(f"  Bearish: {bearish_count}")
    print()

    # Pattern breakdown
    pattern_counts = {}
    for p in patterns_found:
        name = p['pattern']
        pattern_counts[name] = pattern_counts.get(name, 0) + 1

    print("Pattern breakdown:")
    for name, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    print()

    # Results table
    print("avg return by hold period (%)")
    print(f"{'Direction':<12} | " + " | ".join(f"{h:>6}d" for h in HOLD_PERIODS))
    print("-" * 60)


    for direction in ['bullish', 'bearish']:
        row = f"{direction.capitalize():<12} | "
        for h in HOLD_PERIODS:
            if results[direction][h]:
                avg = np.mean(results[direction][h])
                row += f"{avg:>6.2f}% | "
            else:
                row += f"{'N/A':>6} | "
        print(row)
    print()

    # Combined
    print(f"{'Combined':<12} | ", end="")
    for h in HOLD_PERIODS:
        all_returns = results['bullish'][h] + results['bearish'][h]
        if all_returns:
            avg = np.mean(all_returns)
            print(f"{avg:>6.2f}% | ", end="")
        else:
            print(f"{'N/A':>6} | ", end="")
    print()

    # Win rates
    print()
    print("Win rate by holding period")
    print(f"{'Direction':<12} | " + " | ".join(f"{h:>6}d" for h in HOLD_PERIODS))
    print()

    for direction in ['bullish', 'bearish']:
        row = f"{direction.capitalize():<12} | "
        for h in HOLD_PERIODS:
            rets = results[direction][h]
            if rets:
                wins = sum(1 for r in rets if r > 0)
                wr = wins / len(rets) * 100
                row += f"{wr:>6.1f}% | "
            else:
                row += f"{'N/A':>6} | "
        print(row)

    # Last few patterns
    print()
    print("Last 10 patterns found:")    
    print("-" * 60)
    for p in patterns_found[-10:]:
        print(f"  {p['date'].strftime('%Y-%m-%d')} | {p['pattern']:<20} | entry: ${p['entry']:.2f}")


if __name__ == "__main__":
    main()
