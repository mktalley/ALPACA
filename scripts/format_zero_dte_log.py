#!/usr/bin/env python3
"""
format_zero_dte_log.py: Read a zero_dte.log file and produce a human-friendly summary.

Usage:
    python3 scripts/format_zero_dte_log.py [--input path/to/zero_dte.log]
"""

import re
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Format zero_dte.log for human readability")
    parser.add_argument('--input', '-i', default='apps/logs/zero_dte.log', help='Path to zero_dte.log file (default apps/logs/zero_dte.log)')
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        lines = open(args.input, 'r', encoding='utf-8').read().splitlines()
    except Exception as e:
        print(f"Error reading log file {args.input}: {e}")
        sys.exit(1)

    entries = []  # list of (time, msg)
    seen_price = set()

    price_re = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?DEBUG.*?\[(?:Thread-[^\]]+)\] Current prices \[(?P<syms>[^\]]+)\] → (?P<val>[0-9.]+) \(sum\)")
    generic_re = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?(?P<lvl>INFO|WARNING|ERROR)\s+\[[^\]]+\] (?P<msg>.*)")

    for line in lines:
        # Price updates (DEBUG) — include but collapse duplicates
        m = price_re.match(line)
        if m:
            ts = m.group('ts')
            syms = m.group('syms')
            val = m.group('val')
            # normalize symbols string
            symlist = [s.strip().strip("'\"") for s in syms.split(',')]
            key = (ts, tuple(symlist), val)
            if key in seen_price:
                continue
            seen_price.add(key)
            # human format
            sym_str = ' & '.join(symlist)
            entries.append((ts, f"Price sum for {sym_str} → ${val}"))
            continue

        # Skip debug noise
        if 'DEBUG' in line:
            continue

        # Other INFO/WARNING/ERROR
        m2 = generic_re.match(line)
        if m2:
            ts = m2.group('ts')
            msg = m2.group('msg')
            entries.append((ts, msg))
            continue

        # non-matching lines ignored

    # Print human-friendly table
    col_width = max((len(ts) for ts, _ in entries), default=0)
    print(f"{'Time':<{col_width}}  │ Event")
    print('-' * (col_width + 3 + 50))
    for ts, msg in entries:
        print(f"{ts:<{col_width}}  │ {msg}")

if __name__ == '__main__':
    main()
