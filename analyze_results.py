#!/usr/bin/env python3
"""
analyze_results.py - Compare baseline vs CNN-enhanced Stockfish results from PGN files.

Usage:
    python analyze_results.py baseline_results.pgn cnn_results.pgn

Requires python-chess:
    pip install chess
"""

import sys
import chess.pgn


def analyze_pgn(filename):
    wins = 0
    losses = 0
    draws = 0
    trap_success = 0
    smart_mistakes = 0

    try:
        with open(filename) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                result = game.headers.get("Result", "*")
                if result == "1-0":
                    wins += 1
                elif result == "0-1":
                    losses += 1
                elif result == "1/2-1/2":
                    draws += 1

                # Analyze comments for CNN statistics
                for node in game.mainline():
                    comment = node.comment or ""
                    if "TRAP_SUCCESS" in comment:
                        trap_success += 1
                    if "SMART_MISTAKE" in comment:
                        smart_mistakes += 1
    except FileNotFoundError:
        print(f"ERROR: File not found: {filename}", file=sys.stderr)
        sys.exit(1)

    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.0

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total,
        "win_rate": win_rate * 100,
        "trap_success": trap_success,
        "smart_mistakes": smart_mistakes,
    }


def print_stats(label, stats):
    print(f"=== {label} ===")
    print(f"Record:    {stats['wins']}-{stats['losses']}-{stats['draws']}")
    print(f"Total:     {stats['total']} games")
    print(f"Win rate:  {stats['win_rate']:.1f}%")
    if stats["trap_success"] or stats["smart_mistakes"]:
        print(f"Traps set successfully: {stats['trap_success']}")
        print(f"Smart mistakes made:    {stats['smart_mistakes']}")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} baseline.pgn cnn_enhanced.pgn")
        sys.exit(1)

    baseline_file = sys.argv[1]
    cnn_file = sys.argv[2]

    baseline = analyze_pgn(baseline_file)
    cnn = analyze_pgn(cnn_file)

    print()
    print_stats("BASELINE STOCKFISH", baseline)
    print()
    print_stats("CNN-ENHANCED STOCKFISH", cnn)

    improvement = cnn["win_rate"] - baseline["win_rate"]
    print()
    print("=== IMPROVEMENT ===")
    print(f"Win rate delta: {improvement:+.1f}%")
    print()

    if improvement > 10:
        print("EXCELLENT: >10% improvement achieved!")
    elif improvement > 5:
        print("GOOD: 5-10% improvement")
    elif improvement > 0:
        print("MODEST: Small positive improvement")
    elif improvement == 0:
        print("NEUTRAL: No change in win rate")
    else:
        print("NEGATIVE: CNN is not helping — check CNN accuracy and parameters")

    print()
    if baseline["total"] < 50:
        print("WARNING: Sample size is small (< 50 games). Results may not be statistically significant.")
    elif baseline["total"] < 100:
        print("NOTE: Consider running more games (100+) for reliable results.")


if __name__ == "__main__":
    main()
