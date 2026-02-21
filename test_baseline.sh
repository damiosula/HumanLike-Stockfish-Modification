#!/bin/bash
# Test traditional handicapped Stockfish (control group)
# Usage: ./test_baseline.sh [path_to_stockfish]

STOCKFISH="${1:-./src/stockfish}"

echo "=== Baseline Handicapped Stockfish Test ==="
echo "Using binary: $STOCKFISH"
echo ""

"$STOCKFISH" << 'EOF'
setoption name UCI_LimitStrength value true
setoption name UCI_Elo value 1500
setoption name Skill Level value 10
setoption name MultiPV value 4
setoption name UseOpponentModel value false

position startpos
go depth 10
quit
EOF
