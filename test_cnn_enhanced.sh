#!/bin/bash
# Test CNN-enhanced Stockfish (treatment group)
# Usage: ./test_cnn_enhanced.sh [model_path] [opponent_elo] [playing_elo] [stockfish_path]

MODEL_PATH="${1:-/path/to/your/model.onnx}"
OPPONENT_ELO="${2:-1500}"
PLAYING_ELO="${3:-1500}"
STOCKFISH="${4:-./src/stockfish}"

echo "=== CNN-Enhanced Stockfish Test ==="
echo "Using binary:       $STOCKFISH"
echo "Model path:         $MODEL_PATH"
echo "Opponent ELO:       $OPPONENT_ELO"
echo "Playing ELO:        $PLAYING_ELO"
echo ""

"$STOCKFISH" << EOF
setoption name UCI_LimitStrength value true
setoption name UCI_Elo value $PLAYING_ELO
setoption name Skill Level value 10
setoption name MultiPV value 4

setoption name UseOpponentModel value true
setoption name OpponentModelFile value $MODEL_PATH
setoption name OpponentElo value $OPPONENT_ELO
setoption name PlayingElo value $PLAYING_ELO
setoption name CNNWeight value 70

setoption name EnableSmartMistakes value true
setoption name EnableTrapSetting value true
setoption name MistakeFrequency value 15

position startpos
go depth 10
quit
EOF
