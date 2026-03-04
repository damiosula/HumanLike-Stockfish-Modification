import os
import chess
import chess.engine

STOCKFISH = os.path.join(os.path.dirname(__file__), "src/stockfish")
MODEL     = os.path.join(os.path.dirname(__file__), "opponent_model.onnx")

# ONNX Runtime needs this until we rebuild with rpath embedded
os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/onnxruntime/lib"

# ── Start engine ──────────────────────────────────────────────────────────────
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)

# ── Load CNN model ────────────────────────────────────────────────────────────
engine.configure({
    "Skill Level":        10,       # activates handicap path (< 20)
    "UseOpponentModel":   True,
    "OpponentModelFile":  MODEL,
    "OpponentElo":        1500,     # ELO of the human you're simulating against
    "PlayingElo":         1500,     # ELO level the engine should play at
})

# ── Play one move from the starting position ──────────────────────────────────
board  = chess.Board()
result = engine.play(board, chess.engine.Limit(time=1.0))
print(f"Move chosen: {result.move}")

# ── Play a short game (10 moves) and print move choices ───────────────────────
print("\n--- 10-move game ---")
board = chess.Board()
for i in range(10):
    result = engine.play(board, chess.engine.Limit(time=0.5))
    board.push(result.move)
    print(f"  {i+1}. {result.move}  ({'White' if i % 2 == 0 else 'Black'})")
    if board.is_game_over():
        break

print(f"\nFinal FEN: {board.fen()}")

engine.quit()
