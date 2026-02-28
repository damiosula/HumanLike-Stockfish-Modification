"""
Convert human_move_model.pt to ONNX for Stockfish integration.
Architecture recovered from checkpoint inspection:
  - elo_fc:          Linear(1→64→128)
  - input_conv:      Conv2d(12, 128, 3x3) + BN + ReLU
  - residual_blocks: 6x ResidualBlock(128)
  - fc:              Linear(8320→1968)   [8320 = 8192 board + 128 elo]
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class HumanMoveModel(nn.Module):
    def __init__(self, num_moves=1968, channels=128, num_blocks=6):
        super().__init__()
        self.elo_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.fc = nn.Linear(channels * 64 + 128, num_moves)

    def forward(self, board, elo):
        # board: (B, 12, 8, 8)  — 12-plane piece encoding
        # elo:   (B, 1)          — normalised ELO  (divide by 3000 before passing in)
        x = self.input_conv(board)
        for block in self.residual_blocks:
            x = block(x)
        board_feat = x.flatten(1)                              # (B, 8192)
        elo_feat   = self.elo_fc(elo)                          # (B, 128)
        return self.fc(torch.cat([board_feat, elo_feat], 1))   # (B, 1968)


# ── Load checkpoint ────────────────────────────────────────────────────────────
print("Loading checkpoint...")
ckpt = torch.load("human_move_model.pt", map_location="cpu")

model = HumanMoveModel()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Model loaded OK")

# ── Dry run ────────────────────────────────────────────────────────────────────
board_dummy = torch.zeros(1, 12, 8, 8)
elo_dummy   = torch.tensor([[1500.0 / 3000.0]])  # normalised

with torch.no_grad():
    out = model(board_dummy, elo_dummy)

print(f"Input  board: {tuple(board_dummy.shape)}")
print(f"Input  elo:   {tuple(elo_dummy.shape)}")
print(f"Output:       {tuple(out.shape)}  (1968 move logits)")

# ── Export to ONNX ─────────────────────────────────────────────────────────────
ONNX_FILE = "opponent_model.onnx"

torch.onnx.export(
    model,
    (board_dummy, elo_dummy),
    ONNX_FILE,
    input_names=["board", "elo"],
    output_names=["move_logits"],
    dynamic_axes={
        "board":       {0: "batch"},
        "elo":         {0: "batch"},
        "move_logits": {0: "batch"},
    },
    opset_version=17,
)
print(f"\nExported → {ONNX_FILE}")

# ── Save move vocabulary alongside the model ───────────────────────────────────
# move_vocab[i] = UCI move string for output index i
# Used by C++ output_to_moves() to map logit index → move string
move_vocab = ckpt["move_vocab"]   # list of 1968 strings
VOCAB_FILE = "opponent_model_vocab.txt"
with open(VOCAB_FILE, "w") as f:
    for move in move_vocab:
        f.write(move + "\n")
print(f"Saved move vocabulary ({len(move_vocab)} moves) → {VOCAB_FILE}")

# ── Verify with ONNX Runtime ───────────────────────────────────────────────────
try:
    import onnxruntime as rt

    sess = rt.InferenceSession(ONNX_FILE)
    out_ort = sess.run(
        ["move_logits"],
        {
            "board": board_dummy.numpy(),
            "elo":   elo_dummy.numpy(),
        },
    )
    diff = abs(out.numpy() - out_ort[0]).max()
    print(f"\nONNX Runtime check:")
    print(f"  Output shape: {out_ort[0].shape}")
    print(f"  Max diff PyTorch vs ONNX: {diff:.6f}  {'✓ OK' if diff < 1e-4 else '⚠ large'}")

    # Show top 5 predicted moves for the starting position at 1500 ELO
    import torch.nn.functional as F
    probs = F.softmax(torch.tensor(out_ort[0][0]), dim=0).numpy()
    top5  = probs.argsort()[::-1][:5]
    print("\nTop 5 moves (starting position, ELO 1500):")
    for idx in top5:
        print(f"  {move_vocab[idx]:8s}  {probs[idx]*100:.2f}%")

except ImportError:
    print("\nInstall onnxruntime to verify: pip install onnxruntime")
