#!/usr/bin/env python3
"""
Convert Maia Chess LC0 models (.pb.gz) to ONNX format for use with
the HumanLike-Stockfish opponent modeling system.

Maia Chess (https://github.com/CSSLab/maia-chess) provides 9 models,
one per ELO level (1100-1900), each predicting the most likely human move.

Output (written to --output-dir):
  maia-1100.onnx ... maia-1900.onnx   — one ONNX model per ELO level
  maia_vocab.txt                        — shared move vocabulary
                                          (policy index → UCI move string)
ONNX interface:
  Input:  "board"  shape (1, 112, 8, 8)  — LC0 112-plane board encoding
  Output: "policy" shape (1, 1858)        — raw move logits (pre-softmax)

Primary conversion method: uses the `lc0 leela2onnx` command when available
(produces correct results), then post-processes the ONNX to rename tensors.

Fallback: pure-Python conversion via PyTorch (used when lc0 binary not found).

Usage:
  # Download and convert all ELO levels:
  python convert_maia_to_onnx.py --download --output-dir ./maia_models

  # Convert a single pre-downloaded .pb.gz:
  python convert_maia_to_onnx.py --weights maia-1500.pb.gz --elo 1500 --output-dir ./maia_models

  # Force lc0-based conversion (recommended):
  python convert_maia_to_onnx.py --download --output-dir ./maia_models --lc0 /opt/homebrew/bin/lc0
"""

import argparse
import gzip
import os
import shutil
import struct
import subprocess
import tempfile
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Maia download URLs ────────────────────────────────────────────────────────
# From https://github.com/CSSLab/maia-chess/releases/tag/v1.0
MAIA_URLS: Dict[int, str] = {
    elo: f"https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-{elo}.pb.gz"
    for elo in range(1100, 2000, 100)
}

# ── LC0 policy move encoding constants ───────────────────────────────────────
#
# The LC0 policy head produces an (80, 8, 8) tensor (5120 elements, CHW).
# flat_index = channel * 64 + from_sq  where from_sq = rank*8 + file (a1=0, h8=63)
#
# Channel assignments:
#   0-55:  Queen-like moves.  dir = ch//7, dist = ch%7+1
#          Directions (df, dr): N=(0,1), NE=(1,1), E=(1,0), SE=(1,-1),
#                               S=(0,-1), SW=(-1,-1), W=(-1,0), NW=(-1,1)
#   56-63: Knight moves. (df,dr) in order:
#          (1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)
#   64-72: Underpromotions (R/B/N only; queen promo uses queen move channel).
#          channel = 64 + dir_idx*3 + piece_idx
#          dir_idx: 0=NW(-1,1), 1=N(0,1), 2=NE(1,1)
#          piece_idx: 0=R, 1=B, 2=N
#
# The 1858 LC0 policy outputs are ordered as:
#   Part 1 (indices 0-1791):  Regular moves (queen + knight) per from_sq 0..63,
#          each from_sq's moves sorted by destination square (ascending).
#   Part 2 (indices 1792-1857): Underpromotions, from_sq 48..55 (rank 6),
#          per direction NW/N/NE (if valid), pieces in order N, R, B.
#          Note: piece_idx order for channels is R=0,B=1,N=2 but LC0 outputs
#          them as N,R,B, so channels within a direction: [ch_N, ch_R, ch_B].

_QUEEN_DIRS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
_KNIGHT_DELTAS = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]
_UNDER_DIRS = [(-1,1),(0,1),(1,1)]   # NW, N, NE  (df, dr)
# For each underpromo direction index (0=NW,1=N,2=NE):
# the channel sequence in [N_promo, R_promo, B_promo] order
_UNDER_DIR_CHANNELS = {
    0: [66, 64, 65],   # NW: N=ch66, R=ch64, B=ch65
    1: [69, 67, 68],   # N:  N=ch69, R=ch67, B=ch68
    2: [72, 70, 71],   # NE: N=ch72, R=ch70, B=ch71
}
_FILES = 'abcdefgh'


def _lc0_policy_flat_indices() -> List[int]:
    """Return the 1858 flat indices (into the 5120-element policy tensor) in LC0 order."""

    def queen_dest(from_sq, ch):
        r, f = from_sq // 8, from_sq % 8
        dir_idx = ch // 7; dist = ch % 7 + 1
        df, dr = _QUEEN_DIRS[dir_idx]
        tf, tr = f + df*dist, r + dr*dist
        return tr*8 + tf if (0 <= tf <= 7 and 0 <= tr <= 7) else None

    def knight_dest(from_sq, ch):
        r, f = from_sq // 8, from_sq % 8
        df, dr = _KNIGHT_DELTAS[ch - 56]
        tf, tr = f + df, r + dr
        return tr*8 + tf if (0 <= tf <= 7 and 0 <= tr <= 7) else None

    indices: List[int] = []

    # Part 1: regular moves sorted by destination square per from_sq
    for from_sq in range(64):
        moves = []
        for ch in range(56):        # queen moves
            dest = queen_dest(from_sq, ch)
            if dest is not None:
                moves.append((dest, ch * 64 + from_sq))
        for ch in range(56, 64):    # knight moves
            dest = knight_dest(from_sq, ch)
            if dest is not None:
                moves.append((dest, ch * 64 + from_sq))
        moves.sort()
        indices.extend(flat for _, flat in moves)

    # Part 2: underpromotions (rank 6 only, positions 1792-1857)
    for from_sq in range(48, 56):   # rank 6, files 0..7
        from_file = from_sq % 8
        for dir_idx, (df, _) in enumerate(_UNDER_DIRS):
            to_file = from_file + df
            if 0 <= to_file <= 7:
                for ch in _UNDER_DIR_CHANNELS[dir_idx]:  # N, R, B order
                    indices.append(ch * 64 + from_sq)

    return indices


def _flat_to_uci(flat_idx: int) -> Optional[str]:
    """Decode a flat policy index to a UCI move string (or None if invalid)."""
    ch = flat_idx // 64
    from_sq = flat_idx % 64
    from_rank = from_sq // 8
    from_file = from_sq % 8
    from_str = _FILES[from_file] + str(from_rank + 1)

    if ch < 56:     # queen-like move
        dir_idx = ch // 7; dist = ch % 7 + 1
        df, dr = _QUEEN_DIRS[dir_idx]
        to_file = from_file + df * dist
        to_rank = from_rank + dr * dist
        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return None
        to_str = _FILES[to_file] + str(to_rank + 1)
        # Queen promotion: N/NE/NW, dist=1, from rank 6 to rank 7
        if from_rank == 6 and to_rank == 7 and dist == 1:
            return from_str + to_str + 'q'
        return from_str + to_str

    elif ch < 64:   # knight move
        df, dr = _KNIGHT_DELTAS[ch - 56]
        to_file = from_file + df
        to_rank = from_rank + dr
        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return None
        return from_str + _FILES[to_file] + str(to_rank + 1)

    else:           # underpromotion (ch 64-72)
        _PROMO_PIECES = ['r', 'b', 'n']   # R=0, B=1, N=2
        idx = ch - 64
        dir_idx = idx // 3
        piece_idx = idx % 3
        df, dr = _UNDER_DIRS[dir_idx]
        to_file = from_file + df
        to_rank = from_rank + dr
        if not (0 <= to_file <= 7 and to_rank == 7):
            return None
        return from_str + _FILES[to_file] + '8' + _PROMO_PIECES[piece_idx]


# ── Minimal protobuf parser ───────────────────────────────────────────────────

def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, pos
        shift += 7
    raise ValueError("Truncated varint")


def _parse_pb(data: bytes) -> Dict[int, list]:
    """Parse a flat protobuf message into {field_number: [raw_values]}."""
    fields: Dict[int, list] = {}
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        fn = tag >> 3
        wt = tag & 7

        if wt == 0:       # varint
            v, pos = _read_varint(data, pos)
        elif wt == 1:     # 64-bit
            v = data[pos:pos + 8]
            pos += 8
        elif wt == 2:     # length-delimited (sub-message or bytes)
            n, pos = _read_varint(data, pos)
            v = data[pos:pos + n]
            pos += n
        elif wt == 5:     # 32-bit
            v = data[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unknown protobuf wire type {wt} at offset {pos}")

        fields.setdefault(fn, []).append(v)
    return fields


def _layer_to_numpy(layer_bytes: bytes) -> np.ndarray:
    """
    Parse a LC0 Layer sub-message and return float32 weights.

    LC0 format version 2 (used by Maia) uses linear quantisation:
      field 1: min_val  (fixed32 IEEE-754 float, 4 bytes)
      field 2: max_val  (fixed32 IEEE-754 float, 4 bytes)
      field 3: params   (packed uint16 integers, range 0-65535)

    Dequantisation: actual = min + (uint16 / 65535) * (max - min)

    Older/alternative format stores params as packed float32 at field 1.
    """
    import struct as _struct
    fields = _parse_pb(layer_bytes)

    # Detect LC0 quantised format: field 1 and 2 are 4-byte scalars, field 3 is the data
    quantised = (
        1 in fields and isinstance(fields[1][0], bytes) and len(fields[1][0]) == 4
        and 2 in fields and isinstance(fields[2][0], bytes) and len(fields[2][0]) == 4
        and 3 in fields and isinstance(fields[3][0], bytes) and len(fields[3][0]) > 8
    )

    if quantised:
        min_val = _struct.unpack('<f', fields[1][0])[0]
        max_val = _struct.unpack('<f', fields[2][0])[0]
        raw = fields[3][0]
        uint16_vals = np.frombuffer(raw, dtype=np.uint16).astype(np.float32)
        if max_val == min_val:
            return np.full(len(uint16_vals), min_val, dtype=np.float32)
        return (min_val + (uint16_vals / 65535.0) * (max_val - min_val)).astype(np.float32)

    # Fall back to field 1 containing packed floats (older/unquantised format)
    if 1 in fields and isinstance(fields[1][0], bytes) and len(fields[1][0]) > 8:
        params = fields[1][0]
        if len(params) % 4 == 0:
            return np.frombuffer(params, dtype=np.float32).copy()
        elif len(params) % 2 == 0:
            return np.frombuffer(params, dtype=np.float16).astype(np.float32)
        return np.frombuffer(params[:len(params) // 4 * 4], dtype=np.float32).copy()

    return np.empty(0, dtype=np.float32)


def _conv_block_layers(cb_bytes: bytes) -> Dict[str, np.ndarray]:
    """
    Parse a LC0 ConvBlock sub-message.
    Returns dict with keys: weights, bn_means, bn_stddivs, bn_gammas, bn_betas
    """
    fields = _parse_pb(cb_bytes)
    result = {}
    layer_map = {1: "weights", 2: "biases", 3: "bn_means",
                 4: "bn_stddivs", 5: "bn_gammas", 6: "bn_betas"}
    for field_num, name in layer_map.items():
        if field_num in fields:
            result[name] = _layer_to_numpy(fields[field_num][0])
    return result


def _se_unit_layers(se_bytes: bytes) -> Dict[str, np.ndarray]:
    """Parse a LC0 SEunit sub-message. Returns w1, b1, w2, b2."""
    fields = _parse_pb(se_bytes)
    result = {}
    for fn, name in [(1, "w1"), (2, "b1"), (3, "w2"), (4, "b2")]:
        if fn in fields:
            result[name] = _layer_to_numpy(fields[fn][0])
    return result


# ── LC0 weight loader ─────────────────────────────────────────────────────────

def load_lc0_pb_gz(path: str) -> dict:
    """
    Load a LC0 .pb.gz model file and return a structured weight dict:
      {
        "input": {weights, bn_means, bn_stddivs, bn_gammas, bn_betas},
        "residuals": [ {"conv1": ..., "conv2": ..., "se": ...}, ... ],
        "policy_conv": {...},
        "num_filters": int,
        "num_blocks": int,
      }
    """
    with gzip.open(path, "rb") as f:
        raw = f.read()

    # LC0 .pb.gz files start with a 4-byte raw uint32 magic (0x1c0 = 448)
    # before the protobuf payload — skip it before parsing.
    proto_start = 0
    if len(raw) >= 4 and struct.unpack_from('<I', raw, 0)[0] == 0x1c0:
        proto_start = 4

    # Top-level Net message. Field 10 = Weights in Maia/LC0 format version 2.
    net = _parse_pb(raw[proto_start:])

    if 10 not in net:
        raise ValueError(f"No weights field (field 10) found in {path}")

    wdata = net[10][0]
    wfields = _parse_pb(wdata)

    # Field 1 = input ConvBlock
    if 1 not in wfields:
        raise ValueError("Missing input conv block in weights")
    input_cb = _conv_block_layers(wfields[1][0])

    # Infer num_filters from input conv weight shape
    w = input_cb["weights"]
    in_planes = 112
    num_filters = len(w) // (in_planes * 9)
    if num_filters * in_planes * 9 != len(w):
        for ip in [112, 120, 128]:
            nf = len(w) // (ip * 9)
            if nf * ip * 9 == len(w) and nf > 0:
                in_planes = ip
                num_filters = nf
                break

    # Field 2 = repeated Residual blocks
    residuals = []
    for rb_bytes in wfields.get(2, []):
        rb_fields = _parse_pb(rb_bytes)
        rb = {}
        if 1 in rb_fields:
            rb["conv1"] = _conv_block_layers(rb_fields[1][0])
        if 2 in rb_fields:
            rb["conv2"] = _conv_block_layers(rb_fields[2][0])
        if 3 in rb_fields:
            rb["se"] = _se_unit_layers(rb_fields[3][0])
        residuals.append(rb)

    # Policy head: field 3 = policy ConvBlock (3×3 conv, bias=True, NO BN)
    policy_conv = None
    if 3 in wfields:
        policy_conv = _conv_block_layers(wfields[3][0])

    return {
        "input": input_cb,
        "residuals": residuals,
        "policy_conv": policy_conv,
        "num_filters": num_filters,
        "num_blocks": len(residuals),
        "in_planes": in_planes,
    }


# ── PyTorch SE-ResNet (Maia architecture) ─────────────────────────────────────

class _SEUnit(nn.Module):
    def __init__(self, filters: int, se_size: int):
        super().__init__()
        self.fc1 = nn.Linear(filters, se_size)
        self.fc2 = nn.Linear(se_size, filters * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=[2, 3])
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        gamma = torch.sigmoid(s[:, :x.size(1)]).unsqueeze(-1).unsqueeze(-1)
        beta  = s[:, x.size(1):].unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class _ResBlock(nn.Module):
    def __init__(self, filters: int, se_size: Optional[int]):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)
        self.se    = _SEUnit(filters, se_size) if se_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.se is not None:
            x = self.se(x)
        return F.relu(x + residual)


class MaiaNet(nn.Module):
    """Maia Chess SE-ResNet for human move prediction.

    Policy head: 3×3 conv (filters→policy_channels, bias=True, no BN) whose
    output (policy_channels×8×8) is flattened and indexed by policy_indices
    to produce exactly 1858 logits.
    """

    def __init__(self, in_planes: int = 112, filters: int = 64,
                 num_blocks: int = 6, se_size: int = 8,
                 policy_channels: int = 80,
                 policy_indices: Optional[torch.Tensor] = None):
        super().__init__()
        self.input_conv = nn.Conv2d(in_planes, filters, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(filters)
        self.blocks     = nn.ModuleList(
            [_ResBlock(filters, se_size) for _ in range(num_blocks)]
        )
        self.policy_conv = nn.Conv2d(filters, policy_channels, 3, padding=1, bias=True)
        if policy_indices is not None:
            self.register_buffer("policy_indices", policy_indices)
        else:
            self.register_buffer("policy_indices",
                                 _default_policy_indices(policy_channels))

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_bn(self.input_conv(board)))
        for block in self.blocks:
            x = block(x)
        p = self.policy_conv(x)             # (B, policy_channels, 8, 8)
        p = p.flatten(1)                    # (B, policy_channels * 64)
        return p[:, self.policy_indices]    # (B, 1858)


def _default_policy_indices(num_policy_channels: int = 80) -> torch.Tensor:
    """Return the canonical 1858 LC0 policy indices as a torch.LongTensor."""
    return torch.tensor(_lc0_policy_flat_indices(), dtype=torch.long)


def _load_bn(bn: nn.BatchNorm2d, cb: dict) -> None:
    """Load batch-norm parameters from parsed ConvBlock into nn.BatchNorm2d.

    LC0 stores bn_stddivs = 1/sqrt(var + eps)  (the *inverse* standard deviation).
    PyTorch BatchNorm2d uses running_var = var.
    Conversion: running_var = 1/stddivs^2 - eps   (computed in float64 to avoid overflow)
    """
    if "bn_gammas" in cb:
        bn.weight.data = torch.tensor(cb["bn_gammas"].astype(np.float32))
    if "bn_betas" in cb:
        bn.bias.data = torch.tensor(cb["bn_betas"].astype(np.float32))
    if "bn_means" in cb:
        bn.running_mean.data = torch.tensor(cb["bn_means"].astype(np.float32))
    if "bn_stddivs" in cb:
        stddivs = cb["bn_stddivs"].astype(np.float64)
        safe = np.isfinite(stddivs) & (stddivs > 1e-30)
        var = np.where(safe, np.clip(1.0 / (stddivs ** 2) - 1e-5, 0.0, None), 0.0)
        bn.running_var.data = torch.tensor(var.astype(np.float32))
    bn.eval()


def _load_conv(conv: nn.Conv2d, cb: dict, in_ch: int, out_ch: int, ksize: int = 3) -> None:
    """Load conv weights from parsed ConvBlock."""
    if "weights" in cb:
        w = cb["weights"]
        expected = out_ch * in_ch * ksize * ksize
        if len(w) == expected:
            conv.weight.data = torch.tensor(w).reshape(out_ch, in_ch, ksize, ksize)


def build_maia_model(weights: dict) -> MaiaNet:
    """Build and populate a MaiaNet from loaded LC0 weights."""
    nf  = weights["num_filters"]
    nb  = weights["num_blocks"]
    ip  = weights["in_planes"]

    se_size = 8
    for rb in weights["residuals"]:
        if "se" in rb:
            w1 = rb["se"].get("w1", np.array([]))
            if len(w1) > 0 and nf > 0:
                se_size = max(1, len(w1) // nf)
            break

    pc = weights.get("policy_conv")
    policy_channels = 80
    if pc and "biases" in pc and len(pc["biases"]) > 0:
        policy_channels = len(pc["biases"])

    model = MaiaNet(in_planes=ip, filters=nf, num_blocks=nb,
                    se_size=se_size, policy_channels=policy_channels)
    model.eval()

    inp = weights["input"]
    _load_conv(model.input_conv, inp, ip, nf)
    _load_bn(model.input_bn, inp)

    for i, rb in enumerate(weights["residuals"]):
        block = model.blocks[i]
        if "conv1" in rb:
            _load_conv(block.conv1, rb["conv1"], nf, nf)
            _load_bn(block.bn1, rb["conv1"])
        if "conv2" in rb:
            _load_conv(block.conv2, rb["conv2"], nf, nf)
            _load_bn(block.bn2, rb["conv2"])
        if "se" in rb and block.se is not None:
            se = rb["se"]
            if "w1" in se:
                block.se.fc1.weight.data = torch.tensor(se["w1"]).reshape(se_size, nf)
            if "b1" in se:
                block.se.fc1.bias.data = torch.tensor(se["b1"])
            if "w2" in se:
                block.se.fc2.weight.data = torch.tensor(se["w2"]).reshape(nf * 2, se_size)
            if "b2" in se:
                block.se.fc2.bias.data = torch.tensor(se["b2"])

    if pc and "weights" in pc:
        _load_conv(model.policy_conv, pc, nf, policy_channels, ksize=3)
    if pc and "biases" in pc and len(pc["biases"]) == policy_channels:
        model.policy_conv.bias.data = torch.tensor(pc["biases"].astype(np.float32))

    return model


# ── LC0 policy vocabulary ─────────────────────────────────────────────────────

def generate_lc0_vocabulary() -> List[str]:
    """
    Generate the LC0 policy vocabulary: list of 1858 UCI move strings.

    Uses the canonical LC0 policy ordering (same order as _default_policy_indices).
    Moves in the vocabulary match LC0's internal policy index numbering:
      - Positions 0-1791: regular moves (queen + knight) sorted by destination
      - Positions 1792-1857: underpromotions (R/B/N, from rank 6)
    """
    indices = _lc0_policy_flat_indices()
    vocab = []
    for flat_idx in indices:
        move = _flat_to_uci(flat_idx)
        if move is None:
            raise ValueError(f"Invalid flat index {flat_idx} in policy indices")
        vocab.append(move)
    return vocab


# ── lc0 leela2onnx conversion (primary method) ───────────────────────────────

def _find_lc0(lc0_path: Optional[str] = None) -> Optional[str]:
    """Find the lc0 binary. Returns path or None."""
    candidates = [lc0_path] if lc0_path else []
    candidates += [
        shutil.which("lc0"),
        "/opt/homebrew/bin/lc0",
        "/usr/local/bin/lc0",
        "/usr/bin/lc0",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _postprocess_lc0_onnx(ref_onnx_path: str, output_path: str) -> None:
    """
    Post-process an lc0 leela2onnx output to match our C++ interface:
      - Rename input  '/input/planes' → 'board'
      - Rename output '/output/policy' → 'policy'
      - Remove the '/output/wdl' output (not needed for opponent modeling)
    """
    import onnx
    from onnx import TensorProto

    model = onnx.load(ref_onnx_path)
    graph = model.graph

    OLD_INPUT  = "/input/planes"
    NEW_INPUT  = "board"
    OLD_OUTPUT = "/output/policy"
    NEW_OUTPUT = "policy"
    DROP_OUTPUT = "/output/wdl"

    def rename_in_graph(old: str, new: str) -> None:
        """Rename all references to `old` tensor as `new` throughout the graph."""
        # Graph inputs
        for inp in graph.input:
            if inp.name == old:
                inp.name = new
        # Graph outputs
        for out in graph.output:
            if out.name == old:
                out.name = new
        # Node inputs and outputs
        for node in graph.node:
            for i in range(len(node.input)):
                if node.input[i] == old:
                    node.input[i] = new
            for i in range(len(node.output)):
                if node.output[i] == old:
                    node.output[i] = new
        # Initializers
        for init in graph.initializer:
            if init.name == old:
                init.name = new
        # value_info
        for vi in graph.value_info:
            if vi.name == old:
                vi.name = new

    # Rename input and policy output
    rename_in_graph(OLD_INPUT, NEW_INPUT)
    rename_in_graph(OLD_OUTPUT, NEW_OUTPUT)

    # Remove the WDL output from graph.output
    wdl_outputs = [o for o in graph.output if o.name == DROP_OUTPUT]
    for o in wdl_outputs:
        graph.output.remove(o)

    onnx.save(model, output_path)


def convert_via_lc0(pb_gz_path: str, output_path: str, lc0_bin: str) -> bool:
    """
    Use `lc0 leela2onnx` to convert a .pb.gz to ONNX, then post-process.
    Returns True on success, False on failure.
    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_onnx = f.name

    try:
        result = subprocess.run(
            [lc0_bin, "leela2onnx",
             f"--input={pb_gz_path}", f"--output={tmp_onnx}"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  lc0 leela2onnx failed: {result.stderr.strip()}")
            return False

        _postprocess_lc0_onnx(tmp_onnx, output_path)
        return True

    except Exception as e:
        print(f"  lc0 conversion error: {e}")
        return False
    finally:
        if os.path.exists(tmp_onnx):
            os.unlink(tmp_onnx)


# ── PyTorch fallback export ───────────────────────────────────────────────────

def export_to_onnx(model: MaiaNet, output_path: str) -> None:
    board_dummy = torch.zeros(1, 112, 8, 8)
    with torch.no_grad():
        _ = model(board_dummy)

    torch.onnx.export(
        model,
        board_dummy,
        output_path,
        input_names=["board"],
        output_names=["policy"],
        dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"  Exported ONNX → {output_path}")


def verify_onnx(onnx_path: str) -> bool:
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(onnx_path)
        board = np.zeros((1, 112, 8, 8), dtype=np.float32)
        out = sess.run(["policy"], {"board": board})[0]
        print(f"  ONNX Runtime verify: output shape {out.shape}  ✓")
        return True
    except ImportError:
        print("  (Install onnxruntime to verify)")
        return True
    except Exception as e:
        print(f"  ONNX verify failed: {e}")
        return False


# ── Download helper ───────────────────────────────────────────────────────────

def download_maia(elo: int, dest_dir: str) -> str:
    """Download maia-{elo}.pb.gz to dest_dir if not already present."""
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.join(dest_dir, f"maia-{elo}.pb.gz")
    if os.path.exists(fname):
        print(f"  Already downloaded: {fname}")
        return fname
    url = MAIA_URLS[elo]
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, fname)
    print(f"  Saved → {fname}")
    return fname


# ── Main conversion routine ───────────────────────────────────────────────────

def convert_one(pb_gz_path: str, elo: int, output_dir: str,
                lc0_bin: Optional[str] = None) -> None:
    print(f"\n[ELO {elo}] Converting {pb_gz_path}")
    onnx_path = os.path.join(output_dir, f"maia-{elo}.onnx")

    # Primary: use lc0 leela2onnx (most accurate, preserves BN correctly)
    if lc0_bin:
        print(f"  Using lc0 leela2onnx: {lc0_bin}")
        if convert_via_lc0(pb_gz_path, onnx_path, lc0_bin):
            verify_onnx(onnx_path)
            return
        print("  lc0 conversion failed, falling back to Python conversion")

    # Fallback: pure Python via PyTorch
    print("  Using Python/PyTorch conversion")
    weights = load_lc0_pb_gz(pb_gz_path)
    print(f"  Architecture: {weights['num_blocks']} blocks × {weights['num_filters']} filters"
          f"  (in_planes={weights['in_planes']})")

    model = build_maia_model(weights)
    model.eval()
    export_to_onnx(model, onnx_path)
    verify_onnx(onnx_path)


def write_vocab(output_dir: str) -> None:
    vocab = generate_lc0_vocabulary()
    vocab_path = os.path.join(output_dir, "maia_vocab.txt")
    with open(vocab_path, "w") as f:
        for move in vocab:
            f.write(move + "\n")
    print(f"\nVocabulary: {len(vocab)} moves → {vocab_path}")
    if len(vocab) != 1858:
        print(f"  WARNING: Expected 1858 policy moves, got {len(vocab)}.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Maia LC0 models to ONNX")
    parser.add_argument("--download", action="store_true",
                        help="Download all Maia .pb.gz files from GitHub")
    parser.add_argument("--weights", metavar="PATH",
                        help="Path to a single Maia .pb.gz file")
    parser.add_argument("--elo", type=int,
                        help="ELO level when converting a single --weights file")
    parser.add_argument("--output-dir", default="./maia_models",
                        help="Directory for ONNX output files (default: ./maia_models)")
    parser.add_argument("--elos", nargs="+", type=int,
                        default=list(range(1100, 2000, 100)),
                        help="ELO levels to download/convert (default: 1100-1900)")
    parser.add_argument("--lc0", metavar="PATH", default=None,
                        help="Path to lc0 binary (auto-detected if not specified)")
    parser.add_argument("--no-lc0", action="store_true",
                        help="Force Python/PyTorch conversion even if lc0 is available")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find lc0 binary
    lc0_bin = None if args.no_lc0 else _find_lc0(args.lc0)
    if lc0_bin:
        print(f"Using lc0 binary: {lc0_bin}")
    else:
        print("lc0 not found; using Python/PyTorch conversion (may have BN accuracy issues)")

    if args.weights:
        if args.elo is None:
            parser.error("--elo is required when using --weights")
        convert_one(args.weights, args.elo, args.output_dir, lc0_bin)
    elif args.download:
        cache_dir = os.path.join(args.output_dir, "pb_gz_cache")
        for elo in args.elos:
            pb_path = download_maia(elo, cache_dir)
            convert_one(pb_path, elo, args.output_dir, lc0_bin)
    else:
        found = []
        for fname in os.listdir(args.output_dir):
            if fname.endswith(".pb.gz"):
                try:
                    elo = int(fname.replace("maia-", "").replace(".pb.gz", ""))
                    found.append((elo, os.path.join(args.output_dir, fname)))
                except ValueError:
                    pass
        if not found:
            parser.error(
                "Specify --download to fetch Maia models, or --weights PATH --elo N "
                "to convert a specific file, or place .pb.gz files in --output-dir."
            )
        for elo, pb_path in sorted(found):
            convert_one(pb_path, elo, args.output_dir, lc0_bin)

    write_vocab(args.output_dir)
    print(f"\nDone. Set UCI option MaiaModelDir to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
