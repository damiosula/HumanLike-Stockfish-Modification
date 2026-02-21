# CNN Opponent Model Integration — Change Log & Rationale

This document explains every file changed during the integration of the CNN-based opponent
model into weakened Stockfish, and the reasoning behind each decision.

---

## Overview

The goal is to replace Stockfish's random "skill level" move selection with **CNN-guided
smart handicapping**: instead of picking a random bad move, the engine picks a move that
looks reasonable but that players at the target ELO level are statistically likely to
mishandle. The CNN predicts which moves the opponent will make, allowing Stockfish to set
traps and accumulate small inaccuracies that mimic real human play rather than making
obvious computer blunders.

---

## Files Changed

### `src/Makefile`

**What changed:** `opponent_model.cpp` added to `SRCS`; `opponent_model.h` added to
`HEADERS`.

**Why:** The source and header files existed on disk but were never compiled — the Makefile
had no knowledge of them. Without this change the entire opponent model would be silently
ignored at build time, and the linker would report "undefined symbol" errors the moment
any other file tried to reference `OpponentModel`.

---

### `src/opponent_model.h` *(new file)*

**What changed:** Defines three public types and the `OpponentModel` class interface.

| Type | Purpose |
|---|---|
| `OpponentResponse` | A `(move, probability)` pair — one predicted opponent move |
| `MoveExploitability` | Metrics for one of *our* candidate moves: trap potential, blunder rate, difficulty, combined expected value |
| `OpponentModel` | The class that wraps the CNN and exposes prediction + evaluation helpers |

**Key design decisions:**

- **`Position&` (non-const) parameters** — Stockfish's `Position` class deletes its copy
  constructor, so the only way to temporarily explore a position is to call `do_move` /
  `undo_move` on the original object and restore it afterward. Functions that need to look
  ahead (trap detection, blunder rate calculation) therefore take a non-const reference.
  The position is always restored before the function returns.

- **`is_ready()` guard** — Every public method checks `modelLoaded` before doing anything.
  This means the entire system degrades gracefully to the original random-selection
  behavior when no model file has been loaded, with zero performance cost.

- **`cnnModel` omitted for now** — A placeholder comment is left where the framework-
  specific pointer (e.g. `Ort::Session*` for ONNX Runtime, or a LibTorch module) will
  live. Keeping a typed `void*` there caused a compiler warning about an unused private
  field, so it was replaced with a comment until the framework is chosen.

---

### `src/opponent_model.cpp` *(new file)*

**What changed:** Implements all `OpponentModel` methods.

**Why each method exists:**

| Method | Role |
|---|---|
| `load_model()` | Entry point for loading a CNN from disk. Currently a stub returning `false`. Implement here with ONNX Runtime or LibTorch. |
| `predict_responses()` | Runs CNN inference; returns a ranked list of `OpponentResponse`. Currently returns empty (model not loaded). |
| `get_move_probability()` | Convenience wrapper — calls `predict_responses` and looks up one specific move. Used by `is_smart_mistake`. |
| `rank_candidate_moves()` | **Core handicap function.** For each of Stockfish's top MultiPV moves, evaluates how exploitable it is: applies our move, asks the CNN for opponent responses, scores each response, and combines trap potential + blunder rate + difficulty into a single `expectedValue`. |
| `is_smart_mistake()` | Decides whether a suboptimal move is "human-like" (a real player at the target ELO sometimes plays it and the opponent cannot immediately punish it) vs. a dumb computer blunder (obvious refutation with high probability). |
| `evaluate_trap_potential()` | Counts how many of the opponent's plausible responses are bad for them. A move with many plausible-but-losing responses scores high. |
| `evaluate_after_opponent_response()` | Quick evaluation helper — applies an opponent move, calls `Eval::simple_eval`, and returns a normalised score in `[-1, 1]` via `tanh`. |
| `position_to_input()` | **TODO stub.** Converts the board state into the CNN's input tensor (bitboards, piece planes, etc.). |
| `output_to_moves()` | **TODO stub.** Converts the CNN's output logits into a sorted list of legal `OpponentResponse` pairs. |

**Why `Eval::simple_eval()` instead of the full NNUE `evaluate()`:**

The full NNUE evaluation requires `Networks`, `AccumulatorStack`, and `AccumulatorCaches`
— all of which live inside the `Worker` class. `OpponentModel` is intentionally decoupled
from the search to keep it reusable. `Eval::simple_eval()` is a free function that only
needs a `Position` and returns a fast material count. It is accurate enough for the
heuristic scoring done here, and will be replaced anyway once the CNN itself provides
evaluation signals through its output head.

---

### `src/search.h`

**What changed:** Three additions.

**1. `#include "opponent_model.h"`**

Needed so that `SharedState`, `Skill`, and `Worker` can refer to `OpponentModel` by name.
Placed alongside the other engine headers; causes no circular dependency because
`opponent_model.h` only includes `types.h` and `position.h`.

**2. `SharedState` gains an `OpponentModel*` field**

```cpp
SharedState(..., OpponentModel* oppModel = nullptr)
    : ..., opponentModel(oppModel) {}

OpponentModel* opponentModel = nullptr;
```

`SharedState` is the struct that is constructed once and passed to every `Worker` thread
at startup and whenever threads are resized. Adding the model pointer here means it flows
automatically to every worker without touching the threading or pool code. The pointer is
non-owning — the `Engine` owns the model via `std::unique_ptr`; workers just borrow it.
The default is `nullptr` so all existing call sites that do not pass a model continue to
compile unchanged.

**3. `Skill` gains `pick_best_with_cnn()` and `should_make_mistake()`**

```cpp
Move pick_best_with_cnn(const RootMoves&, size_t multiPV,
                        Position&, int opponentElo, int targetElo,
                        OpponentModel*);
bool should_make_mistake() const;
```

These extend the existing `Skill` interface rather than adding a separate class, because
`Skill` already owns the `level` and `best` fields that control move selection. The CNN
path is a drop-in replacement for the existing `pick_best()` random selection.

**4. `Worker` gains a public `OpponentModel*` field**

```cpp
OpponentModel* opponentModel = nullptr;
```

Placed in the public section (alongside the history arrays) so that `iterative_deepening`
can access it directly. It is initialised from `sharedState.opponentModel` in the
constructor, so each worker thread automatically gets the right pointer.

---

### `src/search.cpp`

**What changed:** Four modifications.

**1. Worker constructor initialiser list — `opponentModel` moved to correct position**

```cpp
sharedHistory(sharedState.sharedHistories.at(...)),
opponentModel(sharedState.opponentModel),   // ← public member, initialised first
...
networks(sharedState.networks),
refreshTable(networks[token])               // ← depends on networks
```

C++ initialises members in **declaration order**, not initialiser-list order. Because
`opponentModel` is declared in the public section (before all private members including
`networks` and `refreshTable`), it must appear before them in the initialiser list to
avoid a `-Wreorder-ctor` warning. `refreshTable` must still come after `networks` because
it calls `networks[token]` at init time.

**2. `iterative_deepening()` — mid-search CNN branch**

```cpp
if (skill.enabled() && skill.time_to_pick(rootDepth))
{
    if (opponentModel && opponentModel->is_ready() && int(options["UseOpponentModel"]))
        skill.pick_best_with_cnn(...);
    else
        skill.pick_best(rootMoves, multiPV);   // original path unchanged
}
```

`time_to_pick()` fires at a specific depth determined by the skill level. The CNN path
replaces the random selection at that moment; the fallback keeps the original behaviour
when no model is loaded or the option is off.

**3. `iterative_deepening()` — final move swap**

```cpp
if (skill.enabled())
{
    Move pickedMove = (cnn active) ? pick_best_with_cnn(...) : pick_best(...);
    std::swap(rootMoves[0], *std::find(..., pickedMove));
}
```

At the end of the search, the selected sub-optimal move is swapped to position 0 so the
UCI layer reports it as `bestmove`. Same logic as before, now with a CNN alternative.

**4. `pick_best_with_cnn()` and `should_make_mistake()` implementations**

```
should_make_mistake():
  mistake chance = (20 - level) * 2%    → 0% at full strength, 40% at level 0
  uses a separate static PRNG to avoid correlation with pick_best's PRNG

pick_best_with_cnn():
  1. If model not ready → fall back to pick_best (random)
  2. Rank all MultiPV candidates by exploitability via rank_candidate_moves()
  3. If top move has trap potential > 0.6 → play it (strong trap opportunity)
  4. If should_make_mistake() → search for a "smart mistake" via is_smart_mistake()
  5. Otherwise → play the move with the highest combined exploitability score
```

The three-tier logic (trap → mistake → exploit) reflects the strategic priority:
setting a strong trap is the best outcome; making a convincing human-like error is
second; playing the objectively most exploitable move is the default.

---

### `src/engine.h`

**What changed:** Three additions.

- `#include <memory>` — needed for `std::unique_ptr`.
- `#include "opponent_model.h"` — needed for the `OpponentModel` type.
- `void load_opponent_model(const std::string& path)` — public method declaration.
- `std::unique_ptr<OpponentModel> opponentModel` — private ownership field.

**Why `unique_ptr` in `Engine`:** The `Engine` is the single long-lived owner of the model.
`unique_ptr` ensures automatic cleanup when the engine shuts down or when the model is
replaced (e.g. `opponentModel.reset()` for an empty file path). All workers hold a raw
non-owning pointer that is valid for the lifetime of the model — safe because workers are
always stopped (via `wait_for_search_finished`) before the model is replaced.

---

### `src/engine.cpp`

**What changed:** Four additions.

**1. `#include "opponent_model.h"`**

Required so `Engine::load_opponent_model` can construct and call `OpponentModel` methods.

**2. Eight new UCI options**

| Option | Type | Default | Purpose |
|---|---|---|---|
| `OpponentElo` | spin 800–2800 | 1500 | ELO of the human opponent being faced |
| `PlayingElo` | spin 800–2800 | 1500 | ELO level the engine is simulating |
| `UseOpponentModel` | check | false | Master switch for CNN-guided move selection |
| `OpponentModelFile` | string | (empty) | Path to the model file; triggers load on change |
| `CNNWeight` | spin 0–100 | 70 | Reserved for blending CNN score with Stockfish score |
| `EnableSmartMistakes` | check | true | Allow human-like suboptimal moves |
| `EnableTrapSetting` | check | true | Prioritise moves that set tactical traps |
| `MistakeFrequency` | spin 0–100 | 15 | Percentage of moves on which to attempt a mistake |

`OpponentModelFile` has an `OnChange` callback that calls `load_opponent_model()` whenever
a GUI sends `setoption name OpponentModelFile value /path/to/model.onnx`. This matches the
pattern used by `EvalFile` for NNUE weights.

**3. `load_opponent_model()`**

```
1. wait_for_search_finished()        — stop any running search first
2. if path empty → reset model, log "disabled"
3. else → create OpponentModel if needed, call load_model(path), log result
4. resize_threads()                  — push the (new) pointer to all workers
```

`resize_threads()` recreates all worker threads, causing each new `Worker` constructor to
pull the updated `opponentModel.get()` out of `SharedState`. This is the same mechanism
used when changing the `Threads` count option.

**4. `resize_threads()` updated**

```cpp
threads.set(..., {options, threads, tt, sharedHists, networks, opponentModel.get()}, ...);
```

The sixth argument (`opponentModel.get()`) passes the raw pointer into `SharedState`.
Before this change the brace-initialiser only listed five arguments, matching the original
five-parameter `SharedState` constructor. The new constructor parameter has a default of
`nullptr`, so any other call site that was not updated would still compile.

---

## New Scripts

### `test_baseline.sh`

Runs Stockfish with traditional handicapping (depth-limited + MultiPV random selection,
`UseOpponentModel false`). This is the **control group** for the experiment.

### `test_cnn_enhanced.sh`

Runs Stockfish with the CNN enabled (`UseOpponentModel true`, `OpponentModelFile` set to
your model). This is the **treatment group**. Accepts arguments for model path, ELO
targets, and binary location.

### `analyze_results.py`

Reads two PGN files (baseline vs CNN) and prints win/loss/draw records, win rates, and
the percentage-point improvement. Flags when the sample is too small to be statistically
reliable. Requires `pip install chess`.

---

## What Remains To Be Implemented

The scaffolding is complete and the engine compiles cleanly. The two stubs that must be
filled in before the CNN does anything useful are both in `src/opponent_model.cpp`:

**`position_to_input(const Position& pos, float* inputBuffer)`**
Encode the board state as the CNN's input tensor. Common choices:
- AlphaZero-style: 12 piece-type/colour planes × 64 squares = 768 floats, plus
  auxiliary planes for castling rights, side to move, move count, etc.
- Stockfish-style: use the existing bitboard accessors (`pos.pieces(...)`) to populate
  the planes directly.

**`output_to_moves(const float* output, const Position& pos, responses)`**
Map the CNN's output logits to legal moves:
1. Enumerate legal moves with `MoveList<LEGAL>(pos)`.
2. Map each move to an output index (e.g. `from_sq * 64 + to_sq`, or a policy head
   encoding matching your training scheme).
3. Apply softmax over the legal-move subset if the network outputs raw logits.
4. Sort descending by probability and store in `responses`.

Once these two functions are implemented and `load_model()` returns `true`, the full
pipeline activates automatically — no other files need to change.

---

## Quick Start

```bash
# Build
cd src && make build ARCH=apple-silicon

# Run baseline test (control group)
./test_baseline.sh

# Run CNN-enhanced test (treatment group, after loading your model)
./test_cnn_enhanced.sh /path/to/opponent_model.onnx 1500 1500

# Compare results
python analyze_results.py baseline_results.pgn cnn_results.pgn
```

To enable the model from a GUI or UCI command line:
```
setoption name UseOpponentModel value true
setoption name OpponentModelFile value /absolute/path/to/model.onnx
setoption name OpponentElo value 1500
setoption name PlayingElo value 1500
```
