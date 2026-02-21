# Stockfish CNN Opponent Modeling Integration Guide

## Overview
This guide provides a step-by-step implementation plan for integrating your CNN-based opponent move prediction model into Stockfish to improve play against human opponents of specific ELO ratings.

---

## Phase 1: Create the CNN Predictor Interface (Foundation)

### Step 1.1: Create the OpponentModel Class

**File to create:** `src/opponent_model.h`

```cpp
#ifndef OPPONENT_MODEL_H_INCLUDED
#define OPPONENT_MODEL_H_INCLUDED

#include <vector>
#include "types.h"
#include "position.h"

namespace Stockfish {

// Represents a predicted opponent response
struct OpponentResponse {
    Move move;
    float probability;  // 0.0 to 1.0
};

class OpponentModel {
public:
    OpponentModel();
    ~OpponentModel();
    
    // Initialize CNN model from file
    bool load_model(const std::string& modelPath);
    
    // Main prediction function: given position, predict opponent's likely moves
    // Returns top N moves with probabilities (sorted by probability descending)
    std::vector<OpponentResponse> predict_responses(
        const Position& pos,
        int opponentElo,
        int topN = 5
    ) const;
    
    // Single move probability
    float get_move_probability(
        const Position& pos,
        Move move,
        int opponentElo
    ) const;
    
    // Check if model is loaded and ready
    bool is_ready() const { return modelLoaded; }
    
private:
    bool modelLoaded = false;
    
    // Your CNN implementation details
    // This will depend on your framework (TensorFlow, PyTorch, ONNX, etc.)
    void* cnnModel = nullptr;  // Pointer to your model
    
    // Convert Stockfish Position to CNN input format
    void position_to_input(const Position& pos, float* inputBuffer) const;
    
    // Convert CNN output to move probabilities
    void output_to_moves(const float* output, 
                        const Position& pos,
                        std::vector<OpponentResponse>& responses) const;
};

} // namespace Stockfish

#endif // OPPONENT_MODEL_H_INCLUDED
```

### Step 1.2: Create the Implementation Stub

**File to create:** `src/opponent_model.cpp`

```cpp
#include "opponent_model.h"
#include "movegen.h"
#include <algorithm>
#include <cstring>

namespace Stockfish {

OpponentModel::OpponentModel() {
    // Constructor
}

OpponentModel::~OpponentModel() {
    // Clean up CNN model
    if (cnnModel) {
        // Free your model resources
    }
}

bool OpponentModel::load_model(const std::string& modelPath) {
    // TODO: Load your CNN model from file
    // This depends on your framework:
    // - TensorFlow: load saved_model
    // - PyTorch: load traced/scripted model
    // - ONNX: load .onnx file
    
    // For now, return false (stub)
    modelLoaded = false;
    return modelLoaded;
}

std::vector<OpponentResponse> OpponentModel::predict_responses(
    const Position& pos,
    int opponentElo,
    int topN
) const {
    std::vector<OpponentResponse> responses;
    
    if (!modelLoaded) {
        return responses;
    }
    
    // TODO: Implement CNN inference
    // 1. Convert position to CNN input format
    // 2. Run CNN forward pass
    // 3. Convert output to move probabilities
    // 4. Return top N moves
    
    // STUB: For now, return empty
    // You'll implement this after choosing your CNN framework
    
    return responses;
}

float OpponentModel::get_move_probability(
    const Position& pos,
    Move move,
    int opponentElo
) const {
    if (!modelLoaded) {
        return 0.0f;
    }
    
    // Get all predictions and find this specific move
    auto responses = predict_responses(pos, opponentElo, 20);
    
    for (const auto& resp : responses) {
        if (resp.move == move) {
            return resp.probability;
        }
    }
    
    return 0.0f;  // Move not in top predictions
}

void OpponentModel::position_to_input(const Position& pos, float* inputBuffer) const {
    // TODO: Convert Stockfish position to your CNN's input format
    // Common approaches:
    // - Bitboard representation
    // - Piece-centric encoding
    // - Multi-channel 8x8 image (like AlphaZero)
}

void OpponentModel::output_to_moves(
    const float* output,
    const Position& pos,
    std::vector<OpponentResponse>& responses
) const {
    // TODO: Convert CNN output logits/probabilities to legal moves
    // 1. Get all legal moves
    // 2. Map CNN output indices to moves
    // 3. Sort by probability
    // 4. Apply softmax if needed
}

} // namespace Stockfish
```

---

## Phase 2: Integrate into Stockfish's Thread System

### Step 2.1: Add OpponentModel to Worker Thread

**File to modify:** `src/thread.h`

Find the `Worker` class definition and add:

```cpp
class Worker {
    // ... existing members ...
    
public:
    // Add these new members
    OpponentModel* opponentModel;
    int targetOpponentElo;
    bool useOpponentModel;
    
    // Constructor modification needed
    Worker(...existing params...) {
        // ... existing initialization ...
        opponentModel = nullptr;
        targetOpponentElo = 0;
        useOpponentModel = false;
    }
};
```

### Step 2.2: Add UCI Option for Opponent ELO

**File to modify:** `src/ucioption.cpp`

Find the `init()` function and add:

```cpp
void init(OptionsMap& o) {
    // ... existing options ...
    
    // Add opponent modeling options
    o["OpponentElo"]      << Option(0, 0, 3000);
    o["UseOpponentModel"] << Option(false);
    o["OpponentModelFile"] << Option("<empty>");
    
    // ... rest of options ...
}
```

### Step 2.3: Load Model When UCI Option Changes

**File to modify:** `src/uci.cpp` or `src/ucioption.cpp`

Add option change handler:

```cpp
// In the option change handler section
void on_opponent_model_change(const Option& o) {
    std::string modelFile = std::string(o);
    
    if (modelFile == "<empty>") {
        // Disable opponent model
        for (Thread* th : threads) {
            Worker* w = static_cast<Worker*>(th);
            w->useOpponentModel = false;
        }
        return;
    }
    
    // Load model for all worker threads
    for (Thread* th : threads) {
        Worker* w = static_cast<Worker*>(th);
        
        if (w->opponentModel == nullptr) {
            w->opponentModel = new OpponentModel();
        }
        
        if (w->opponentModel->load_model(modelFile)) {
            w->useOpponentModel = true;
            sync_cout << "info string Opponent model loaded successfully" << sync_endl;
        } else {
            w->useOpponentModel = false;
            sync_cout << "info string Failed to load opponent model" << sync_endl;
        }
    }
}

// Connect the handler to the option
o["OpponentModelFile"] << Option("<empty>", on_opponent_model_change);
```

---

## Phase 3: Implement Expected Value Calculation

### Step 3.1: Create Expected Value Function

**File to modify:** `src/search.cpp`

Add this function in the anonymous namespace at the top:

```cpp
namespace {

// ... existing helper functions ...

// Calculate expected value of a move considering opponent's likely responses
Value expected_value_vs_opponent(
    Position& pos,
    Move move,
    Value baseValue,
    const Worker& worker,
    int depth
) {
    if (!worker.useOpponentModel || !worker.opponentModel->is_ready()) {
        return baseValue;  // Fall back to normal evaluation
    }
    
    if (depth <= 0) {
        return baseValue;  // Don't use at leaves
    }
    
    StateInfo st;
    pos.do_move(move, st);
    
    // Get opponent's likely responses
    auto responses = worker.opponentModel->predict_responses(
        pos,
        worker.targetOpponentElo,
        5  // Top 5 moves
    );
    
    if (responses.empty()) {
        pos.undo_move(move);
        return baseValue;
    }
    
    // Calculate weighted average of outcomes
    Value expectedVal = VALUE_ZERO;
    float totalProb = 0.0f;
    
    for (const auto& resp : responses) {
        if (!MoveList<LEGAL>(pos).contains(resp.move)) {
            continue;  // Skip illegal moves
        }
        
        StateInfo st2;
        pos.do_move(resp.move, st2);
        
        // Quick evaluation of resulting position
        Value val = -evaluate(pos);
        
        pos.undo_move(resp.move);
        
        expectedVal += Value(resp.probability * val);
        totalProb += resp.probability;
    }
    
    pos.undo_move(move);
    
    // Normalize and blend with base value
    if (totalProb > 0.0f) {
        expectedVal = Value(expectedVal / totalProb);
        
        // Blend: 70% expected value, 30% base value
        // This prevents over-relying on opponent model
        return Value(0.7 * expectedVal + 0.3 * baseValue);
    }
    
    return baseValue;
}

} // namespace
```

---

## Phase 4: Apply at Root (Main Integration Point)

### Step 4.1: Modify Root Move Selection

**File to modify:** `src/search.cpp`

Find the `iterative_deepening()` function (around line 220-500) and locate where root moves are scored. Add expected value calculation:

```cpp
// Around line 400-450, after rootMoves are searched
// Find the loop that processes rootMoves

for (RootMove& rm : rootMoves) {
    // Existing code calculates rm.score
    // Add expected value adjustment
    
    if (worker.useOpponentModel && worker.opponentModel->is_ready()) {
        Value expectedVal = expected_value_vs_opponent(
            rootPos,
            rm.pv[0],
            rm.score,
            *this,
            depth
        );
        
        // Store both values for analysis
        rm.baseScore = rm.score;
        rm.score = expectedVal;  // Use expected value for move selection
    }
}
```

### Step 4.2: Add Fields to RootMove

**File to modify:** `src/search.h`

Find the `RootMove` struct and add:

```cpp
struct RootMove {
    // ... existing members ...
    
    Value baseScore = -VALUE_INFINITE;  // Score without opponent model
    
    // ... existing methods ...
};
```

---

## Phase 5: Enhanced Pruning (Optional But Powerful)

### Step 5.1: Opponent-Aware Futility Pruning

**File to modify:** `src/search.cpp`

Find futility pruning section (around line 1072-1114):

```cpp
// Futility pruning
if (futilityValue <= alpha && !ss->inCheck && moveCount > 0) {
    // Standard pruning logic
    
    // NEW: Check if this move might exploit opponent weakness
    if (worker.useOpponentModel) {
        // If opponent likely to blunder after this move, don't prune
        float trapPotential = evaluate_trap_potential(pos, move, worker);
        
        if (trapPotential > 0.3f) {  // High chance opponent makes mistake
            continue;  // Don't prune - search this move
        }
    }
    
    // Otherwise, apply standard pruning
    continue;
}
```

### Step 5.2: Add Trap Potential Function

```cpp
namespace {

// Evaluate if a move sets a trap opponent might fall for
float evaluate_trap_potential(
    const Position& pos,
    Move move,
    const Worker& worker
) {
    if (!worker.useOpponentModel || !worker.opponentModel->is_ready()) {
        return 0.0f;
    }
    
    StateInfo st;
    pos.do_move(move, st);
    
    auto responses = worker.opponentModel->predict_responses(
        pos,
        worker.targetOpponentElo,
        10
    );
    
    pos.undo_move(move);
    
    if (responses.empty()) {
        return 0.0f;
    }
    
    // Sum probability of bad moves
    float blunderProb = 0.0f;
    
    for (const auto& resp : responses) {
        StateInfo st2;
        pos.do_move(move, st);
        
        if (MoveList<LEGAL>(pos).contains(resp.move)) {
            pos.do_move(resp.move, st2);
            Value val = -evaluate(pos);
            pos.undo_move(resp.move);
            
            // If this response loses material or position
            if (val < -100) {  // Bad for opponent
                blunderProb += resp.probability;
            }
        }
        
        pos.undo_move(move);
    }
    
    return blunderProb;
}

} // namespace
```

---

## Phase 6: Testing and Validation

### Step 6.1: Create Test Position

**File to create:** `test_opponent_model.cpp`

```cpp
#include "position.h"
#include "opponent_model.h"
#include "uci.h"
#include <iostream>

using namespace Stockfish;

int main() {
    UCIEngine uci;
    
    // Create test position
    Position pos;
    StateInfo st;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &st);
    
    // Load model
    OpponentModel model;
    if (!model.load_model("path/to/your/model.onnx")) {
        std::cout << "Failed to load model\n";
        return 1;
    }
    
    // Test prediction
    auto responses = model.predict_responses(pos, 1500, 5);
    
    std::cout << "Top 5 predicted moves for 1500 ELO:\n";
    for (const auto& resp : responses) {
        std::cout << UCI::move(resp.move, false) 
                  << " - probability: " << resp.probability << "\n";
    }
    
    return 0;
}
```

### Step 6.2: Compile and Test

```bash
# Add to Makefile
OBJS += opponent_model.o

# Compile
make build ARCH=x86-64-modern

# Test with UCI
./stockfish
setoption name OpponentElo value 1500
setoption name OpponentModelFile value /path/to/model.onnx
setoption name UseOpponentModel value true
position startpos
go depth 10
```

---

## Phase 7: CNN Model Integration (Framework-Specific)

### Option A: ONNX Runtime (Recommended - Cross-Platform)

```cpp
// In opponent_model.cpp
#include <onnxruntime_cxx_api.h>

class OpponentModel {
private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    
public:
    bool load_model(const std::string& modelPath) {
        try {
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "StockfishOppModel");
            
            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads(1);
            
            session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);
            
            modelLoaded = true;
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX error: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::vector<OpponentResponse> predict_responses(
        const Position& pos,
        int opponentElo,
        int topN
    ) const {
        // Prepare input tensor
        std::vector<float> inputData(INPUT_SIZE);
        position_to_input(pos, inputData.data());
        
        // Add ELO as feature
        inputData.push_back(opponentElo / 3000.0f);  // Normalize
        
        // Run inference
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<int64_t> inputShape = {1, INPUT_SIZE + 1};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputData.data(),
            inputData.size(),
            inputShape.data(),
            inputShape.size()
        );
        
        const char* inputNames[] = {"input"};
        const char* outputNames[] = {"output"};
        
        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );
        
        // Process output
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        size_t outputSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::vector<OpponentResponse> responses;
        output_to_moves(outputData, pos, responses);
        
        // Sort and return top N
        std::sort(responses.begin(), responses.end(),
            [](const OpponentResponse& a, const OpponentResponse& b) {
                return a.probability > b.probability;
            });
        
        if (responses.size() > topN) {
            responses.resize(topN);
        }
        
        return responses;
    }
};
```

### Option B: LibTorch (PyTorch C++)

```cpp
#include <torch/script.h>

class OpponentModel {
private:
    torch::jit::script::Module module;
    
public:
    bool load_model(const std::string& modelPath) {
        try {
            module = torch::jit::load(modelPath);
            module.eval();
            modelLoaded = true;
            return true;
        } catch (const c10::Error& e) {
            std::cerr << "PyTorch error: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::vector<OpponentResponse> predict_responses(
        const Position& pos,
        int opponentElo,
        int topN
    ) const {
        // Create input tensor
        std::vector<float> inputData(INPUT_SIZE);
        position_to_input(pos, inputData.data());
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor inputTensor = torch::from_blob(
            inputData.data(),
            {1, INPUT_SIZE},
            options
        ).clone();
        
        // Add ELO
        torch::Tensor eloTensor = torch::tensor({opponentElo / 3000.0f}, options);
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        inputs.push_back(eloTensor);
        
        // Forward pass
        auto output = module.forward(inputs).toTensor();
        
        // Convert to responses
        auto outputAccessor = output.accessor<float, 2>();
        
        std::vector<OpponentResponse> responses;
        // ... convert output to moves ...
        
        return responses;
    }
};
```

---

## Phase 8: Debugging and Monitoring

### Step 8.1: Add Logging

```cpp
// Add to search.cpp in expected_value_vs_opponent()

if (worker.options["Debug"]) {
    sync_cout << "info string Expected value for " 
              << UCI::move(move, pos.is_chess960())
              << " base: " << baseValue
              << " expected: " << expectedVal
              << sync_endl;
    
    for (const auto& resp : responses) {
        sync_cout << "info string   Opponent might play "
                  << UCI::move(resp.move, pos.is_chess960())
                  << " (" << (resp.probability * 100) << "%)"
                  << sync_endl;
    }
}
```

### Step 8.2: Add UCI Info Output

```cpp
// In iterative_deepening(), add info about opponent model usage

if (useOpponentModel) {
    sync_cout << "info string Using opponent model for ELO " 
              << targetOpponentElo 
              << sync_endl;
}
```

---

## Phase 9: Performance Optimization

### Step 9.1: Cache CNN Predictions

```cpp
// Add to opponent_model.h
#include <unordered_map>

class OpponentModel {
private:
    // Cache for position predictions
    struct CacheEntry {
        std::vector<OpponentResponse> responses;
        uint64_t timestamp;
    };
    
    mutable std::unordered_map<Key, CacheEntry> predictionCache;
    const size_t MAX_CACHE_SIZE = 10000;
    
public:
    std::vector<OpponentResponse> predict_responses(
        const Position& pos,
        int opponentElo,
        int topN
    ) const {
        // Check cache first
        Key posKey = pos.key();
        auto it = predictionCache.find(posKey);
        
        if (it != predictionCache.end()) {
            return it->second.responses;  // Cache hit
        }
        
        // Cache miss - run CNN
        auto responses = run_cnn_inference(pos, opponentElo);
        
        // Store in cache
        if (predictionCache.size() < MAX_CACHE_SIZE) {
            predictionCache[posKey] = {responses, 0};
        }
        
        return responses;
    }
};
```

### Step 9.2: Limit CNN Calls

```cpp
// Only use CNN at certain depths
Value expected_value_vs_opponent(
    Position& pos,
    Move move,
    Value baseValue,
    const Worker& worker,
    int depth
) {
    // Only use CNN in tactical range (depth 3-12)
    if (depth < 3 || depth > 12) {
        return baseValue;
    }
    
    // Only use at root and first few plies
    if (ss->ply > 5) {
        return baseValue;
    }
    
    // ... rest of function ...
}
```

---

## Phase 10: Validation and Tuning

### Step 10.1: Self-Play Testing

```bash
# Create test suite
cutechess-cli \
    -engine name=Stockfish-OpponentModel cmd=./stockfish \
           option."OpponentElo"=1500 \
           option."UseOpponentModel"=true \
    -engine name=Stockfish-Baseline cmd=./stockfish \
    -each tc=40/60 \
    -rounds 100 \
    -pgnout games.pgn
```

### Step 10.2: Tune Blending Parameter

```cpp
// Make this a UCI option
o["OpponentModelWeight"] << Option(70, 0, 100);

// Use in expected_value_vs_opponent:
float weight = worker.options["OpponentModelWeight"] / 100.0f;
return Value(weight * expectedVal + (1.0f - weight) * baseValue);
```

### Step 10.3: Track Statistics

```cpp
// Add to Worker class
struct OpponentModelStats {
    int predictions_made = 0;
    int cache_hits = 0;
    int predictions_correct = 0;  // Track in actual games
};

OpponentModelStats oppModelStats;
```

---

## Complete File Checklist

Files to CREATE:
- ✅ `src/opponent_model.h`
- ✅ `src/opponent_model.cpp`
- ✅ `test_opponent_model.cpp` (optional)

Files to MODIFY:
- ✅ `src/thread.h` - Add opponentModel to Worker
- ✅ `src/ucioption.cpp` - Add UCI options
- ✅ `src/uci.cpp` - Handle option changes
- ✅ `src/search.h` - Add baseScore to RootMove
- ✅ `src/search.cpp` - Main integration (expected_value, pruning)
- ✅ `Makefile` - Add opponent_model.o

---

## Implementation Timeline

### Week 1: Foundation
- Days 1-2: Create opponent_model.h/cpp stubs
- Days 3-4: Integrate into thread system
- Days 5-7: Add UCI options and test loading

### Week 2: Core Integration  
- Days 1-3: Implement expected_value_vs_opponent
- Days 4-5: Apply at root move selection
- Days 6-7: Basic testing with dummy predictions

### Week 3: CNN Integration
- Days 1-3: Choose framework (ONNX/PyTorch) and implement
- Days 4-5: Implement position_to_input for your CNN format
- Days 6-7: Test CNN predictions working end-to-end

### Week 4: Optimization & Testing
- Days 1-2: Add caching
- Days 3-4: Tune blending parameters
- Days 5-7: Self-play testing and validation

---

## Next Immediate Steps (Start Here!)

1. **Choose your CNN framework:**
   - ONNX Runtime (recommended - easiest cross-platform)
   - LibTorch (if you trained in PyTorch)
   - TensorFlow Lite (if you trained in TensorFlow)

2. **Create the basic files:**
   ```bash
   cd stockfish/src
   touch opponent_model.h opponent_model.cpp
   ```

3. **Copy the stub code** from Step 1.1 and 1.2 above

4. **Modify Makefile:**
   ```makefile
   OBJS = ... existing objects ... opponent_model.o
   ```

5. **Compile to verify:**
   ```bash
   make clean
   make build ARCH=x86-64-modern
   ```

6. **Implement position_to_input()** based on your CNN's input format

Would you like me to elaborate on any specific step or help you with the CNN framework integration details?