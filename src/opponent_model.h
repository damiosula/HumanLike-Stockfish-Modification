#ifndef OPPONENT_MODEL_H_INCLUDED
#define OPPONENT_MODEL_H_INCLUDED

#include <vector>
#include "types.h"
#include "position.h"

namespace Stockfish {

// Represents a predicted opponent response
struct OpponentResponse {
    Move move;
    float probability;  // 0.0 to 1.0 - how likely opponent plays this
};

// Represents move quality from opponent's perspective
struct MoveExploitability {
    Move move;
    float difficulty;      // 0.0-1.0: how hard for opponent to find refutation
    float blunderRate;     // 0.0-1.0: how often opponent makes mistake after this
    float trapPotential;   // 0.0-1.0: likelihood this sets a trap opponent falls for
    float expectedValue;   // Combined metric for move selection
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
    
    // NEW: Evaluate which of OUR candidate moves is best to play
    // This considers what opponent is likely to do in response
    // Returns moves ranked by exploitability (best to worst from our perspective)
    std::vector<MoveExploitability> rank_candidate_moves(
        const Position& pos,
        const std::vector<Move>& candidates,
        int opponentElo
    ) const;
    
    // NEW: Smart handicap - evaluate if we should make this "mistake"
    // Returns true if this is a "human-like" mistake worth making
    // vs. a stupid computer blunder
    bool is_smart_mistake(
        const Position& pos,
        Move bestMove,
        Move candidateMove,
        int targetElo,
        int opponentElo
    ) const;
    
    // NEW: Detect if a position sets a trap
    float evaluate_trap_potential(
        const Position& pos,
        Move ourMove,
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
    
    // Helper: Simulate opponent response and evaluate resulting position
    float evaluate_after_opponent_response(
        Position& pos,
        Move ourMove,
        const OpponentResponse& oppResponse
    ) const;
};

} // namespace Stockfish

#endif // OPPONENT_MODEL_H_INCLUDED