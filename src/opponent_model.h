#ifndef OPPONENT_MODEL_H_INCLUDED
#define OPPONENT_MODEL_H_INCLUDED

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "position.h"
#include "types.h"

namespace Stockfish {

// Represents a predicted opponent response
struct OpponentResponse {
    Move  move;
    float probability;  // 0.0 to 1.0 - how likely opponent plays this
};

// Represents move quality from opponent's perspective
struct MoveExploitability {
    Move  move;
    float difficulty;     // 0.0-1.0: how hard for opponent to find refutation
    float blunderRate;    // 0.0-1.0: how often opponent makes mistake after this
    float trapPotential;  // 0.0-1.0: likelihood this sets a trap opponent falls for
    float expectedValue;  // Combined metric for move selection
};

class OpponentModel {
   public:
    OpponentModel();
    ~OpponentModel();

    // Load ONNX model from file. Also loads the companion vocab .txt file
    // (same path, .onnx extension replaced with _vocab.txt).
    bool load_model(const std::string& modelPath);

    // Main prediction function: given position, predict opponent's likely moves.
    // Returns top N moves with probabilities (sorted by probability descending).
    std::vector<OpponentResponse> predict_responses(const Position& pos,
                                                    int             opponentElo,
                                                    int             topN = 5) const;

    // Single move probability
    float get_move_probability(const Position& pos, Move move, int opponentElo) const;

    // Evaluate which of OUR candidate moves is best to play.
    // pos is non-const: moves are made/undone internally.
    std::vector<MoveExploitability> rank_candidate_moves(Position&                pos,
                                                         const std::vector<Move>& candidates,
                                                         int opponentElo) const;

    // Returns true if candidateMove is a human-like mistake at targetElo
    // that opponentElo is unlikely to punish.
    bool is_smart_mistake(Position& pos,
                          Move      bestMove,
                          Move      candidateMove,
                          int       targetElo,
                          int       opponentElo) const;

    // Evaluate the likelihood that ourMove sets a trap the opponent will fall into.
    float evaluate_trap_potential(Position& pos, Move ourMove, int opponentElo) const;

    bool is_ready() const { return modelLoaded; }

   private:
    bool modelLoaded = false;

    // ONNX Runtime session (opaque — OrtContext defined in opponent_model.cpp
    // so ONNX headers are not pulled into every translation unit).
    struct OrtContext;
    std::unique_ptr<OrtContext> ort;

    // Move vocabulary: idx_to_move[i] = UCI string for output index i
    // move_to_idx[uci_string] = output index
    std::vector<std::string>            idx_to_move;
    std::unordered_map<std::string, int> move_to_idx_map;

    // Convert Stockfish Position to CNN input (12-plane 8×8 board encoding).
    // buf must be at least 768 floats (12 * 64).
    void position_to_input(const Position& pos, float* buf) const;

    // Convert CNN output logits to a list of (move, probability) pairs.
    // Only legal moves in pos are included; probabilities are softmax-normalised.
    void output_to_moves(const float*                   logits,
                         const Position&                pos,
                         std::vector<OpponentResponse>& responses) const;

    // Evaluate position after the opponent plays oppResponse.move.
    float evaluate_after_opponent_response(Position&               pos,
                                           const OpponentResponse& oppResponse) const;

    // Convert a Stockfish Move to a UCI string (e.g. "e2e4", "e7e8q").
    static std::string move_to_uci(Move m);
};

}  // namespace Stockfish

#endif  // OPPONENT_MODEL_H_INCLUDED
