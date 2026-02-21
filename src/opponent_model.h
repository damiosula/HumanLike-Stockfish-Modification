#ifndef OPPONENT_MODEL_H_INCLUDED
#define OPPONENT_MODEL_H_INCLUDED

#include <vector>
#include "types.h"
#include "position.h"

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

    // Initialize CNN model from file
    bool load_model(const std::string& modelPath);

    // Main prediction function: given position, predict opponent's likely moves.
    // Returns top N moves with probabilities (sorted by probability descending).
    // pos is non-const because inference may temporarily modify it.
    std::vector<OpponentResponse> predict_responses(const Position& pos,
                                                    int             opponentElo,
                                                    int             topN = 5) const;

    // Single move probability
    float get_move_probability(const Position& pos, Move move, int opponentElo) const;

    // Evaluate which of OUR candidate moves is best to play.
    // Considers what the opponent is likely to do in response.
    // Returns moves ranked by exploitability (best first from our perspective).
    // pos is taken by non-const reference so moves can be made/undone internally.
    std::vector<MoveExploitability> rank_candidate_moves(Position&                pos,
                                                         const std::vector<Move>& candidates,
                                                         int opponentElo) const;

    // Smart handicap: returns true if candidateMove is a human-like mistake worth making
    // rather than a stupid computer blunder.
    // pos is taken by non-const reference so moves can be made/undone internally.
    bool is_smart_mistake(Position& pos,
                          Move      bestMove,
                          Move      candidateMove,
                          int       targetElo,
                          int       opponentElo) const;

    // Evaluate the likelihood that ourMove sets a trap the opponent will fall into.
    // pos is taken by non-const reference so moves can be made/undone internally.
    float evaluate_trap_potential(Position& pos, Move ourMove, int opponentElo) const;

    // Check if model is loaded and ready
    bool is_ready() const { return modelLoaded; }

   private:
    bool modelLoaded = false;
    // void* cnnModel — add a framework-specific model pointer here when integrating
    //                  ONNX Runtime (Ort::Session*) or LibTorch (torch::jit::Module*)

    // Convert Stockfish Position to CNN input format
    void position_to_input(const Position& pos, float* inputBuffer) const;

    // Convert CNN output logits/probabilities to legal move list
    void output_to_moves(const float*                   output,
                         const Position&                pos,
                         std::vector<OpponentResponse>& responses) const;

    // Evaluate position after the opponent plays oppResponse.move.
    // pos must already have ourMove applied; it will be restored on return.
    float evaluate_after_opponent_response(Position&               pos,
                                           const OpponentResponse& oppResponse) const;
};

}  // namespace Stockfish

#endif  // OPPONENT_MODEL_H_INCLUDED
