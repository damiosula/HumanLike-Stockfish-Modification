#ifndef OPPONENT_MODEL_H_INCLUDED
#define OPPONENT_MODEL_H_INCLUDED

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "position.h"
#include "types.h"

namespace Stockfish {

struct OpponentResponse {
    Move move;
    float probability;
};

struct MoveExploitability {
    Move move;
    float avgResponseEvalGain;   
    float bestResponseEvalGain; 
    float expectedValue;
};

class OpponentModel {
   public:

    OpponentModel();
    ~OpponentModel();

    using EvalFn = std::function<Value(const Position&)>;
    using DoMoveFn = std::function<void(Position&, Move, StateInfo&)>;
    using UndoMoveFn = std::function<void(Position&, Move)>;

    bool load_model(const std::string& modelPath);

    std::vector<OpponentResponse> predict_responses(const Position& pos, int opponentElo, int topN = 5) const;

    float get_move_probability(const Position& pos, Move move, int opponentElo) const;

    std::vector<MoveExploitability> rank_candidate_moves(Position& pos, const std::vector<Move>& candidates, int opponentElo, const EvalFn& evalFn, const DoMoveFn& doMoveFn, const UndoMoveFn& undoMoveFn) const;

    bool is_smart_mistake(Position& pos, Move bestMove, Move candidateMove, int targetElo, int opponentElo, const EvalFn& evalFn, const DoMoveFn& doMoveFn, const UndoMoveFn& undoMoveFn) const;

    float evaluate_trap_potential(Position& pos, Move ourMove, int opponentElo, const EvalFn& evalFn, const DoMoveFn& doMoveFn, const UndoMoveFn& undoMoveFn) const;

    bool is_ready() const { return modelLoaded; }

    void set_weights(float avgExploitGainWeight, float BestResponseEvalGainWeight) {
        avg_response_eval_gain_weight = avgExploitGainWeight;
        best_response_eval_gain_weight = BestResponseEvalGainWeight;
    }

   private:
    float avg_response_eval_gain_weight = 0.5f;
    float best_response_eval_gain_weight = 0.5f;
    bool modelLoaded = false;

    struct OrtContext;
    std::unique_ptr<OrtContext> ort;

    std::vector<std::string> idx_to_move;
    std::unordered_map<std::string, int> move_to_idx_map;

    void position_to_input(const Position& pos, float* buf) const;

    void output_to_moves(const float* logits, const Position& pos, std::vector<OpponentResponse>& responses) const;

    float evaluate_after_opponent_response(Position& pos, const OpponentResponse& oppResponse, const EvalFn& evalFn, const DoMoveFn& doMoveFn, const UndoMoveFn& undoMoveFn, Value evalAfterCandidate) const;

    static std::string move_to_uci(Move m);
};

}  // namespace Stockfish

#endif  // OPPONENT_MODEL_H_INCLUDED
