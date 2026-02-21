#include "opponent_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "evaluate.h"
#include "movegen.h"

namespace Stockfish {

OpponentModel::OpponentModel() {}

OpponentModel::~OpponentModel() {
    // Clean up framework-specific model resources here when CNN is integrated.
    // e.g. for ONNX: delete session; session = nullptr;
}

bool OpponentModel::load_model(const std::string& /*modelPath*/) {
    // TODO: Load CNN model from file.
    // Choose one framework and implement here:
    //   ONNX Runtime: Ort::Session session(*env, modelPath.c_str(), sessionOptions);
    //   LibTorch:     module = torch::jit::load(modelPath);
    modelLoaded = false;
    return modelLoaded;
}

// Predict the top N moves an opponent at opponentElo is likely to play.
// Returns an empty vector until the CNN is integrated.
std::vector<OpponentResponse>
OpponentModel::predict_responses(const Position& /*pos*/, int /*opponentElo*/, int /*topN*/) const {
    std::vector<OpponentResponse> responses;

    if (!modelLoaded)
        return responses;

    // TODO: Implement CNN inference:
    //   1. position_to_input(pos, inputBuffer)
    //   2. Run forward pass
    //   3. output_to_moves(outputBuffer, pos, responses)
    //   4. Sort by probability, truncate to topN

    return responses;
}

float OpponentModel::get_move_probability(const Position& pos, Move move, int opponentElo) const {
    if (!modelLoaded)
        return 0.0f;

    auto responses = predict_responses(pos, opponentElo, 20);
    for (const auto& resp : responses)
        if (resp.move == move)
            return resp.probability;

    return 0.0f;
}

// Rank candidate moves by how exploitable they are against the opponent.
// Uses do/undo moves on pos so no Position copy is required.
std::vector<MoveExploitability>
OpponentModel::rank_candidate_moves(Position& pos, const std::vector<Move>& candidates,
                                     int opponentElo) const {
    std::vector<MoveExploitability> rankings;

    if (!modelLoaded || candidates.empty())
        return rankings;

    for (const Move& move : candidates)
    {
        MoveExploitability exploit;
        exploit.move = move;

        // Apply our move temporarily to evaluate trap potential and responses
        StateInfo st;
        pos.do_move(move, st);

        exploit.trapPotential = evaluate_trap_potential(pos, Move::none(), opponentElo);

        auto responses = predict_responses(pos, opponentElo, 5);

        float expectedOutcome = 0.0f;
        float totalProb       = 0.0f;
        float blunderProb     = 0.0f;

        for (const auto& resp : responses)
        {
            float outcome = evaluate_after_opponent_response(pos, resp);
            expectedOutcome += resp.probability * outcome;
            totalProb += resp.probability;

            if (outcome > 0.5f)  // We gain significant advantage after their response
                blunderProb += resp.probability;
        }

        pos.undo_move(move);

        exploit.blunderRate  = blunderProb;
        exploit.difficulty   = 1.0f - (totalProb > 0 ? expectedOutcome / totalProb : 0.5f);
        exploit.expectedValue = 0.4f * exploit.trapPotential + 0.4f * exploit.blunderRate
                              + 0.2f * exploit.difficulty;

        rankings.push_back(exploit);
    }

    std::sort(rankings.begin(), rankings.end(),
              [](const MoveExploitability& a, const MoveExploitability& b) {
                  return a.expectedValue > b.expectedValue;
              });

    return rankings;
}

// Returns true if candidateMove is a "smart" human-like mistake:
//   - Players at targetElo sometimes make it (probability > 5%)
//   - The opponent at opponentElo is unlikely to punish it immediately
bool OpponentModel::is_smart_mistake(Position& pos,
                                      Move      /*bestMove*/,
                                      Move      candidateMove,
                                      int       targetElo,
                                      int       opponentElo) const {
    if (!modelLoaded)
        return false;

    // Check how often target-ELO players make this move
    float targetProb = get_move_probability(pos, candidateMove, targetElo);
    if (targetProb < 0.05f)
        return false;  // Move too rare for target ELO — looks inhuman

    // Play candidateMove and check if the opponent can easily punish it
    StateInfo st;
    pos.do_move(candidateMove, st);

    auto responses = predict_responses(pos, opponentElo, 3);

    bool opponentPunishes = false;
    for (const auto& resp : responses)
    {
        StateInfo st2;
        pos.do_move(resp.move, st2);
        // After opponent plays, evaluate from our (side_to_move) perspective.
        // A strongly negative value means the opponent successfully punished us.
        Value eval = Value(Eval::simple_eval(pos));
        pos.undo_move(resp.move);

        if (eval < -300 && resp.probability > 0.4f)
        {
            opponentPunishes = true;
            break;
        }
    }

    pos.undo_move(candidateMove);
    return !opponentPunishes;
}

// Evaluate how likely ourMove is to set a trap the opponent will fall into.
// When called internally from rank_candidate_moves, ourMove is Move::none()
// because the move has already been applied to pos.
float OpponentModel::evaluate_trap_potential(Position& pos, Move ourMove, int opponentElo) const {
    if (!modelLoaded)
        return 0.0f;

    // If a move is provided, apply it first
    StateInfo stOur;
    bool      didOurMove = ourMove != Move::none();
    if (didOurMove)
        pos.do_move(ourMove, stOur);

    auto responses = predict_responses(pos, opponentElo, 10);

    float trapScore    = 0.0f;
    int   plausible    = 0;
    int   badResponses = 0;

    for (const auto& resp : responses)
    {
        if (resp.probability <= 0.05f)
            continue;

        ++plausible;

        StateInfo st2;
        pos.do_move(resp.move, st2);
        // After opponent plays, positive eval = we still have advantage = bad for opponent
        Value eval = Value(Eval::simple_eval(pos));
        pos.undo_move(resp.move);

        if (eval > 100)
        {
            ++badResponses;
            trapScore += resp.probability;
        }
    }

    if (didOurMove)
        pos.undo_move(ourMove);

    // High trap potential: many plausible responses and many are poor for the opponent
    if (plausible > 3 && badResponses >= 2)
        return std::min(1.0f, trapScore * 1.5f);

    return trapScore;
}

// Evaluate the position after the opponent plays oppResponse.move.
// pos must have our preceding move already applied; it will be restored on return.
float OpponentModel::evaluate_after_opponent_response(Position&               pos,
                                                       const OpponentResponse& oppResponse) const {
    StateInfo st;
    pos.do_move(oppResponse.move, st);

    // Simple material evaluation from our (side_to_move) perspective.
    // Positive = good for us. Normalised to [-1, 1] via tanh.
    float eval = static_cast<float>(Eval::simple_eval(pos)) / 300.0f;

    pos.undo_move(oppResponse.move);

    return std::tanh(eval);
}

void OpponentModel::position_to_input(const Position& /*pos*/, float* /*inputBuffer*/) const {
    // TODO: Encode the position as CNN input.
    // Common approaches:
    //   - Multi-channel 8×8 planes (AlphaZero style): piece type × colour × square
    //   - Bitboard representation flattened to float array
    //   - Piece-centric feature list
}

void OpponentModel::output_to_moves(const float* /*output*/, const Position& /*pos*/,
                                     std::vector<OpponentResponse>& /*responses*/) const {
    // TODO: Convert CNN output logits/probabilities to a list of (move, probability) pairs.
    //   1. Enumerate legal moves
    //   2. Map each move to an output index (e.g. (from_sq * 64 + to_sq) encoding)
    //   3. Apply softmax over legal move indices if needed
    //   4. Sort by probability descending
}

}  // namespace Stockfish
