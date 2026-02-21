#include "opponent_model.h"
#include "movegen.h"
#include "evaluate.h"
#include <algorithm>
#include <cstring>
#include <cmath>

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

// NEW: Core function for smart handicapping
std::vector<MoveExploitability> OpponentModel::rank_candidate_moves(
    const Position& pos,
    const std::vector<Move>& candidates,
    int opponentElo
) const {
    std::vector<MoveExploitability> rankings;
    
    if (!modelLoaded || candidates.empty()) {
        return rankings;
    }
    
    // For each candidate move, evaluate how exploitable it is
    for (const Move& move : candidates) {
        MoveExploitability exploit;
        exploit.move = move;
        
        // Evaluate trap potential
        exploit.trapPotential = evaluate_trap_potential(pos, move, opponentElo);
        
        // Calculate expected value based on opponent responses
        StateInfo st;
        Position posCopy = pos;
        posCopy.do_move(move, st);
        
        auto responses = predict_responses(posCopy, opponentElo, 5);
        
        // Calculate weighted outcome
        float expectedOutcome = 0.0f;
        float totalProb = 0.0f;
        float blunderProb = 0.0f;
        
        for (const auto& resp : responses) {
            float outcome = evaluate_after_opponent_response(posCopy, move, resp);
            expectedOutcome += resp.probability * outcome;
            totalProb += resp.probability;
            
            // Track if opponent likely to blunder
            if (outcome > 2.0f) {  // We gain significant advantage
                blunderProb += resp.probability;
            }
        }
        
        posCopy.undo_move(move);
        
        exploit.blunderRate = blunderProb;
        exploit.difficulty = 1.0f - (totalProb > 0 ? expectedOutcome / totalProb : 0.5f);
        
        // Combined metric: balance trap potential, blunder rate, and difficulty
        exploit.expectedValue = 
            0.4f * exploit.trapPotential +
            0.4f * exploit.blunderRate +
            0.2f * exploit.difficulty;
        
        rankings.push_back(exploit);
    }
    
    // Sort by expected value (highest first)
    std::sort(rankings.begin(), rankings.end(),
        [](const MoveExploitability& a, const MoveExploitability& b) {
            return a.expectedValue > b.expectedValue;
        });
    
    return rankings;
}

// NEW: Determine if a "mistake" is human-like or stupid
bool OpponentModel::is_smart_mistake(
    const Position& pos,
    Move bestMove,
    Move candidateMove,
    int targetElo,
    int opponentElo
) const {
    if (!modelLoaded) {
        return false;  // Without model, can't evaluate
    }
    
    // A "smart mistake" is one that:
    // 1. A player at targetElo might reasonably make
    // 2. Is hard for opponentElo to punish
    // 3. Doesn't lose material immediately
    
    // Check if target ELO players make this move
    float targetProb = get_move_probability(pos, candidateMove, targetElo);
    
    if (targetProb < 0.05f) {
        return false;  // Move too rare for target ELO
    }
    
    // Check opponent's ability to punish
    StateInfo st;
    Position posCopy = pos;
    posCopy.do_move(candidateMove, st);
    
    auto responses = predict_responses(posCopy, opponentElo, 3);
    
    // If opponent likely finds the refutation, this is a stupid mistake
    for (const auto& resp : responses) {
        StateInfo st2;
        posCopy.do_move(resp.move, st2);
        Value eval = -evaluate(posCopy);
        posCopy.undo_move(resp.move);
        
        if (eval < -300 && resp.probability > 0.4f) {
            posCopy.undo_move(candidateMove);
            return false;  // Opponent likely punishes hard
        }
    }
    
    posCopy.undo_move(candidateMove);
    return true;  // Move is reasonably human-like
}

// NEW: Evaluate trap potential
float OpponentModel::evaluate_trap_potential(
    const Position& pos,
    Move ourMove,
    int opponentElo
) const {
    if (!modelLoaded) {
        return 0.0f;
    }
    
    StateInfo st;
    Position posCopy = pos;
    posCopy.do_move(ourMove, st);
    
    auto responses = predict_responses(posCopy, opponentElo, 10);
    
    float trapScore = 0.0f;
    
    // A move has trap potential if:
    // - Opponent has multiple plausible responses
    // - Several of them lead to bad outcomes for opponent
    // - The "best" response is not obvious
    
    int plausibleMoves = 0;
    int badResponses = 0;
    
    for (const auto& resp : responses) {
        if (resp.probability > 0.05f) {
            plausibleMoves++;
            
            StateInfo st2;
            posCopy.do_move(resp.move, st2);
            Value eval = -evaluate(posCopy);
            posCopy.undo_move(resp.move);
            
            // If this response is bad for opponent
            if (eval > 100) {
                badResponses++;
                trapScore += resp.probability;
            }
        }
    }
    
    posCopy.undo_move(ourMove);
    
    // High trap potential: multiple plausible moves, many are bad
    if (plausibleMoves > 3 && badResponses >= 2) {
        return std::min(1.0f, trapScore * 1.5f);
    }
    
    return trapScore;
}

float OpponentModel::evaluate_after_opponent_response(
    Position& pos,
    Move ourMove,
    const OpponentResponse& oppResponse
) const {
    // Quick evaluation of position after opponent responds
    StateInfo st;
    pos.do_move(oppResponse.move, st);
    
    // Use Stockfish's evaluate function
    Value eval = evaluate(pos);
    
    pos.undo_move(oppResponse.move);
    
    // Convert to normalized score (-1.0 to 1.0)
    return std::tanh(eval / 300.0f);
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