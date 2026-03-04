#include "opponent_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

// ONNX Runtime — only included here so other TUs don't need it
#include <onnxruntime_cxx_api.h>

#include "misc.h"
#include "movegen.h"

namespace Stockfish {

// ── OrtContext ────────────────────────────────────────────────────────────────
// Keeps all ONNX Runtime state together. Defined here so the header stays
// free of ONNX Runtime dependencies.

struct OpponentModel::OrtContext {
    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "OpponentModel"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;

    OrtContext() { opts.SetIntraOpNumThreads(1); }
};

// ── Helpers ───────────────────────────────────────────────────────────────────

// Convert a Stockfish Move to its UCI string (e.g. "e2e4", "e7e8q").
// Avoids pulling in uci.h / engine state.
std::string OpponentModel::move_to_uci(Move m) {
    auto sq_str = [](Square sq) -> std::string {
        std::string s;
        s += char('a' + file_of(sq));
        s += char('1' + rank_of(sq));
        return s;
    };

    std::string result = sq_str(m.from_sq()) + sq_str(m.to_sq());

    if (m.type_of() == PROMOTION) {
        const char promoChar[] = {'n', 'b', 'r', 'q'};
        result += promoChar[m.promotion_type() - KNIGHT];
    }
    return result;
}

// ── Construction / destruction ────────────────────────────────────────────────

OpponentModel::OpponentModel()  = default;
OpponentModel::~OpponentModel() = default;

// ── load_model ────────────────────────────────────────────────────────────────

bool OpponentModel::load_model(const std::string& modelPath) {
    modelLoaded = false;
    idx_to_move.clear();
    move_to_idx_map.clear();

    // Derive companion vocab filename: replace ".onnx" with "_vocab.txt"
    std::string vocabPath = modelPath;
    {
        auto pos = vocabPath.rfind(".onnx");
        if (pos != std::string::npos)
            vocabPath.replace(pos, 5, "_vocab.txt");
        else
            vocabPath += "_vocab.txt";
    }

    // Load vocab file (one UCI move string per line)
    {
        std::ifstream f(vocabPath);
        if (!f.is_open()) {
            sync_cout << "info string OpponentModel: cannot open vocab file "
                      << vocabPath << sync_endl;
            return false;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty()) {
                move_to_idx_map[line] = static_cast<int>(idx_to_move.size());
                idx_to_move.push_back(line);
            }
        }
        if (idx_to_move.empty()) {
            sync_cout << "info string OpponentModel: vocab file is empty" << sync_endl;
            return false;
        }
    }

    // Load ONNX model
    try {
        ort = std::make_unique<OrtContext>();
        ort->session = std::make_unique<Ort::Session>(
            ort->env, modelPath.c_str(), ort->opts);
    } catch (const Ort::Exception& e) {
        sync_cout << "info string OpponentModel ONNX error: " << e.what() << sync_endl;
        ort.reset();
        return false;
    }

    modelLoaded = true;
    sync_cout << "info string OpponentModel loaded ("
              << idx_to_move.size() << " moves) from " << modelPath << sync_endl;
    return true;
}

// ── position_to_input ─────────────────────────────────────────────────────────
// Encodes the position as 12 planes × 8 × 8 = 768 floats.
// Plane order (matching standard AlphaZero-style training):
//   0  White Pawns      6  Black Pawns
//   1  White Knights    7  Black Knights
//   2  White Bishops    8  Black Bishops
//   3  White Rooks      9  Black Rooks
//   4  White Queens    10  Black Queens
//   5  White King      11  Black King
// Within each plane, float[sq] = 1.0 if a piece of that type occupies sq.
// Square indices: 0 = a1, 7 = h1, 56 = a8, 63 = h8 (Stockfish convention).

void OpponentModel::position_to_input(const Position& pos, float* buf) const {
    std::memset(buf, 0, 768 * sizeof(float));

    constexpr PieceType ptypes[6] = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};

    for (int ci = 0; ci < 6; ++ci) {
        Bitboard wb = pos.pieces(WHITE, ptypes[ci]);
        Bitboard bb = pos.pieces(BLACK, ptypes[ci]);
        float*   wp = buf + ci * 64;         // white plane
        float*   bp = buf + (ci + 6) * 64;  // black plane
        while (wb) { wp[pop_lsb(wb)] = 1.0f; }
        while (bb) { bp[pop_lsb(bb)] = 1.0f; }
    }
}

// ── predict_responses ─────────────────────────────────────────────────────────

std::vector<OpponentResponse>
OpponentModel::predict_responses(const Position& pos, int opponentElo, int topN) const {
    std::vector<OpponentResponse> responses;
    if (!modelLoaded || !ort || !ort->session)
        return responses;

    // Build inputs
    std::vector<float> boardBuf(768);
    position_to_input(pos, boardBuf.data());

    // ELO normalised to [0, 1] — matches training convention
    float eloNorm = static_cast<float>(opponentElo) / 3000.0f;
    std::vector<float> eloBuf = {eloNorm};

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // board tensor: (1, 12, 8, 8)
    const std::array<int64_t, 4> boardShape{1, 12, 8, 8};
    auto boardTensor = Ort::Value::CreateTensor<float>(
        memInfo, boardBuf.data(), boardBuf.size(), boardShape.data(), boardShape.size());

    // elo tensor: (1, 1)
    const std::array<int64_t, 2> eloShape{1, 1};
    auto eloTensor = Ort::Value::CreateTensor<float>(
        memInfo, eloBuf.data(), eloBuf.size(), eloShape.data(), eloShape.size());

    // Run inference
    const char* inputNames[]  = {"board", "elo"};
    const char* outputNames[] = {"move_logits"};
    Ort::Value  inputs[]      = {std::move(boardTensor), std::move(eloTensor)};

    std::vector<Ort::Value> outputTensors;
    try {
        outputTensors = ort->session->Run(
            Ort::RunOptions{nullptr}, inputNames, inputs, 2, outputNames, 1);
    } catch (const Ort::Exception& e) {
        sync_cout << "info string OpponentModel inference error: " << e.what() << sync_endl;
        return responses;
    }

    const float* logits = outputTensors[0].GetTensorData<float>();

    output_to_moves(logits, pos, responses);

    // Sort descending by probability and truncate
    std::sort(responses.begin(), responses.end(),
              [](const OpponentResponse& a, const OpponentResponse& b) {
                  return a.probability > b.probability;
              });
    if (static_cast<int>(responses.size()) > topN)
        responses.resize(topN);

    return responses;
}

// ── output_to_moves ───────────────────────────────────────────────────────────
// Maps raw logits → (legal move, probability) pairs via softmax.

void OpponentModel::output_to_moves(const float*                   logits,
                                     const Position&                pos,
                                     std::vector<OpponentResponse>& responses) const {
    // Enumerate legal moves and find their logits
    float maxLogit = -1e9f;

    struct Entry {
        Move  move;
        float logit;
    };
    std::vector<Entry> entries;
    entries.reserve(64);

    for (const auto& m : MoveList<LEGAL>(pos)) {
        std::string uci = move_to_uci(m);
        auto        it  = move_to_idx_map.find(uci);
        if (it == move_to_idx_map.end())
            continue;  // move not in training vocabulary — skip

        float logit = logits[it->second];
        if (logit > maxLogit)
            maxLogit = logit;
        entries.push_back({m, logit});
    }

    if (entries.empty())
        return;

    // Softmax over legal moves only (numerically stable)
    float sumExp = 0.0f;
    for (auto& e : entries) {
        e.logit = std::exp(e.logit - maxLogit);
        sumExp += e.logit;
    }

    responses.reserve(entries.size());
    for (const auto& e : entries)
        responses.push_back({e.move, e.logit / sumExp});
}

// ── get_move_probability ──────────────────────────────────────────────────────

float OpponentModel::get_move_probability(const Position& pos, Move move, int opponentElo) const {
    if (!modelLoaded)
        return 0.0f;
    auto responses = predict_responses(pos, opponentElo, 20);
    for (const auto& r : responses)
        if (r.move == move)
            return r.probability;
    return 0.0f;
}

// ── rank_candidate_moves ──────────────────────────────────────────────────────

std::vector<MoveExploitability>
OpponentModel::rank_candidate_moves(Position& pos, const std::vector<Move>& candidates,
                                     int opponentElo, const EvalFn& evalFn) const {
    std::vector<MoveExploitability> rankings;
    if (!modelLoaded || candidates.empty())
        return rankings;

    for (const Move& move : candidates) {
        MoveExploitability exploit;
        exploit.move = move;

        StateInfo st;
        pos.do_move(move, st);

        exploit.trapPotential = evaluate_trap_potential(pos, Move::none(), opponentElo, evalFn);

        auto responses = predict_responses(pos, opponentElo, 5);

        float expectedOutcome = 0.0f;
        float totalProb       = 0.0f;
        float blunderProb     = 0.0f;

        for (const auto& resp : responses) {
            float outcome = evaluate_after_opponent_response(pos, resp, evalFn);
            expectedOutcome += resp.probability * outcome;
            totalProb += resp.probability;
            if (outcome > 0.5f)
                blunderProb += resp.probability;
        }

        pos.undo_move(move);

        exploit.blunderRate   = blunderProb;
        exploit.difficulty    = 1.0f - (totalProb > 0 ? expectedOutcome / totalProb : 0.5f);
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

// ── is_smart_mistake ─────────────────────────────────────────────────────────

bool OpponentModel::is_smart_mistake(Position& pos, Move /*bestMove*/, Move candidateMove,
                                      int targetElo, int opponentElo, const EvalFn& evalFn) const {
    if (!modelLoaded)
        return false;

    float targetProb = get_move_probability(pos, candidateMove, targetElo);
    if (targetProb < 0.05f)
        return false;

    StateInfo st;
    pos.do_move(candidateMove, st);

    auto responses        = predict_responses(pos, opponentElo, 3);
    bool opponentPunishes = false;

    for (const auto& resp : responses) {
        StateInfo st2;
        pos.do_move(resp.move, st2);
        Value eval = evalFn(pos);
        pos.undo_move(resp.move);

        if (eval < -300 && resp.probability > 0.4f) {
            opponentPunishes = true;
            break;
        }
    }

    pos.undo_move(candidateMove);
    return !opponentPunishes;
}

// ── evaluate_trap_potential ───────────────────────────────────────────────────

float OpponentModel::evaluate_trap_potential(Position& pos, Move ourMove, int opponentElo,
                                              const EvalFn& evalFn) const {
    if (!modelLoaded)
        return 0.0f;

    StateInfo stOur;
    bool      didOurMove = ourMove != Move::none();
    if (didOurMove)
        pos.do_move(ourMove, stOur);

    auto responses = predict_responses(pos, opponentElo, 10);

    float trapScore    = 0.0f;
    int   plausible    = 0;
    int   badResponses = 0;

    for (const auto& resp : responses) {
        if (resp.probability <= 0.05f)
            continue;
        ++plausible;

        StateInfo st2;
        pos.do_move(resp.move, st2);
        Value eval = evalFn(pos);
        pos.undo_move(resp.move);

        if (eval > 100) {
            ++badResponses;
            trapScore += resp.probability;
        }
    }

    if (didOurMove)
        pos.undo_move(ourMove);

    if (plausible > 3 && badResponses >= 2)
        return std::min(1.0f, trapScore * 1.5f);

    return trapScore;
}

// ── evaluate_after_opponent_response ─────────────────────────────────────────

float OpponentModel::evaluate_after_opponent_response(Position&               pos,
                                                       const OpponentResponse& oppResponse,
                                                       const EvalFn&           evalFn) const {
    StateInfo st;
    pos.do_move(oppResponse.move, st);
    float eval = static_cast<float>(evalFn(pos)) / 300.0f;
    pos.undo_move(oppResponse.move);
    return std::tanh(eval);
}

}  // namespace Stockfish
