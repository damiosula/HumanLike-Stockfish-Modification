#include "opponent_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
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

// Flip a UCI move string for the side-to-move perspective when black is
// playing: swap rank digits  ('1'↔'8', '2'↔'7', '3'↔'6', '4'↔'5').
// This maps black's move (e.g. "e7e5") to the LC0 flipped-board coordinates
// ("e2e4"), so a single shared vocabulary covers both colours.
static std::string flip_uci(const std::string& uci) {
    auto flip_rank = [](char c) -> char {
        if (c >= '1' && c <= '8')
            return char('1' + ('8' - c));
        return c;
    };
    std::string flipped = uci;
    if (flipped.size() >= 4) {
        flipped[1] = flip_rank(flipped[1]);  // from-rank
        flipped[3] = flip_rank(flipped[3]);  // to-rank
    }
    return flipped;
}

// Round an arbitrary ELO to the nearest Maia model level (1100–1900, step 100).
static int round_to_maia_elo(int elo) {
    elo = std::max(1100, std::min(1900, elo));
    return ((elo + 50) / 100) * 100;
}

// ── Construction / destruction ────────────────────────────────────────────────

OpponentModel::OpponentModel()  = default;
OpponentModel::~OpponentModel() = default;

// ── load_vocab ────────────────────────────────────────────────────────────────

static bool load_vocab(const std::string& vocabPath,
                       std::vector<std::string>& idx_to_move,
                       std::unordered_map<std::string, int>& move_to_idx_map) {
    idx_to_move.clear();
    move_to_idx_map.clear();

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
    return true;
}

// ── load_model ────────────────────────────────────────────────────────────────

bool OpponentModel::load_model(const std::string& modelPath) {
    modelLoaded = false;
    is_maia_    = false;  // legacy CNN format

    // Derive companion vocab filename: replace ".onnx" with "_vocab.txt"
    std::string vocabPath = modelPath;
    {
        auto pos = vocabPath.rfind(".onnx");
        if (pos != std::string::npos)
            vocabPath.replace(pos, 5, "_vocab.txt");
        else
            vocabPath += "_vocab.txt";
    }

    if (!load_vocab(vocabPath, idx_to_move, move_to_idx_map))
        return false;

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

// ── load_model_for_elo ────────────────────────────────────────────────────────

bool OpponentModel::load_model_for_elo(const std::string& modelDir, int elo) {
    int maia_elo = round_to_maia_elo(elo);

    // Skip reload if the correct model is already loaded
    if (modelLoaded && model_dir_ == modelDir && loaded_elo_ == maia_elo)
        return true;

    // Build model path: {dir}/maia-{elo}.onnx
    std::string modelPath = modelDir;
    if (!modelPath.empty() && modelPath.back() != '/' && modelPath.back() != '\\')
        modelPath += '/';
    modelPath += "maia-" + std::to_string(maia_elo) + ".onnx";

    // The shared vocab lives at {dir}/maia_vocab.txt
    std::string vocabPath = modelDir;
    if (!vocabPath.empty() && vocabPath.back() != '/' && vocabPath.back() != '\\')
        vocabPath += '/';
    vocabPath += "maia_vocab.txt";

    modelLoaded = false;
    if (!load_vocab(vocabPath, idx_to_move, move_to_idx_map))
        return false;

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
    is_maia_    = true;
    model_dir_  = modelDir;
    loaded_elo_ = maia_elo;

    sync_cout << "info string Maia model loaded (ELO " << maia_elo
              << ", " << idx_to_move.size() << " moves) from " << modelPath << sync_endl;
    return true;
}

// ── position_to_input ─────────────────────────────────────────────────────────
// Two encoding modes depending on is_maia_:
//
// LEGACY (12-plane, original CNN):
//   Planes 0–5:  White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
//   Planes 6–11: Black pieces (same order)
//   buf must hold at least 768 floats (12 × 64).
//
// MAIA (112-plane, LC0 format):
//   Planes 0–5:    Side-to-move's pieces
//   Planes 6–11:   Opponent's pieces
//   Plane  12:     Repetition (0.0)
//   Planes 13–103: History (zeros — not tracked)
//   Planes 104–107: Castling rights (our QS, our KS, their QS, their KS)
//   Plane  108:    En-passant file
//   Plane  109:    Rule-50 counter (/ 99)
//   Plane  110:    Zero
//   Plane  111:    Whose turn (1=white, 0=black)
//   When black to move the board is rank-flipped (sq ^ 56).
//   buf must hold at least 7168 floats (112 × 64).

void OpponentModel::position_to_input(const Position& pos, float* buf) const {
    constexpr PieceType ptypes[6] = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};

    if (!is_maia_) {
        // ── Legacy 12-plane encoding ─────────────────────────────────────────
        std::memset(buf, 0, 768 * sizeof(float));
        for (int ci = 0; ci < 6; ++ci) {
            Bitboard wb = pos.pieces(WHITE, ptypes[ci]);
            Bitboard bb = pos.pieces(BLACK, ptypes[ci]);
            float*   wp = buf + ci * 64;
            float*   bp = buf + (ci + 6) * 64;
            while (wb) { wp[pop_lsb(wb)] = 1.0f; }
            while (bb) { bp[pop_lsb(bb)] = 1.0f; }
        }
        return;
    }

    // ── Maia 112-plane encoding ───────────────────────────────────────────────
    std::memset(buf, 0, 112 * 64 * sizeof(float));

    const Color us   = pos.side_to_move();
    const Color them = ~us;
    const bool  flip = (us == BLACK);  // rank-flip when black to move

    auto mapped_sq = [flip](Square sq) -> int {
        return flip ? (static_cast<int>(sq) ^ 56) : static_cast<int>(sq);
    };

    // Planes 0–5: our pieces; planes 6–11: their pieces
    for (int ci = 0; ci < 6; ++ci) {
        float* our_plane   = buf + ci       * 64;
        float* their_plane = buf + (ci + 6) * 64;

        Bitboard our_bb   = pos.pieces(us,   ptypes[ci]);
        Bitboard their_bb = pos.pieces(them, ptypes[ci]);

        while (our_bb)   our_plane[mapped_sq(pop_lsb(our_bb))]     = 1.0f;
        while (their_bb) their_plane[mapped_sq(pop_lsb(their_bb))] = 1.0f;
    }

    // Castling rights (planes 104–107)
    auto fill_plane = [buf](int plane_idx) {
        float* p = buf + plane_idx * 64;
        std::fill(p, p + 64, 1.0f);
    };

    const CastlingRights our_qs  = (us == WHITE) ? WHITE_OOO : BLACK_OOO;
    const CastlingRights our_ks  = (us == WHITE) ? WHITE_OO  : BLACK_OO;
    const CastlingRights them_qs = (us == WHITE) ? BLACK_OOO : WHITE_OOO;
    const CastlingRights them_ks = (us == WHITE) ? BLACK_OO  : WHITE_OO;

    if (pos.can_castle(our_qs))  fill_plane(104);
    if (pos.can_castle(our_ks))  fill_plane(105);
    if (pos.can_castle(them_qs)) fill_plane(106);
    if (pos.can_castle(them_ks)) fill_plane(107);

    // En-passant file (plane 108) — flip only affects rank, not file
    Square ep = pos.ep_square();
    if (ep != SQ_NONE) {
        int    ep_file = file_of(ep);
        float* ep_plane = buf + 108 * 64;
        for (int rank = 0; rank < 8; ++rank)
            ep_plane[rank * 8 + ep_file] = 1.0f;
    }

    // Rule-50 (plane 109), normalised
    {
        float rule50 = static_cast<float>(pos.rule50_count()) / 99.0f;
        float* p = buf + 109 * 64;
        std::fill(p, p + 64, rule50);
    }

    // Plane 111: whose turn
    if (us == WHITE) {
        float* p = buf + 111 * 64;
        std::fill(p, p + 64, 1.0f);
    }
}

// ── predict_responses ─────────────────────────────────────────────────────────

std::vector<OpponentResponse>
OpponentModel::predict_responses(const Position& pos, int opponentElo, int topN) const {
    std::vector<OpponentResponse> responses;
    if (!modelLoaded || !ort || !ort->session)
        return responses;

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> outputTensors;

    if (is_maia_) {
        // ── Maia path: single "board" input (112-plane), "policy" output ─────
        std::vector<float> boardBuf(112 * 64);
        position_to_input(pos, boardBuf.data());

        const std::array<int64_t, 4> boardShape{1, 112, 8, 8};
        auto boardTensor = Ort::Value::CreateTensor<float>(
            memInfo, boardBuf.data(), boardBuf.size(), boardShape.data(), boardShape.size());

        const char* inputNames[]  = {"board"};
        const char* outputNames[] = {"policy"};
        Ort::Value  inputs[]      = {std::move(boardTensor)};

        try {
            outputTensors = ort->session->Run(
                Ort::RunOptions{nullptr}, inputNames, inputs, 1, outputNames, 1);
        } catch (const Ort::Exception& e) {
            sync_cout << "info string OpponentModel inference error: " << e.what() << sync_endl;
            return responses;
        }
    } else {
        // ── Legacy CNN path: "board" (12-plane) + "elo" inputs, "move_logits" output ──
        std::vector<float> boardBuf(768);
        position_to_input(pos, boardBuf.data());

        float eloNorm = static_cast<float>(opponentElo) / 3000.0f;
        std::vector<float> eloBuf = {eloNorm};

        const std::array<int64_t, 4> boardShape{1, 12, 8, 8};
        auto boardTensor = Ort::Value::CreateTensor<float>(
            memInfo, boardBuf.data(), boardBuf.size(), boardShape.data(), boardShape.size());

        const std::array<int64_t, 2> eloShape{1, 1};
        auto eloTensor = Ort::Value::CreateTensor<float>(
            memInfo, eloBuf.data(), eloBuf.size(), eloShape.data(), eloShape.size());

        const char* inputNames[]  = {"board", "elo"};
        const char* outputNames[] = {"move_logits"};
        Ort::Value  inputs[]      = {std::move(boardTensor), std::move(eloTensor)};

        try {
            outputTensors = ort->session->Run(
                Ort::RunOptions{nullptr}, inputNames, inputs, 2, outputNames, 1);
        } catch (const Ort::Exception& e) {
            sync_cout << "info string OpponentModel inference error: " << e.what() << sync_endl;
            return responses;
        }
    }

    const float* logits = outputTensors[0].GetTensorData<float>();
    output_to_moves(logits, pos, responses);

    std::sort(responses.begin(), responses.end(),
              [](const OpponentResponse& a, const OpponentResponse& b) {
                  return a.probability > b.probability;
              });
    if (static_cast<int>(responses.size()) > topN)
        responses.resize(topN);

    return responses;
}

// ── output_to_moves ───────────────────────────────────────────────────────────
// Maps raw policy logits → (legal move, probability) pairs via softmax.
// Because the board was encoded from the side-to-move's perspective,
// we flip the UCI string before vocabulary lookup when black is to move.

void OpponentModel::output_to_moves(const float*                   logits,
                                     const Position&                pos,
                                     std::vector<OpponentResponse>& responses) const {
    // Rank-flip UCI strings only in Maia mode (LC0 board is encoded from side-to-move PoV)
    const bool black_to_move = is_maia_ && (pos.side_to_move() == BLACK);

    float maxLogit = -1e9f;

    struct Entry {
        Move  move;
        float logit;
    };
    std::vector<Entry> entries;
    entries.reserve(64);

    for (const auto& m : MoveList<LEGAL>(pos)) {
        std::string uci = move_to_uci(m);
        // When black is to move, flip ranks so the key matches the vocabulary
        // (which is generated from white's perspective, rank-flipped board).
        if (black_to_move)
            uci = flip_uci(uci);

        auto it = move_to_idx_map.find(uci);
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
                                     int opponentElo, const EvalFn& evalFn,
                                     const DoMoveFn& doMoveFn,
                                     const UndoMoveFn& undoMoveFn) const {
    std::vector<MoveExploitability> rankings;
    if (!modelLoaded || candidates.empty())
        return rankings;

    for (const Move& move : candidates) {
        MoveExploitability exploit;
        exploit.move = move;

        // Baseline eval before our move — accumulator is at root level here, correct.
        float baselineOutcome = std::tanh(static_cast<float>(evalFn(pos)) / 300.0f);

        // Use doMoveFn so the NNUE accumulator stack stays in sync.
        StateInfo st;
        doMoveFn(pos, move, st);

        exploit.trapPotential = evaluate_trap_potential(pos, Move::none(), opponentElo,
                                                        evalFn, doMoveFn, undoMoveFn);

        auto responses = predict_responses(pos, opponentElo, 5);

        sync_cout << "info string   Candidate " << move_to_uci(move)
                  << " | baseline=" << std::fixed << std::setprecision(3) << baselineOutcome
                  << " | trap=" << exploit.trapPotential
                  << " | Maia-" << opponentElo << " top responses:" << sync_endl;

        float expectedOutcome = 0.0f;
        float totalProb       = 0.0f;
        float blunderProb     = 0.0f;

        for (const auto& resp : responses) {
            float outcome = evaluate_after_opponent_response(pos, resp, evalFn,
                                                             doMoveFn, undoMoveFn);
            sync_cout << "info string     -> " << move_to_uci(resp.move)
                      << " prob=" << std::setprecision(3) << resp.probability
                      << " outcome=" << outcome
                      << (outcome > baselineOutcome + 0.1f ? " [BLUNDER]" : "")
                      << sync_endl;
            expectedOutcome += resp.probability * outcome;
            totalProb += resp.probability;
            // Blunder: opponent's response is WORSE for them than the baseline
            // (i.e., we gained MORE advantage than we had before the move)
            if (outcome > baselineOutcome + 0.1f)
                blunderProb += resp.probability;
        }

        undoMoveFn(pos, move);

        // difficulty: how hard the resulting position is FOR THE OPPONENT.
        // High = good for us. meanOutcome in [-1,1] (tanh range); map to [0,1].
        float meanOutcome     = (totalProb > 0 ? expectedOutcome / totalProb : 0.0f);
        exploit.blunderRate   = blunderProb;
        exploit.difficulty    = std::clamp(0.5f + 0.5f * meanOutcome, 0.0f, 1.0f);
        exploit.expectedValue = w_trap_       * exploit.trapPotential
                              + w_blunder_    * exploit.blunderRate
                              + w_difficulty_ * exploit.difficulty;

        sync_cout << "info string   => blunder=" << std::setprecision(3) << exploit.blunderRate
                  << " difficulty=" << exploit.difficulty
                  << " expectedValue=" << exploit.expectedValue << sync_endl;

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
                                      int targetElo, int opponentElo, const EvalFn& evalFn,
                                      const DoMoveFn& doMoveFn,
                                      const UndoMoveFn& undoMoveFn) const {
    if (!modelLoaded)
        return false;

    float targetProb = get_move_probability(pos, candidateMove, targetElo);
    if (targetProb < 0.05f)
        return false;

    StateInfo st;
    doMoveFn(pos, candidateMove, st);

    auto  responses          = predict_responses(pos, opponentElo, 5);
    float punishingProbTotal = 0.0f;

    for (const auto& resp : responses) {
        StateInfo st2;
        doMoveFn(pos, resp.move, st2);
        Value eval = evalFn(pos);
        undoMoveFn(pos, resp.move);

        // Accumulate probability of any response that decisively punishes us
        if (eval < -300)
            punishingProbTotal += resp.probability;
    }

    // Punished if the cumulative probability of refutations exceeds 10%
    // Strict threshold: over many moves even 15% punish rate costs games
    bool opponentPunishes = (punishingProbTotal > 0.10f);

    undoMoveFn(pos, candidateMove);
    return !opponentPunishes;
}

// ── evaluate_trap_potential ───────────────────────────────────────────────────

float OpponentModel::evaluate_trap_potential(Position& pos, Move ourMove, int opponentElo,
                                              const EvalFn& evalFn, const DoMoveFn& doMoveFn,
                                              const UndoMoveFn& undoMoveFn) const {
    if (!modelLoaded)
        return 0.0f;

    StateInfo stOur;
    bool      didOurMove = ourMove != Move::none();
    if (didOurMove)
        doMoveFn(pos, ourMove, stOur);

    auto responses = predict_responses(pos, opponentElo, 10);

    float trapScore    = 0.0f;
    int   plausible    = 0;
    int   badResponses = 0;

    for (const auto& resp : responses) {
        if (resp.probability <= 0.05f)
            continue;
        ++plausible;

        StateInfo st2;
        doMoveFn(pos, resp.move, st2);
        Value eval = evalFn(pos);
        undoMoveFn(pos, resp.move);

        if (eval > 100) {
            ++badResponses;
            trapScore += resp.probability;
        }
    }

    if (didOurMove)
        undoMoveFn(pos, ourMove);

    if (plausible > 3 && badResponses >= 2)
        return std::min(1.0f, trapScore * 1.5f);

    return trapScore;
}

// ── evaluate_after_opponent_response ─────────────────────────────────────────

float OpponentModel::evaluate_after_opponent_response(Position&               pos,
                                                       const OpponentResponse& oppResponse,
                                                       const EvalFn&           evalFn,
                                                       const DoMoveFn&         doMoveFn,
                                                       const UndoMoveFn&       undoMoveFn) const {
    StateInfo st;
    doMoveFn(pos, oppResponse.move, st);
    float eval = static_cast<float>(evalFn(pos)) / 300.0f;
    undoMoveFn(pos, oppResponse.move);
    return std::tanh(eval);
}

}  // namespace Stockfish
