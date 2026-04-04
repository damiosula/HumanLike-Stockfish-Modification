#include "opponent_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <onnxruntime_cxx_api.h>

#include "misc.h"
#include "movegen.h"

namespace Stockfish {

OpponentModel::OpponentModel() = default;
OpponentModel::~OpponentModel() = default;

struct OpponentModel::OrtContext {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "OpponentModel"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;

    OrtContext() { opts.SetIntraOpNumThreads(1); }
};

std::string OpponentModel::move_to_uci(Move m) {
    auto sq_str = [](Square sq) -> std::string {
        std::string s;
        s += char('a' + file_of(sq));
        s += char('1' + rank_of(sq));
        return s;
    };

    std::string result = sq_str(m.from_sq()) + sq_str(m.to_sq());

    if (m.type_of() == PROMOTION) {
        static const std::unordered_map<PieceType, char> promoChar = {
            {KNIGHT, 'n'}, {BISHOP, 'b'}, {ROOK, 'r'}, {QUEEN, 'q'}
        };
        result += promoChar.at(m.promotion_type());
    }

    return result;
}

static std::string flip_uci(const std::string& uci) {
    auto flip_rank = [](char c) -> char {
        if (c >= '1' && c <= '8')
            return char('1' + ('8' - c));
        return c;
    };

    std::string flipped = uci;

    if (flipped.size() >= 4) {
        flipped[1] = flip_rank(flipped[1]);
        flipped[3] = flip_rank(flipped[3]);
    }

    return flipped;
}

static bool load_vocab(const std::string& vocabPath, std::vector<std::string>& idx_to_move, std::unordered_map<std::string, int>& move_to_idx_map) {
    idx_to_move.clear();
    move_to_idx_map.clear();

    std::ifstream f(vocabPath);

    if (!f.is_open()) {
        sync_cout << "Error opening vocab file " << vocabPath << sync_endl;
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
        sync_cout << "Error: vocab file is empty" << sync_endl;
        return false;
    }

    return true;
}

bool OpponentModel::load_model(const std::string& modelPath) {
    modelLoaded = false;

    try {
        ort = std::make_unique<OrtContext>();
        ort->session = std::make_unique<Ort::Session>(ort->env, modelPath.c_str(), ort->opts);
    } catch (const Ort::Exception& e) {
        sync_cout << "OpponentModel loading ONNX error: " << e.what() << sync_endl;
        ort.reset();
        return false;
    }

    std::string vocabPath = modelPath;
    auto ext = vocabPath.rfind(".onnx");

    if (ext != std::string::npos)
        vocabPath.replace(ext, 5, "_vocab.txt");
    else
        vocabPath += "_vocab.txt";

    if (!load_vocab(vocabPath, idx_to_move, move_to_idx_map))
        return false;

    modelLoaded = true;

    sync_cout << "OpponentModel vocab loaded (" << idx_to_move.size() << " moves) from " << modelPath << sync_endl;
    return true;
}

void OpponentModel::position_to_input(const Position& pos, float* buf) const {
    constexpr PieceType ptypes[6] = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};

    std::memset(buf, 0, 112 * 64 * sizeof(float));

    const Color stockfishColour = pos.side_to_move();
    const Color opponentColour = ~stockfishColour;
    const bool flip = (stockfishColour == BLACK); 

    auto mapped_sq = [flip](Square sq) -> int {
        return flip ? (static_cast<int>(sq) ^ 56) : static_cast<int>(sq);
    };

    for (int ci = 0; ci < 6; ++ci) {
        float* stockfish_plane = buf + ci * 64;
        float* opponent_plane = buf + (ci + 6) * 64;

        Bitboard stockfish_bb = pos.pieces(stockfishColour, ptypes[ci]);
        Bitboard opponent_bb = pos.pieces(opponentColour, ptypes[ci]);

        while (stockfish_bb) stockfish_plane[mapped_sq(pop_lsb(stockfish_bb))] = 1.0f;
        while (opponent_bb) opponent_plane[mapped_sq(pop_lsb(opponent_bb))] = 1.0f;
    }

    auto fill_plane = [buf](int plane_idx) {
        float* p = buf + plane_idx * 64;
        std::fill(p, p + 64, 1.0f);
    };

    const CastlingRights stockfish_qs = (stockfishColour == WHITE) ? WHITE_OOO : BLACK_OOO;
    const CastlingRights stockfish_ks = (stockfishColour == WHITE) ? WHITE_OO : BLACK_OO;
    const CastlingRights opponent_qs = (stockfishColour == WHITE) ? BLACK_OOO : WHITE_OOO;
    const CastlingRights opponent_ks = (stockfishColour == WHITE) ? BLACK_OO : WHITE_OO;

    if (pos.can_castle(stockfish_qs)) fill_plane(104);
    if (pos.can_castle(stockfish_ks)) fill_plane(105);
    if (pos.can_castle(opponent_qs)) fill_plane(106);
    if (pos.can_castle(opponent_ks)) fill_plane(107);

    Square ep = pos.ep_square();
    if (ep != SQ_NONE) {
        int ep_file = file_of(ep);
        float* ep_plane = buf + 108 * 64;
        for (int rank = 0; rank < 8; ++rank)
            ep_plane[rank * 8 + ep_file] = 1.0f;
    }

    {
        float rule50 = static_cast<float>(pos.rule50_count()) / 99.0f;
        float* p = buf + 109 * 64;
        std::fill(p, p + 64, rule50);
    }

    if (stockfishColour == WHITE) {
        float* p = buf + 111 * 64;
        std::fill(p, p + 64, 1.0f);
    }
}

std::vector<OpponentResponse>OpponentModel::predict_responses(const Position& pos, int opponentElo, int topN) const {
    std::vector<OpponentResponse> responses;
    if (!modelLoaded || !ort || !ort->session)
        return responses;

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> outputTensors;

    std::vector<float> boardBuf(112 * 64);
    position_to_input(pos, boardBuf.data());

    const std::array<int64_t, 4> boardShape{1, 112, 8, 8};
    auto boardTensor = Ort::Value::CreateTensor<float>(
        memInfo, boardBuf.data(), boardBuf.size(), boardShape.data(), boardShape.size());

    int64_t elo = static_cast<int64_t>(opponentElo);
    const std::array<int64_t, 0> eloShape{};
    auto eloTensor = Ort::Value::CreateTensor<int64_t>(
        memInfo, &elo, 1, eloShape.data(), 0);

    const char* inputNames[] = {"board", "elo"};
    const char* outputNames[] = {"policy"};
    Ort::Value inputs[] = {std::move(boardTensor), std::move(eloTensor)};

    try {
        outputTensors = ort->session->Run(
            Ort::RunOptions{nullptr}, inputNames, inputs, 2, outputNames, 1);
    } catch (const Ort::Exception& e) {
        sync_cout << "WARNING: OpponentElo " << opponentElo << " is out of the model's range. Not using opponent model" << sync_endl;
        return responses;
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

void OpponentModel::output_to_moves(const float* logits, const Position& pos, std::vector<OpponentResponse>& responses) const {
    const bool black_to_move = (pos.side_to_move() == BLACK);

    float maxLogit = -1e9f;

    struct Entry {
        Move move;
        float logit;
    };
    std::vector<Entry> entries;
    entries.reserve(64);

    for (const auto& m : MoveList<LEGAL>(pos)) {
        std::string uci = move_to_uci(m);

        if (black_to_move)
            uci = flip_uci(uci);

        auto it = move_to_idx_map.find(uci);
        if (it == move_to_idx_map.end())
            continue; 

        float logit = logits[it->second];
        if (logit > maxLogit)
            maxLogit = logit;

        entries.push_back({m, logit});
    }

    if (entries.empty())
        return;

    float sumExp = 0.0f;
    for (auto& e : entries) {
        e.logit = std::exp(e.logit - maxLogit);
        sumExp += e.logit;
    }

    responses.reserve(entries.size());
    for (const auto& e : entries)
        responses.push_back({e.move, e.logit / sumExp});
}

float OpponentModel::get_move_probability(const Position& pos, Move move, int opponentElo) const {
    if (!modelLoaded)
        return 0.0f;

    auto responses = predict_responses(pos, opponentElo, 20);

    for (const auto& r : responses)
        if (r.move == move)
            return r.probability;

    return 0.0f;
}


std::vector<MoveExploitability>OpponentModel::rank_candidate_moves(Position& pos, const std::vector<Move>& candidates, int opponentElo, 
                                                                    const EvalFn& evalFn, const DoMoveFn& doMoveFn, const UndoMoveFn& undoMoveFn) const {
    std::vector<MoveExploitability> rankings;
    if (!modelLoaded || candidates.empty())
        return rankings;

    for (const Move& move : candidates) {
        MoveExploitability exploit;
        exploit.move = move;

        StateInfo st;
        doMoveFn(pos, move, st);

        Value evalAfterCandidate = evalFn(pos);

        auto responses = predict_responses(pos, opponentElo, 5);

        sync_cout << "Candidate move: " << move_to_uci(move)
                  << " Eval from opponents PoV = " << evalAfterCandidate
                  << "\nOpponent's top responses:" << sync_endl;

        struct RespResult { Move move; float probability; float rawSum; };
        std::vector<RespResult> respResults;
        float minRawSum = std::numeric_limits<float>::max();

        for (const auto& resp : responses) {
            float rawSum = evaluate_after_opponent_response(pos, resp, evalFn,
                                                            doMoveFn, undoMoveFn,
                                                            evalAfterCandidate);
            respResults.push_back({resp.move, resp.probability, rawSum});
            minRawSum = std::min(minRawSum, rawSum);
        }

        float norm = std::max(std::abs(static_cast<float>(evalAfterCandidate)), 300.0f);

        float weightedRawSum = 0.0f;
        float totalProb = 0.0f;

        for (const auto& rr : respResults) {
            float normalised = std::tanh(rr.rawSum / norm);
            sync_cout << move_to_uci(rr.move)
                      << " prob of move = " << std::setprecision(3) << rr.probability
                      << " eval outcome = " << normalised << sync_endl;
            weightedRawSum += rr.probability * rr.rawSum;
            totalProb += rr.probability;
        }

        undoMoveFn(pos, move);

        float meanRawSum = (totalProb > 0 ? weightedRawSum / totalProb : 0.0f);
        exploit.avgResponseEvalGain = std::clamp(0.5f + 0.5f * std::tanh(meanRawSum / norm), 0.0f, 1.0f);
        exploit.bestResponseEvalGain = std::clamp(0.5f + 0.5f * std::tanh(minRawSum / norm), 0.0f, 1.0f);
        exploit.expectedValue = avg_response_eval_gain_weight  * exploit.avgResponseEvalGain
                                 + best_response_eval_gain_weight * exploit.bestResponseEvalGain;

        sync_cout << "avg response eval gain = " << std::setprecision(3) << exploit.avgResponseEvalGain
                  << " best response eval gain = " << exploit.bestResponseEvalGain
                  << " expected value = " << exploit.expectedValue << sync_endl;

        rankings.push_back(exploit);
    }

    std::sort(rankings.begin(), rankings.end(),
              [](const MoveExploitability& a, const MoveExploitability& b) {
                  return a.expectedValue > b.expectedValue;
              });

    return rankings;
}

float OpponentModel::evaluate_after_opponent_response(Position& pos, const OpponentResponse& oppResponse, const EvalFn& evalFn,
                                                       const DoMoveFn& doMoveFn, const UndoMoveFn& undoMoveFn,
                                                       Value evalAfterCandidate) const {
    StateInfo st;
    doMoveFn(pos, oppResponse.move, st);
    float rawSum = static_cast<float>(evalFn(pos)) + static_cast<float>(evalAfterCandidate);
    undoMoveFn(pos, oppResponse.move);
    return rawSum;
}

} // namespace Stockfish
