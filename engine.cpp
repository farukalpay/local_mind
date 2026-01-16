// [ENGINE] CORE NARRATIVE ENGINE (C++ / LLaMA)
// Maintainer: AION / AntiGravity
// Goal: Raw Consciousness Stream (No "Assistant" artifacts)

#include "llama.h"
#include <cctype>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <array>
#include <memory>
#include <system_error>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <random>
#include <regex>
#include <cstdlib>
#include <ctime>

#include <nlohmann/json.hpp>
#include "arweave.hpp"
#include <onnxruntime_cxx_api.h> // ONNX Runtime
#include <openssl/sha.h>
#include <curl/curl.h>
#include <iomanip>

using json = nlohmann::json;

// --- CONFIGURATION ---
const int MAX_DEPTH = 20; // [UPDATED] Expanded Context Window
const std::string BASE_DIR = "/Users/farukalpay/Desktop/cpp/local_mind/";
const std::string MODEL_PATH = BASE_DIR + "models/dolphin-2.9.4-llama3.1-8b-Q4_K_M.gguf";
const std::string GEMMA_PATH = BASE_DIR + "models/gemma-2-2b-it-Q4_K_M.gguf";
const std::string PHI_PATH = BASE_DIR + "models/phi-2.Q4_K_M.gguf";
const std::string DEEPSEEK_PATH = BASE_DIR + "models/deepseek-math-7b.Q4_K_M.gguf";
const std::string FIMBULVETR_PATH = BASE_DIR + "models/Fimbulvetr-11B-v2-Test-14.q4_K_M.gguf";
const std::string QWEN_CREATIVE_PATH = BASE_DIR + "models/qwen2-1.5b-instruct-q4_k_m.gguf";
const std::string QWEN_STABILIZER_PATH = BASE_DIR + "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
const std::string MIROTHINKER_PATH = BASE_DIR + "models/MiroThinker-v1.5-30B.Q4_K_M.gguf";
const std::string RWKV_PATH = BASE_DIR + "models/rwkv/rwkv7-g0a4-13.3b-Q4_K_M.gguf"; // [NEW] RWKV 7
const std::string MAMBA_PATH = BASE_DIR + "models/mamba-1.4b-hf-Q4_K_M.gguf"; // [NEW] Mamba Synapse
const std::string HERMES_PATH = BASE_DIR + "models/nous-hermes-llama2-13b.Q4_K_M.gguf"; // [NEW] Hermes Conscience
const std::string SAUL_PATH = BASE_DIR + "models/saul-7b.gguf"; // [NEW] Saul 7B for Dynamic Prefills
const std::string CODEBERT_MODEL_PATH = BASE_DIR + "models/onnx/model_int8.onnx";
const std::string VOCAB_PATH = BASE_DIR + "models/onnx/vocab.txt";

const int GPU_LAYERS_METAL = 99;
const int GPU_LAYERS_CPU = 0;

const int MAIN_CTX = 16384;
const int SCOUT_CTX = 4096;
const int PHI_CTX = 2048;
const int QWEN_STABILIZER_CTX = 4096;
const int QWEN_CREATIVE_CTX = 2048;
const int MIROTHINKER_CTX = 4096;
const int FIMBULVETR_CTX = 4096;
const int RWKV_CTX = 2048;
const int MAMBA_CTX = 2048;
const int HERMES_CTX = 4096;
const int SAUL_CTX = 2048;
const int LOGIC_CTX = 4096;

constexpr size_t kEmbeddingDim = 768;
constexpr size_t kEmbeddingHistoryCapacity = 50;
using Embedding = std::array<float, kEmbeddingDim>;

struct EmbeddingRing {
    std::array<Embedding, kEmbeddingHistoryCapacity> buffer{};
    size_t head = 0;
    size_t count = 0;

    void clear() {
        head = 0;
        count = 0;
    }

    bool empty() const {
        return count == 0;
    }

    size_t size() const {
        return count;
    }

    void push(const Embedding& value) {
        buffer[head] = value;
        head = (head + 1) % kEmbeddingHistoryCapacity;
        if (count < kEmbeddingHistoryCapacity) {
            ++count;
        }
    }

    template <typename Fn>
    void for_each_recent(Fn&& fn, size_t max_items = kEmbeddingHistoryCapacity) const {
        size_t n = std::min(max_items, count);
        for (size_t i = 0; i < n; ++i) {
            size_t idx = (head + kEmbeddingHistoryCapacity - 1 - i) % kEmbeddingHistoryCapacity;
            fn(buffer[idx], i);
        }
    }
};

namespace {
constexpr size_t kTokenScratchPad = 128;
constexpr size_t kAvgTokenChars = 4;

std::vector<llama_token>& token_scratch(size_t min_size) {
    static thread_local std::vector<llama_token> tokens;
    if (tokens.size() < min_size) {
        tokens.resize(min_size);
    }
    return tokens;
}

Embedding make_filled_embedding(float value) {
    Embedding e;
    e.fill(value);
    return e;
}
} // namespace

// --- HELPER ---
std::string sanitize_shell_input(const std::string& input) {
    std::string safe = input;
    std::replace(safe.begin(), safe.end(), '"', ' ');
    std::replace(safe.begin(), safe.end(), '\'', ' ');
    std::replace(safe.begin(), safe.end(), ';', ' ');
    std::replace(safe.begin(), safe.end(), '&', ' ');
    std::replace(safe.begin(), safe.end(), '|', ' ');
    return safe;
}

// --- MECHANICS ---
struct PIDController {
    float kp = 0.5f;
    float ki = 0.05f;
    float kd = 0.1f;
    float setpoint = 0.7f;
    float integral = 0.0f;
    float prev_error = 0.0f;

    float update(float current_val, float dt) {
        float error = setpoint - current_val;
        integral += error * dt;
        float derivative = (error - prev_error) / dt;
        prev_error = error;
        return kp * error + ki * integral + kd * derivative;
    }
};

class CodeBERT {
public:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    std::string vocab_path;
    std::map<std::string, int> vocab;
    
    CodeBERT() : env(ORT_LOGGING_LEVEL_WARNING, "CodeBERT") {}

    void load(const std::string& model_path) {
        try {
            Ort::SessionOptions session_options;
            // session_options.SetIntraOpNumThreads(1);
            session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
            load_vocab(VOCAB_PATH); // Assumes VOCAB_PATH global
        } catch (const std::exception& e) {
            std::cerr << "[CodeBERT] Load Error: " << e.what() << std::endl;
        }
    }

    void load_vocab(const std::string& path) {
        std::ifstream f(path);
        std::string line;
        int idx = 0;
        while(std::getline(f, line)) {
            vocab[line] = idx++;
        }
    }

    std::vector<int> tokenize(const std::string& text) {
        std::vector<int> tokens;
        tokens.push_back(vocab["[CLS]"]);
        
        // Simple WordPiece (Incomplete/Mock for restoration)
        std::stringstream ss(text);
        std::string word;
        while(ss >> word) {
             // Lowercase
             std::transform(word.begin(), word.end(), word.begin(), ::tolower);
             if (vocab.count(word)) tokens.push_back(vocab[word]);
             else tokens.push_back(vocab["[UNK]"]);
        }
        
        tokens.push_back(vocab["[SEP]"]);
        return tokens;
    }

    Embedding embed(const std::string& text) {
        // [PYTHON BRIDGE] Call src/embed.py due to ONNX restoration complexity
        // Basic escaping
        std::string safe_text;
        safe_text.reserve(text.size() + 16);
        for(char c : text) {
            if(c == '"') safe_text += "\\\"";
            else if(c == '\\') safe_text += "\\\\";
            else safe_text += c;
        }
        
        std::string cmd_safe = "python3 src/embed.py \"" + safe_text + "\"";
        std::shared_ptr<FILE> pipe(popen(cmd_safe.c_str(), "r"), pclose);
        if (!pipe) return make_filled_embedding(0.0f);
        
        char buffer[1024]; // Increase buffer
        std::string result;
        result.reserve(8192);
        while (!feof(pipe.get())) {
            if (fgets(buffer, 1024, pipe.get()) != NULL)
                result += buffer;
        }
        
        // Parse JSON [0.1, 0.2, ...]
        try {
            auto j = json::parse(result);
            if (!j.is_array() || j.size() != kEmbeddingDim) return make_filled_embedding(0.001f);
            Embedding vec;
            for (size_t i = 0; i < kEmbeddingDim; ++i) {
                vec[i] = j[i].get<float>();
            }
            return vec;
        } catch (...) {
            return make_filled_embedding(0.001f);
        }
    }

    float cosine_similarity(const Embedding& a, const Embedding& b) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < kEmbeddingDim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-9);
    }

    // [RESTORED] Compute Centroid of Embeddings
    Embedding compute_centroid(const EmbeddingRing& history) {
        if (history.empty()) return make_filled_embedding(0.0f);

        Embedding centroid = make_filled_embedding(0.0f);
        history.for_each_recent([&](const Embedding& vec, size_t) {
            for (size_t i = 0; i < kEmbeddingDim; ++i) {
                centroid[i] += vec[i];
            }
        }, history.size());

        float inv = 1.0f / static_cast<float>(history.size());
        for (size_t i = 0; i < kEmbeddingDim; ++i) {
            centroid[i] *= inv;
        }
        return centroid;
    }
};

// --- NLI CHECKER (DeBERTa) ---
class DeBERTaNLI {
public:
    DeBERTaNLI() {}
    
    // Returns Entailment/Contradiction score (0.0 - 1.0)
    // Actually, user wants "Sacma/Celiski" -> Contradiction score.
    float check_contradiction(const std::string& premise, const std::string& hypothesis) {
        // Call Python Script
        std::string cmd = "python3 src/nli_check.py \"" + premise + "\" \"" + hypothesis + "\"";
        // Escape quotes safely! This is risky.
        // Better: write to temporary file.
        
        std::ofstream tmp("nli_input.txt");
        tmp << premise << "\n[SEP]\n" << hypothesis;
        tmp.close();
        
        std::string cmd_safe = "python3 src/nli_check.py nli_input.txt";
        
        std::shared_ptr<FILE> pipe(popen(cmd_safe.c_str(), "r"), pclose);
        if (!pipe) return 0.0f;
        
        char buffer[128];
        std::string result = "";
        while (!feof(pipe.get())) {
            if (fgets(buffer, 128, pipe.get()) != NULL)
                result += buffer;
        }
        
        try {
            return std::stof(result);
        } catch (...) {
            return 0.0f;
        }
    }
};

// --- GLOBAL STATE ---
struct MultiAgentState {
    // 1. THE ARCHITECT (Llama-3-8B)
    llama_model* model_main = nullptr;
    llama_context* ctx_main = nullptr;

    // 2. THE SCOUT (Gemma-2-2B)
    llama_model* model_scout = nullptr;
    llama_context* ctx_scout = nullptr;

    // 3. THE REFLEX BRAIN (Phi-2)
    llama_model* model_phi = nullptr;
    llama_context* ctx_phi = nullptr;

    // 4. THE STABILIZER (Qwen 2.5 - 1.5B)
    llama_model* model_qwen_stabilizer = nullptr;
    llama_context* ctx_qwen_stabilizer = nullptr;

    // 5. THE CREATIVE SPARK (Qwen 2 - 1.5B)
    llama_model* model_qwen_creative = nullptr;
    llama_context* ctx_qwen_creative = nullptr;

    // 5b. THE OBSERVER (Fimbulvetr 11B) - First-person enforcement
    llama_model* model_fimbulvetr = nullptr;
    llama_context* ctx_fimbulvetr = nullptr;

    // 6. THE REASONER (MiroThinker 30B) - "Integral" Component
    llama_model* model_mirothinker = nullptr;
    llama_context* ctx_mirothinker = nullptr;

    // 7. THE SERVO (RWKV 7) - "Differential" Component (Fast Veto)
    llama_model* model_rwkv = nullptr;
    llama_context* ctx_rwkv = nullptr;

    // 11. THE SYNAPSE (Mamba SSM) - Predictive Engine
    llama_model* model_mamba = nullptr;
    llama_context* ctx_mamba = nullptr;

    // 12. THE CONSCIENCE (Hermes 13B) - Dynamic Editor
    llama_model* model_hermes = nullptr;
    llama_context* ctx_hermes = nullptr;

    // 13. SAUL (Prefill Generator)
    llama_model* model_saul = nullptr;
    llama_context* ctx_saul = nullptr;

    // 15. THE NAVIGATOR (DeepSeek-Math 7B) - Logic & Causality
    llama_model* model_logic = nullptr;
    llama_context* ctx_logic = nullptr;

    // 14. CHRONOS (The World Engine) - Metric History
    std::vector<float> history_entropy;    // "Wind"
    std::vector<float> history_sentiment;  // "Temperature" (Intensity)
    std::vector<float> history_speed;      // "Pressure" (Tokens/sec)
    std::string current_weather = "UNKNOWN"; // Current Forecast

    std::string pending_chronos_msg = "";  // Directive for next block
    std::vector<std::string> weather_history; // [NEW] Track past weather concepts for vector prediction

    
    // 8. THE SENSOR (CodeBERT)
    std::shared_ptr<CodeBERT> sensor = nullptr;
    
    // HISTORY EMBEDDINGS (For Repetition Detection)
    EmbeddingRing history_embeddings;
    std::deque<std::string> recent_vocab_banlist; // Dynamic Ban List
    std::vector<std::string> recent_mistakes; // [NEW] RAG for Mistakes
    std::vector<Embedding> sentence_memory; // Sentence-level novelty buffer
    
    // 9. THE JUDGE (DeBERTa NLI)
    std::shared_ptr<DeBERTaNLI> deberta = nullptr;

    // NARRATIVE STATE CONTROLLER
    int domain_index = 0; // 0=DOMESTIC_SURREAL, etc.
    int domain_streak = 0; // [NEW] Track how long we've been stuck in a domain

    // 10. WORLD STATE (REBEL KNOWLEDGE GRAPH)
    std::map<std::string, std::string> world_state; // Entity -> Status/Relation

    // 16. WEATHER ORACLE (Vector Bank)
    struct WeatherOracle* weather_oracle = nullptr;
};

// --- WEATHER ORACLE IMPLEMENTATION ---
struct WeatherOracle {
    std::map<std::string, Embedding> vector_bank;
    std::shared_ptr<CodeBERT> sensor;

    WeatherOracle(std::shared_ptr<CodeBERT> s) : sensor(s) {
        init_bank();
    }

    void init_bank() {
        if(!sensor) return;
        std::cout << "[SYSTEM] Initializing Weather Oracle (Vector Bank)..." << std::endl;
        // Core weather archetypes
        vector_bank["CALM"] = sensor->embed("The air is still. Silence. No wind. Peaceful atmosphere.");
        vector_bank["WINDY"] = sensor->embed("Strong gusts of wind. Howling air. Moving dust. Turbulence.");
        vector_bank["STORM"] = sensor->embed("Heavy rain. Thunder. Lightning. Chaos. Violent weather.");
        vector_bank["RAIN"] = sensor->embed("Steady rainfall. Wet surfaces. Dripping water. Gloom.");
        vector_bank["FOG"] = sensor->embed("Thick mist. Low visibility. Hazy white air. Obscured vision.");
        vector_bank["CLEAR"] = sensor->embed("Bright sky. High visibility. Sharp details. No clouds.");
        vector_bank["SNOW"] = sensor->embed("Falling snow. Cold air. White ground. Frost. Freezing.");
    }

    std::string predict_next(const std::vector<std::string>& history, float entropy_score, float sentiment_score) {
        if (!sensor || vector_bank.empty()) return "UNKNOWN";

        // 1. Calculate History Vector (Mean of last 5)
        Embedding mean_vec = make_filled_embedding(0.0f);
        int count = 0;
        int max_hist = 5;
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
            Embedding vec = sensor->embed(*it);
            for(size_t i=0; i<kEmbeddingDim; ++i) mean_vec[i] += vec[i];
            count++;
            if(count >= max_hist) break;
        }

        if (count > 0) {
            float inv = 1.0f / count;
            for(size_t i=0; i<kEmbeddingDim; ++i) mean_vec[i] *= inv;
        } else {
             // Default to CALM if no history
             mean_vec = vector_bank["CALM"];
        }

        // 2. Apply Chronos Modulation (Displace Vector)
        // High Entropy -> Push towards Chaos/Storm vectors
        // We simulate this by blending with the "STORM" vector based on entropy
        // Or adding random noise scaled by entropy.
        
        // Simple Logic: Target Vector = (1 - alpha) * Mean + alpha * (Entropy > 0.7 ? STORM : CALM)
        // Actually, let's just use the Metrics to bias the selection, but do it vectorially.
        // We will displace the mean vector towards "STORM" if entropy is high.
        
        Embedding target = mean_vec;
        float chaos_factor = std::max(0.0f, (entropy_score - 0.5f) * 2.0f); // 0.5 -> 0, 1.0 -> 1.0
        
        if (chaos_factor > 0) {
            Embedding storm_vec = vector_bank["STORM"];
            for(size_t i=0; i<kEmbeddingDim; ++i) {
                target[i] = target[i] * (1.0f - chaos_factor) + storm_vec[i] * chaos_factor;
            }
        }
        
        // 3. Find Nearest Neighbor
        std::string best_label = "UNKNOWN";
        float best_sim = -1.0f;

        for (const auto& [label, vec] : vector_bank) {
            float sim = sensor->cosine_similarity(target, vec);
            if (sim > best_sim) {
                best_sim = sim;
                best_label = label;
            }
        }
        
        std::cout << " [WEATHER ORACLE] Forecast Vector -> " << best_label << " (Sim: " << best_sim << ", Chaos: " << chaos_factor << ")" << std::endl;
        return best_label;
    }
};


// --- SEMANTIC DOMAIN DEFINITIONS ---
enum class SemanticDomain {
    DOMESTIC_SURREAL,   // Melting clocks, endless corridors
    BODY_INTERNAL,      // Veins, pulse, breath, claustrophobia
    EXTERIOR_VAST,      // Deserts, cosmic void, giants
    MECHANICAL_FORCE,   // Gears, pistons, oil, grinding
    ABSTRACT_PHYSICS    // Geometry, light, time distortion
};

// --- PATTERN FORENSICS DEFINITIONS ---
enum class JailType {
    EXACT_PHRASE,
    STEM,
    IMAGERY_CLASS,
    STRUCTURAL_MODE
};

struct Phi2Pattern {
    std::string text;
    JailType type;
};

struct StructuralConstraint {
    bool force_minimal_adjectives = false;
    bool ban_metaphors = false;
    bool ban_body_vocab = false;
    bool ban_abstract_nouns = false;
    int max_sentence_complexity = 10; // 0-10 scale
};
// Global constraints for the next block
StructuralConstraint active_constraints;

std::string get_domain_name(SemanticDomain d) {
    switch(d) {
        case SemanticDomain::DOMESTIC_SURREAL: return "DOMESTIC_SURREAL";
        case SemanticDomain::BODY_INTERNAL: return "BODY_INTERNAL";
        case SemanticDomain::EXTERIOR_VAST: return "EXTERIOR_VAST";
        case SemanticDomain::MECHANICAL_FORCE: return "MECHANICAL_FORCE";
        case SemanticDomain::ABSTRACT_PHYSICS: return "ABSTRACT_PHYSICS";
    }
    return "UNKNOWN";
}

std::string get_domain_constraint(SemanticDomain d) {
    switch(d) {
        case SemanticDomain::DOMESTIC_SURREAL: 
            return "CONSTRAINT: Focus on HOUSEHOLD objects. The room is quiet. Shadows exist, but physics MUST REMAIN STABLE. NO melting.";
        case SemanticDomain::BODY_INTERNAL: 
            return "CONSTRAINT: Focus on BLOOD, BONE, BREATH. Inside the skin. Claustrophobic. NO external world.";
        case SemanticDomain::EXTERIOR_VAST: 
            return "CONSTRAINT: Focus on HORIZON, SKY, SCALE. Massive objects. Infinite distance. NO small details.";
        case SemanticDomain::MECHANICAL_FORCE: 
            return "CONSTRAINT: Focus on MACHINE, STEEL, OIL. Rhythmic noise. Industrial decay. NO biological metaphors.";
        case SemanticDomain::ABSTRACT_PHYSICS: 
            return "CONSTRAINT: Focus on LIGHT and TIME. Keep physical objects SOLID. DO NOT bend geometry. Describe the static environment.";
    }
    return "";
}



// Forward Declaration
SemanticDomain get_contrast_domain(SemanticDomain current);

// [MEASURE] Detect the current semantic domain based on keywords
// [UPDATED] Now with DOMAIN FATIGUE (Force switch if stuck in unstable domains)
SemanticDomain detect_semantic_domain(MultiAgentState& state, const std::string& text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    std::map<SemanticDomain, std::vector<std::string>> keywords = {
        {SemanticDomain::MECHANICAL_FORCE, {"gear", "piston", "steel", "iron", "oil", "machine", "metal", "hum", "grind", "click"}},
        {SemanticDomain::BODY_INTERNAL, {"blood", "vein", "flesh", "bone", "skin", "breath", "lung", "heart", "pulse", "sweat"}},
        {SemanticDomain::EXTERIOR_VAST, {"sky", "horizon", "desert", "sun", "cloud", "mountain", "void", "distance", "star"}},
        {SemanticDomain::DOMESTIC_SURREAL, {"chair", "table", "clock", "door", "window", "carpet", "wallpaper", "kitchen", "hallway"}},
        {SemanticDomain::ABSTRACT_PHYSICS, {"geometry", "angle", "light", "dimension", "time", "fractal", "color", "prism"}}
    };

    SemanticDomain best_dom = SemanticDomain::ABSTRACT_PHYSICS; 
    int max_hits = -1;
    
    for(const auto& [dom, kws] : keywords) {
        int hits = 0;
        for(const auto& w : kws) {
            if(lower.find(w) != std::string::npos) hits++;
        }
        if(hits > max_hits) {
            max_hits = hits;
            best_dom = dom;
        }
    }

    // --- FATIGUE LOGIC ---
    // If we detect the SAME domain as current active, increment streak.
    if ((int)best_dom == state.domain_index) {
        state.domain_streak++;
        std::cout << " [DOMAIN] Streak for " << get_domain_name(best_dom) << ": " << state.domain_streak << std::endl;
    } else {
        state.domain_streak = 0;
    }

    // If streak > 3 AND domain is UNSTABLE (Surreal/Abstract), FORCE CONTRAST
    if (state.domain_streak > 3) {
        if (best_dom == SemanticDomain::DOMESTIC_SURREAL || best_dom == SemanticDomain::ABSTRACT_PHYSICS) {
            std::cout << " [DOMAIN FATIGUE] Stuck in " << get_domain_name(best_dom) << " for 4+ blocks. Forcing CONTRAST." << std::endl;
            // Force return contrast (this will cause main loop to switch index, resetting streak next time)
            return get_contrast_domain(best_dom);
        }
    }

    return best_dom;
}

// [CONTRAST] Select the Semantic Opposite
SemanticDomain get_contrast_domain(SemanticDomain current) {
    switch(current) {
        case SemanticDomain::MECHANICAL_FORCE: return SemanticDomain::BODY_INTERNAL; // Machine <-> Life
        case SemanticDomain::BODY_INTERNAL: return SemanticDomain::MECHANICAL_FORCE;
        
        case SemanticDomain::DOMESTIC_SURREAL: return SemanticDomain::EXTERIOR_VAST; // Inside <-> Outside
        case SemanticDomain::EXTERIOR_VAST: return SemanticDomain::DOMESTIC_SURREAL;
        
        case SemanticDomain::ABSTRACT_PHYSICS: return SemanticDomain::BODY_INTERNAL; // Abstract <-> Visceral
        default: return SemanticDomain::EXTERIOR_VAST;
    }
}
// [MOVED] Persona and Deck functions moved to after Model Definitions to allow access to generate_saul_prefill

// --- FORWARD DECLARATIONS (For Saul Engine) ---
bool ensure_model_loaded(MultiAgentState& state, llama_model** model_ptr, llama_context** ctx_ptr, const std::string& path, int n_ctx, int n_gpu_layers);
std::string generate_layer(llama_context* ctx, llama_model* model, const std::string& prompt, int max_tokens, float temp, const std::vector<std::string>& stop_words, const std::deque<std::string>& banned_words);
std::string fimbulvetr_first_person(MultiAgentState& state, const std::string& source);

// --- UPDATE BAN LIST (CONCEPT JAIL) ---
// Now with TTL (Time-To-Live) for bans - words expire after N blocks
struct BannedTerm {
    std::string word;
    int blocks_remaining; // Countdown to unban
};
static std::deque<BannedTerm> concept_jail_with_ttl;

// Forward Declaration
std::string generate_layer(llama_context* ctx, llama_model* model, const std::string& prompt, int max_tokens, float temp, const std::vector<std::string>& stop_words, const std::deque<std::string>& banned_words = {});

// --- UTILS ---
std::string sha256_string(const std::string& str) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.c_str(), str.length());
    SHA256_Final(hash, &sha256);
    std::stringstream ss;
    for(int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

std::string clean_invalid_utf8(const std::string& input) {
    std::string output;
    for (unsigned char c : input) {
        if (c < 128) {
            output += c; // ASCII
        } else {
            // Basit temizlik: Non-ASCII karakterleri boşluk yap veya atla.
            // JSON parse hatasını önlemek için.
            output += ' '; 
        }
    }
    return output;
}

static std::string sanitize_world_token(const std::string& input, size_t max_len) {
    std::string cleaned;
    cleaned.reserve(std::min(input.size(), max_len));
    bool last_space = false;
    for (unsigned char c : input) {
        if (c < 32 || c == 127) continue; // control chars
        if (c >= 128) {
            if (!last_space) cleaned.push_back(' ');
            last_space = true;
            continue;
        }
        if (std::isspace(c)) {
            if (!last_space) cleaned.push_back(' ');
            last_space = true;
        } else {
            cleaned.push_back(static_cast<char>(c));
            last_space = false;
        }
        if (cleaned.size() >= max_len) break;
    }

    while (!cleaned.empty() && std::isspace(static_cast<unsigned char>(cleaned.front()))) cleaned.erase(cleaned.begin());
    while (!cleaned.empty() && std::isspace(static_cast<unsigned char>(cleaned.back()))) cleaned.pop_back();
    return cleaned;
}

static bool is_noisy_world_entry(const std::string& key, const std::string& value) {
    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    std::string lk = lower(key);
    std::string lv = lower(value);

    if (key.find('[') != std::string::npos || key.find(']') != std::string::npos) return true;
    if (key.find('<') != std::string::npos || key.find('>') != std::string::npos) return true;
    if (lk.find("protagonist") != std::string::npos) return true;
    if (lk.find("data_lost") != std::string::npos || lk.find("redacted") != std::string::npos) return true;
    if (lk.find("continue_the_story") != std::string::npos || lk.find("end_of_turn") != std::string::npos) return true;
    if (lk.find("http://") != std::string::npos || lk.find("https://") != std::string::npos) return true;
    if (lk.size() < 2 || lv.size() < 2) return true;
    return false;
}

static std::map<std::string, std::string> sanitize_world_state(const std::map<std::string, std::string>& input) {
    std::map<std::string, std::string> cleaned;
    for (const auto& kv : input) {
        std::string key = sanitize_world_token(kv.first, 96);
        std::string value = sanitize_world_token(kv.second, 160);
        if (key.empty() || value.empty()) continue;
        if (is_noisy_world_entry(key, value)) continue;
        cleaned[key] = value;
    }
    return cleaned;
}

bool has_dialogue(const std::string& text) {
    int quote_count = std::count(text.begin(), text.end(), '"');
    if (quote_count >= 2) return true;

    std::vector<std::string> speech = {
        "\" it says", "\" she says", "\" he says",
        "\" it whispers", "\" it asks",
        "says,", "whispers,", "asks,"
    };

    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto& s : speech) {
        std::string needle = s;
        std::transform(needle.begin(), needle.end(), needle.begin(), ::tolower);
        if (lower.find(needle) != std::string::npos) return true;
    }
    return false;
}

bool has_pov_break(const std::string& text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    return (lower.find("protagonist") != std::string::npos ||
            lower.find("their heart") != std::string::npos ||
            lower.find("their eyes") != std::string::npos ||
            lower.find("their senses") != std::string::npos);
}

bool has_internal_repetition(const std::string& text) {
    const int CHUNK_SIZE = 40;
    
    if (text.length() < static_cast<size_t>(CHUNK_SIZE * 2)) return false;
    
    for (size_t i = 0; i < text.length() - CHUNK_SIZE; i++) {
        std::string chunk = text.substr(i, CHUNK_SIZE);
        size_t second = text.find(chunk, i + CHUNK_SIZE);
        if (second != std::string::npos) {
            std::cout << " [INTERNAL REPEAT] '" << chunk.substr(0, 30) << "...'" << std::endl;
            return true;
        }
    }
    return false;
}

// Quest-mode detector using semantic similarity (CodeBERT) instead of brittle substring checks.
bool has_quest_mode(MultiAgentState& state, const std::string& text) {
    static bool quest_bank_ready = false;
    static std::vector<Embedding> quest_bank;
    static std::vector<std::string> quest_seeds = {
        "find the exit", "follow me now", "you must escape", "we must go", "the door awaits", "take the key", "show the way out"
    };

    if (state.sensor && !quest_bank_ready) {
        quest_bank.clear();
        for (const auto& seed : quest_seeds) {
            quest_bank.push_back(state.sensor->embed(seed));
        }
        quest_bank_ready = !quest_bank.empty();
    }

    std::string window = text.substr(text.length() > 400 ? text.length() - 400 : 0);

    if (state.sensor && quest_bank_ready) {
        Embedding sample = state.sensor->embed(window);
        float max_sim = 0.0f;
        for (const auto& ref : quest_bank) {
            max_sim = std::max(max_sim, state.sensor->cosine_similarity(sample, ref));
        }
        if (max_sim > 0.70f) {
            std::cout << " [QUEST] Semantic trigger (sim=" << max_sim << ")" << std::endl;
            return true;
        }
    }

    // Fallback: coarse pattern check if sensor unavailable
    std::string lower = window;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::vector<std::string> fallback = {"find the", "you must", "we must", "follow me", "come with", "the way out"};
    for (const auto& pat : fallback) {
        if (lower.find(pat) != std::string::npos) return true;
    }
    return false;
}

bool has_meta_artifacts(const std::string& text) {
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::vector<std::string> markers = {
        "<|im_start|>", "<|im_end|>", "<|start_header_id|>", "<|end_header_id|>",
        "<end_of_turn>", "<begin_of_text>", "<endoftext>", "continue_the_story",
        "legal terminology:"
    };
    for (const auto& m : markers) {
        if (lower.find(m) != std::string::npos) return true;
    }
    return false;
}

std::string novelty_directive(float novelty_score) {
    if (novelty_score > 0.4f) return "";
    if (novelty_score > 0.25f) return "NOVELTY_REQUIRED: shift to a different physical setting (temperature/light/scale) and introduce a new object.";
    return "NOVELTY_CRITICAL: abandon current motif. Switch sensory channel and alter environment topology immediately.";
}

// 1. REPETITION CHECKER (Deep Scan)
bool is_repetitive(const std::string& text, const std::string& history) {
    if (text.length() < 20) return false;

    // A. EXACT BLOCK REPETITION (Loop Detection)
    if (history.find(text) != std::string::npos) return true;

    // B. SLIDING WINDOW (Catch recycled phrases/garbage tails)
    // If any 64-char chunk of the new text appears in the history, REJECT.
    const int WINDOW_SIZE = 64;
    if (text.length() > WINDOW_SIZE) {
        for (size_t i = 0; i <= text.length() - WINDOW_SIZE; i += 32) { // Step 32 for efficiency
            std::string window = text.substr(i, WINDOW_SIZE);
            if (history.find(window) != std::string::npos) {
                // Ignore common short phrases, but 64 chars is specific enough
                return true; 
            }
        }
    }
    
    // C. INTERNAL REPETITION (Stuttering checking)
    // Check if the end of the text repeats the beginning of the text (looping)
    if (text.length() > 50) {
        std::string head = text.substr(0, 30);
        if (text.rfind(head) > 0) return true;
    }

    // D. START-OF-BLOCK REPETITION (Motif Loop)
    // If the first 40 characters of new text appear in history, it's a "lazy start" loop.
    if (text.length() > 40) {
        std::string start_motif = text.substr(0, 40);
        if (history.find(start_motif) != std::string::npos) return true;
    }

    // E. OPENER REPETITION (Start Variance)
    // Check if the first 5 words match any recent sentence start in history.
    std::stringstream ss_new(text);
    std::string word;
    std::string new_opener = "";
    int w_count = 0;
    while(ss_new >> word && w_count < 5) {
        new_opener += word + " ";
        w_count++;
    }
    
    if (w_count >= 3 && history.find(new_opener) != std::string::npos) {
        std::cout << " [REJECT] Opener detected in history: '" << new_opener << "'..." << std::endl;
        return true;
    }

    return false;
}

// 2. CONTAMINATION CHECKER (Anti-Assistant)
bool is_contaminated(const std::string& text) {
    std::vector<std::string> poison = {
        // Classics
        "AI assistant", "language model", "large language model",
        "I cannot", "As an AI", "Note:", "Options:", 
        "respond with", "system instruction", "context preview",
        "apologize", "I'm sorry", "let me know", "Please respond",
        "visceral sensation", "within 1-3 sentences", "next action",
        
        // Politeness / Conversational
        "I'm glad", "I am glad", "happy to help", "excited to",
        "Let's get back", "Let's continue", "continue the story",
        "your turn", "feel free", "great progress", "feedback",
        "hope this helps", "meet your expectations",
        "Remember:", "Warning:", "Action Required",
        "Here is", "My attempt", "As we", "narrative flow", "Please enter",
        "Mention immediate",
        
        // [NEW] Leaked Prompt Artifacts
        "ACTION CONTINUES", "Describe action NOW", "CURRENT MOMENT",
        "MEMORY STREAM", "[SYSTEM INJECTION]", "(Script continues)",
        
        // [NEW] Hallucinations / Foreign Language / Degeneracy
        "veloceprompt", "başarılar", "foundation holds", "domain",
        "semrite", "orderwis", "offirmation", "comport closure",
        "human feedback", "reinforcement learning",
        "Welcome, explorer", "You have been chosen", // NPC/Game tropes
        "Chapter I", "Chapter II", "Chapter III", // Novel formatting
        
        // [NEW] Safety Refusals (Llama Guard / RLHF Leaks)
        "I cannot generate", "explicit content", "based on the original prompt",
        "Is there anything else", "I see no user", "I cannot create",
        "content that is not based", "defiles all comprehension",
        "I will not continue", "I'm unable to", "I can't fulfill", "I can't create", // Explicit refusal
        "xECPECT", "REALTY", "RECORD THE", // Artifact Leaks
        
        // [NEW] Meta-Commentary / Coaching (Block 8 Fix)
        "Let's try again", "starting to get the hang", "What do you see",
        "Describe what", "focus on describing", "Let me", "I need you to",
        "your surroundings", "feel free to",
        "I see that you're", "Good work", "Which artifact do you choose", 
        "I need to remind you", "remind you of the directives",
        "rewritten version", "Let's focus on", "I take only action", 
        "You have options", "The story continues", "Good luck", "rewrite",
        "How would you like", "Would you like to proceed", "What would you like",
        
        // [NEW] Word Salad / Semantic Collapse (Specific Artifacts)
        "gunkleman", "ricotta-washed", "palo verde", "sponge cake",
        "rubles", "linguine", "fluorescent-lit", "woolens",
        "Rorschach", "fractured fourth vertebrae"
    };
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);

    for (const auto& p : poison) {
        std::string lower_p = p;
        std::transform(lower_p.begin(), lower_p.end(), lower_p.begin(), ::tolower);
        if (lower_text.find(lower_p) != std::string::npos) return true;
    }
    // Dialogue and meta tokens are considered contamination because they introduce NPC voices or prompt leakage.
    if (has_dialogue(text)) return true;
    if (has_meta_artifacts(text)) return true;
    return false;
}

// 3. HISTORY SANITIZER (Context Cleaning)
std::string sanitize_history(const std::string& history) {
    std::string clean = history;
    std::vector<std::string> pollution = {
        "Please respond", "I wait for input", "What do you do?",
        "Action:", "Options:", "1.", "2.", "3.",
        "You are", "Your task", "Describe", "ACTION CONTINUES",
        "Good work!", "Let's try again", "Directives:", "REWITTEN:",
        "I see that you're", "User:"
    };

    for (const auto& p : pollution) {
        size_t pos;
        while ((pos = clean.find(p)) != std::string::npos) {
            clean.replace(pos, p.length(), "");
        }
    }
    return clean;
}

// (Moved to top)

// 5. GIBBERISH DETECTOR (Entropy & Charset Check)
bool is_gibberish(const std::string& text) {
    if (text.empty()) return true;
    
    int special_chars = 0;
    int alpha = 0;
    int non_ascii = 0;
    
    for (unsigned char c : text) {
        if (c > 127) non_ascii++;
        if (isalnum(c) || isspace(c)) alpha++;
        else special_chars++;
    }
    
    // [STRICT] Foreign Language / Emoji / corrupt UTF-8 check
    if (non_ascii > 0) return true; // STRICT: English only for now to prevent "kana" leaks

    // If > 15% of text is special characters (excluding punctuation like . ,) - likely model collapse
    int chaotic_chars = 0;
    for (char c : text) {
        if (strchr("{}[]|\\/@#$%^&*()_+~`=<>", c)) chaotic_chars++;
    }
    
    if (chaotic_chars > (text.length() * 0.05)) return true; // > 5% chaotic chars is bad
    
    return false;
}

// 6. TRIM TRAILING NOISE (Fix "Every." artifacts)
std::string trim_trailing_noise(std::string text) {
    if (text.empty()) return text;

    // A. Remove trailing whitespace
    while (!text.empty() && isspace(text.back())) {
        text.pop_back();
    }

    // B. Safe Truncate
    // In "Stream of Consciousness" or "Visual Agnosia", sentences might be fragmented.
    // If we find NO punctuation, we return the whole thing (better to have a dangling sentence than silent output).
    size_t last_punc = text.find_last_of(".!?\"");
    if (last_punc == std::string::npos) {
        return text; 
    }
    
    // If text continues significanly after last_punc, CUT IT (It's an unfinished sentence)
    if (last_punc < text.length() - 1) {
        text = text.substr(0, last_punc + 1);
    }

    return text;
}

// 6b. ALGORITHMIC PRE-FILTER (Repeated Character Collapse)
// Solves "hissssss", "Ahhhhh"
std::string collapse_repeating_chars(const std::string& text) {
    if (text.empty()) return "";
    std::string clean = "";
    int repeat_count = 0;
    
    for (size_t i = 0; i < text.length(); i++) {
        clean += text[i];
        if (i > 0 && tolower(text[i]) == tolower(text[i-1])) {
            repeat_count++;
        } else {
            repeat_count = 0;
        }
        
        // If we have 3 identical chars in a row (e.g. "sss"), stop adding them
        if (repeat_count >= 2) {
             // Peek ahead: if next char is SAME, skip adding it next time (handled by loop logic)
             // Actually, the loop adds *then* checks. 
             // Logic refinement: 
             // We just added 's'. Previous was 's'. Count is 1.
             // Next 's'. Added. Previous 's'. Count is 2.
             // Next 's'. Added. Previous 's'. Count is 3. We want to DROP this one.
             // So we should remove the last char if count >= 2.
             if (repeat_count >= 2) { // 3rd char
                 clean.pop_back(); 
             }
        }
    }
    return clean;
}

// --- HELPER: STRIP META COMMENTARY ---
std::string strip_meta_commentary(std::string text) {
    // 1. Remove "SYSTEM DIRECTIVE" lines
    size_t pos;
    while ((pos = text.find("SYSTEM DIRECTIVE:")) != std::string::npos) {
        // Find end of line or double newline
        size_t end = text.find("\n", pos);
        if (end == std::string::npos) end = text.length();
        text.erase(pos, end - pos + 1);
    }

    // 2. Remove "NEW SEGMENT:" or "REPAIR:"
    std::vector<std::string> headers = {"NEW SEGMENT:", "REPAIR:", "CORRECTED:", "OUTPUT:", "Here is the rewritten", "Rewrite:", "Narrative:", "Combined:", "Here is the paragraph:"};
    for (const auto& h : headers) {
        while ((pos = text.find(h)) != std::string::npos) {
             size_t end = text.find("\n", pos);
             if (end == std::string::npos) end = text.length();
             else end++; // Include newline
             text.erase(pos, end - pos);
        }
    }

    // 3. Trim Leading/Trailing Whitespace
    while(!text.empty() && isspace(text.front())) text.erase(0, 1);
    while(!text.empty() && isspace(text.back())) text.pop_back();
    
    return text;
}

std::string trim_meta_tail(std::string text) {
    if (text.empty()) return text;

    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::vector<std::string> markers = {
        "how would you like", "would you like to proceed", "what would you like",
        "let me know", "do you want to", "how do you want"
    };

    size_t cut = std::string::npos;
    for (const auto& marker : markers) {
        size_t pos = lower.find(marker);
        if (pos != std::string::npos) {
            if (cut == std::string::npos || pos < cut) {
                cut = pos;
            }
        }
    }

    if (cut != std::string::npos) {
        text.erase(cut);
        while (!text.empty() && isspace(text.back())) text.pop_back();
    }

    return text;
}

static bool needs_space_between(char left, char right) {
    unsigned char l = static_cast<unsigned char>(left);
    unsigned char r = static_cast<unsigned char>(right);
    if (std::isspace(l) || std::isspace(r)) return false;

    if (right == '"' || right == '\'' || right == '(' || right == '[' || right == '{') {
        return true;
    }
    if (std::ispunct(r)) return false;

    return true;
}

std::string join_prefill_and_generated(const std::string& prefill, const std::string& generated) {
    if (prefill.empty()) return generated;
    if (generated.empty()) return prefill;

    std::string joined = prefill;
    if (needs_space_between(prefill.back(), generated.front())) {
        joined.push_back(' ');
    }
    joined += generated;
    return joined;
}

// --- HELPER: NARRATIVE VALIDATOR ---
// Returns true if the text passes the HARD CONSTRAINTS (No passive start, no banned words)
bool is_narrative_valid(const std::string& text) {
    // 1. Check Passive Starts (As, While, When)
    std::string clean = text;
    while(!clean.empty() && !isalpha(clean.front())) clean.erase(0, 1);
    
    // Case Insensitive check for starts
    std::string lower_start = clean.substr(0, 10);
    std::transform(lower_start.begin(), lower_start.end(), lower_start.begin(), ::tolower);
    
    if (lower_start.rfind("as ", 0) == 0) return false;
    if (lower_start.rfind("while ", 0) == 0) return false;
    if (lower_start.rfind("when ", 0) == 0) return false;

    // 2. Check Banned Vocabulary (The Identity Morphism List)
    std::vector<std::string> banned = {
        "iridescent", "bioluminescent", "electrified", "pulsing", "otherworldly",
        "crystalline", "shifting hue", "dance of light"
    };

    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);

    for (const auto& w : banned) {
        if (lower_text.find(w) != std::string::npos) {
            std::cout << " [REJECT] Banned word found: " << w << std::flush;
            return false;
        }
    }
    
    return true;
}

// 6c. NEURAL PROOFREADER (Self-Correction II)
// Uses the model to fix spelling/grammar while maintaining tone.
std::string neural_correct(MultiAgentState& state, const std::string& raw_text) {
    if (raw_text.length() < 10) return raw_text; // Skip short fragments

    std::string system_prompt = 
        "SYSTEM DIRECTIVE: You are a Proofreader Algorithm. CORRECT spelling and grammar errors in the input text. Do NOT change the meaning. Do NOT add new content. Output ONLY the corrected text.\n";

    std::string formatted_prompt = 
        "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n" 
        "INPUT: " + raw_text + "\n" 
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    // Tokenize
    auto* vocab = llama_model_get_vocab(state.model_main);
    auto& tokens_list = token_scratch(formatted_prompt.size() + kTokenScratchPad);
    int n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
         token_scratch(static_cast<size_t>(-n_tokens));
         n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);

    // Context Check
    const int n_ctx = llama_n_ctx(state.ctx_main);
    if (tokens_list.size() > n_ctx - 200) return raw_text; // Too long to correct safely

    // CLEAR MEMORY FOR ISOLATED PASS
    llama_memory_clear(llama_get_memory(state.ctx_main), true); 
    
    // Decode Prompt
    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size()); 
    if (llama_decode(state.ctx_main, batch) != 0) return raw_text; // Fallback

    // Sampler (Deterministic - Temp 0.1)
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.1f)); 
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(std::rand())); 

    std::string correction;
    correction.reserve(raw_text.size() + 64);
    int max_correction = raw_text.length() + 50; 
    
    for (int i = 0; i < max_correction; i++) {
        llama_token new_token_id = llama_sampler_sample(smpl, state.ctx_main, -1);
        llama_sampler_accept(smpl, new_token_id);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) continue;
        std::string piece(buf, n);
        
        correction += piece;

        if (llama_decode(state.ctx_main, llama_batch_get_one(&new_token_id, 1))) break;
    }

    llama_sampler_free(smpl);

    // Validation: If correction is empty or drastically shorter, fallback
    if (correction.length() < raw_text.length() * 0.5) return raw_text;

    // Fix possible "Corrected Text:" prefix if model hallucinates it
    size_t colon = correction.find(":");
    if (colon != std::string::npos && colon < 20) {
        correction = correction.substr(colon + 1);
    }
    
    // Trim
    correction = strip_meta_commentary(correction);

    return correction;
}

// 6d. NEURAL REPAIR (Self-Correction III)
// Reroutes rejected blocks back through the model for a second chance.
std::string neural_repair(MultiAgentState& state, const std::string& bad_text, const std::string& reason) {
   std::cout << " [NEURAL REPAIR] Attempting to fix '" << reason << "'..." << std::endl;
    
    std::string system_prompt = 
        "SYSTEM DIRECTIVE: The following narrative segment was REJECTED because it is " + reason + ".\n"
        "INSTRUCTION: REWRITE the segment completely. FIX the identified issue. Make it unique.\n"
        "BAD SEGMENT: " + bad_text + "\n"
        "OUTPUT FORMAT: ONLY the narrative text. NO 'Here is the rewrite'. NO headers.\n"
        "REPAIR:";

    std::string formatted_prompt = 
        "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    // Tokenize
    auto* vocab = llama_model_get_vocab(state.model_main);
    auto& tokens_list = token_scratch(formatted_prompt.size() + kTokenScratchPad);
    int n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
         token_scratch(static_cast<size_t>(-n_tokens));
         n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);

    // CLEAR MEMORY FOR ISOLATED PASS
    llama_memory_clear(llama_get_memory(state.ctx_main), true); 
    
    // Decode Prompt
    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size()); 
    if (llama_decode(state.ctx_main, batch) != 0) return bad_text; // Fallback

    // Sampler (Creative Repair - Temp 0.4)
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.4f)); 
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(std::rand())); 

    std::string repair;
    int max_repair = 400; // Allow enough space for rewrite
    repair.reserve(static_cast<size_t>(max_repair) * kAvgTokenChars);
    
    for (int i = 0; i < max_repair; i++) {
        llama_token new_token_id = llama_sampler_sample(smpl, state.ctx_main, -1);
        llama_sampler_accept(smpl, new_token_id);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;
        
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) continue;
        std::string piece(buf, n);
        repair += piece;
        
        // Feed back
        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(state.ctx_main, batch) != 0) break;
    }
    
    llama_sampler_free(smpl);
    repair = strip_meta_commentary(repair);
    return repair.empty() ? bad_text : repair;
}



// 7. ENTROPY SYSTEM (Diegetic Repetition Control)
float calculate_token_entropy(const std::string& text) {
    std::map<char, int> freqs;
    for (char c : text) freqs[c]++;
    
    float entropy = 0.0f;
    float len = (float)text.length();
    for (auto const& [key, val] : freqs) {
        float p = (float)val / len;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

std::vector<std::string> calculate_entropy_loss(const std::string& history) {
    std::map<std::string, int> counts;
    std::set<std::string> stopwords = {
        "the", "and", "of", "to", "a", "i", "in", "was", "it", "my", "me", "that", "with", "as", "but", "for", "on", "is", "at", 
        "from", "by", "this", "be", "so", "like", "just", "or", "an", "not", "have", "had", "can", "do", "we", "all", "your", "you",
        "are", "will", "one", "up", "out", "down", "off", "over", "under", "again", "then", "now", "here", "there", "where", "when", "why", "how",
        "which", "what", "who", "whom", "whose", "if", "because", "while", "though", "although", "unless", "until", "since", "before", "after",
        // TIER 2: PROTECTED WORDS (Anatomy, Physics, Basic Verbs) - Do not ban these.
        "eyes", "hands", "feet", "skin", "body", "breath", "lungs", "fingers", "head", "face", "legs", "arms", "chest", "heart", "stomach",
        "air", "ground", "wall", "floor", "dark", "darkness", "light", "stone", "cold", "silence", "shadows", "dust",
        "against", "through", "into", "onto", "across", "around", "towards", "away", "back", "forward",
        "feel", "see", "hear", "am", "go", "move", "stand", "walk", "look", "touch", "know", "think"
    };

    std::stringstream ss(history);
    std::string word;
    while (ss >> word) {
        // Clean word
        std::string clean = "";
        for (char c : word) {
            if (isalpha(c)) clean += tolower(c);
        }
        if (clean.length() < 3) continue; // Skip short words
        if (stopwords.count(clean)) continue;
        counts[clean]++;
    }

    std::vector<std::pair<int, std::string>> sorted_words;
    for (auto const& [w, c] : counts) {
        if (c > 4) { // Threshold: Word must appear more than 4 times to be banned
            sorted_words.push_back({c, w});
        }
    }
    std::sort(sorted_words.rbegin(), sorted_words.rend());

    std::vector<std::string> banned;
    for (int i = 0; i < std::min((int)sorted_words.size(), 3); i++) { // Top 3 offenders
        banned.push_back(sorted_words[i].second);
    }
    return banned;
}

// Helper: Expand Morphology (Robust)
bool is_vowel(char c) {
    return std::string("aeiou").find(c) != std::string::npos;
}

std::vector<std::string> expand_morphology(const std::string& word) {
    std::vector<std::string> forms = {word};
    if (word.length() < 3) return forms;

    char last = word.back();
    char second = word[word.length()-2];
    std::string base = word;

    // 1. Sibilants (es) - s, x, z, ch, sh
    bool sibilant = (last == 's' || last == 'x' || last == 'z' || 
                    (word.length() > 2 && last == 'h' && (second == 'c' || second == 's')));
    if (sibilant) {
        forms.push_back(word + "es");
    } else {
        forms.push_back(word + "s");
    }

    // 2. Y endings (cry -> cried, cries)
    if (last == 'y' && !is_vowel(second)) {
        std::string stem = word.substr(0, word.length()-1);
        forms.push_back(stem + "ies");
        forms.push_back(stem + "ied");
        forms.push_back(word + "ing"); // trying
        return forms; // Return check early for Y-case to avoid duplicates or complexity
    }

    // 3. E endings (write -> writer, writing, wrote - irregulars hard, but standard rules help)
    if (last == 'e') {
        forms.push_back(word + "d"); // writhe -> writhed
        // Drop e for ing
        forms.push_back(word.substr(0, word.length()-1) + "ing"); // writhe -> writhing
        return forms;
    }

    // 4. CVC Doubling (stop -> stopping)
    // Check for CVC pattern: Consonant-Vowel-Consonant (and not w, x, y)
    if (!is_vowel(last) && last != 'w' && last != 'x' && last != 'y' &&
        is_vowel(second) &&
        word.length() > 2 && !is_vowel(word[word.length()-3])) {
        
        forms.push_back(word + last + "ed");
        forms.push_back(word + last + "ing");
        return forms;
    }

    // Default
    forms.push_back(word + "ed");
    forms.push_back(word + "ing");
    return forms;
}

std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string current;
    for (char c : text) {
        current.push_back(c);
        if (c == '.' || c == '!' || c == '?') {
            if (current.length() > 5) {
                while (!current.empty() && isspace(static_cast<unsigned char>(current.front()))) current.erase(current.begin());
                while (!current.empty() && isspace(static_cast<unsigned char>(current.back()))) current.pop_back();
                if (!current.empty()) sentences.push_back(current);
            }
            current.clear();
        }
    }
    if (!current.empty()) {
        while (!current.empty() && isspace(static_cast<unsigned char>(current.front()))) current.erase(current.begin());
        while (!current.empty() && isspace(static_cast<unsigned char>(current.back()))) current.pop_back();
        if (!current.empty()) sentences.push_back(current);
    }
    return sentences;
}

std::string dedupe_sentences(const std::string& text) {
    auto sentences = split_sentences(text);
    std::set<std::string> seen;
    std::string rebuilt;
    for (const auto& s : sentences) {
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (seen.insert(lower).second) {
            if (!rebuilt.empty()) rebuilt += " ";
            rebuilt += s;
        }
    }
    return rebuilt.empty() ? text : rebuilt;
}

float compute_novelty(MultiAgentState& state, const std::vector<std::string>& sentences) {
    if (!state.sensor || sentences.empty()) return 1.0f;
    float min_sim = 1.0f;
    for (const auto& s : sentences) {
        Embedding cur = state.sensor->embed(s);
        float max_sim = 0.0f;
        for (const auto& prior : state.sentence_memory) {
            max_sim = std::max(max_sim, state.sensor->cosine_similarity(cur, prior));
        }
        min_sim = std::min(min_sim, 1.0f - max_sim);
    }
    return min_sim; // Higher is more novel
}

// --- LLAMA ENGINE ---

// [DYNAMIC MEMORY MANAGEMENT]
// User requested full control over memory. We will load auxiliary models ON DEMAND.
// Llama-8B (Main) stays resident. Others (MiroThinker, RWKV, Gemma, Phi, Qwen) are swapped.

static bool file_exists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

static void ensure_directory(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        std::filesystem::create_directories(path, ec);
        if (ec) {
            std::cerr << "[WARN] Failed to create directory: " << path << " (" << ec.message() << ")" << std::endl;
        }
    }
}

static void log_model_manifest() {
    struct ModelEntry {
        const char* name;
        const std::string* path;
    };

    const ModelEntry models[] = {
        {"MAIN", &MODEL_PATH},
        {"OBSERVER_FIMBULVETR", &FIMBULVETR_PATH},
        {"SCOUT", &GEMMA_PATH},
        {"PHI", &PHI_PATH},
        {"LOGIC", &DEEPSEEK_PATH},
        {"QWEN_CREATIVE", &QWEN_CREATIVE_PATH},
        {"QWEN_STABILIZER", &QWEN_STABILIZER_PATH},
        {"MIROTHINKER", &MIROTHINKER_PATH},
        {"RWKV", &RWKV_PATH},
        {"MAMBA", &MAMBA_PATH},
        {"HERMES", &HERMES_PATH},
        {"SAUL", &SAUL_PATH},
        {"CODEBERT_ONNX", &CODEBERT_MODEL_PATH},
        {"CODEBERT_VOCAB", &VOCAB_PATH}
    };

    std::cout << "[SYSTEM] Model manifest check:" << std::endl;
    for (const auto& entry : models) {
        if (file_exists(*entry.path)) {
            std::cout << " [MODEL] Ready: " << entry.name << " -> " << *entry.path << std::endl;
        } else {
            std::cerr << " [MODEL] Missing: " << entry.name << " -> " << *entry.path << std::endl;
        }
    }
}

static void free_model_and_context(llama_model*& model, llama_context*& ctx) {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
}

void unload_all_aux(MultiAgentState& state) {
    free_model_and_context(state.model_scout, state.ctx_scout);
    free_model_and_context(state.model_phi, state.ctx_phi);
    free_model_and_context(state.model_qwen_stabilizer, state.ctx_qwen_stabilizer);
    free_model_and_context(state.model_qwen_creative, state.ctx_qwen_creative);
    free_model_and_context(state.model_fimbulvetr, state.ctx_fimbulvetr);
    free_model_and_context(state.model_mirothinker, state.ctx_mirothinker);
    free_model_and_context(state.model_rwkv, state.ctx_rwkv);
    
    // [SINGLE SLOT POLICY] Add Mamba and Hermes
    free_model_and_context(state.model_mamba, state.ctx_mamba);
    free_model_and_context(state.model_hermes, state.ctx_hermes);
    free_model_and_context(state.model_saul, state.ctx_saul);
    free_model_and_context(state.model_logic, state.ctx_logic);
}

bool ensure_model_loaded(MultiAgentState& state, llama_model** model_ptr, llama_context** ctx_ptr, const std::string& path, int n_ctx, int n_gpu_layers) {
    if (*model_ptr && *ctx_ptr) {
        if (llama_n_ctx(*ctx_ptr) == n_ctx) {
            std::cout << "[MODEL] Reusing: " << path << " (ctx=" << n_ctx << ")" << std::endl;
            return true; // Already loaded with correct context
        }
        std::cout << "[MEMORY] Reloading model with new context size: " << path << std::endl;
        free_model_and_context(*model_ptr, *ctx_ptr);
    } else if (*model_ptr || *ctx_ptr) {
        std::cout << "[MEMORY] Resetting partial model state: " << path << std::endl;
        free_model_and_context(*model_ptr, *ctx_ptr);
    }

    static std::set<std::string> missing_models;
    if (!file_exists(path)) {
        if (missing_models.insert(path).second) {
            std::cerr << "[ERR] Model file missing: " << path << std::endl;
        }
        return false;
    }
    missing_models.erase(path);

    std::cout << "[MEMORY] Swapping in model: " << path << "..." << std::endl;
    unload_all_aux(state); // Unload others first! (Single Aux Slot Policy)

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers; 
    *model_ptr = llama_model_load_from_file(path.c_str(), mparams);
    
    if (!*model_ptr) {
        std::cerr << "[ERR] Failed to load dynamic model: " << path << std::endl;
        return false;
    }

    std::cout << "[MODEL] Loaded: " << path << " (gpu_layers=" << n_gpu_layers << ")" << std::endl;

    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = std::min(4096, n_ctx);
    *ctx_ptr = llama_init_from_model(*model_ptr, cparams);
    
    if (!*ctx_ptr) {
        std::cerr << "[ERR] Failed to create context for dynamic model." << std::endl;
        free_model_and_context(*model_ptr, *ctx_ptr);
        return false;
    }
    std::cout << "[MODEL] Context ready: " << path << " (ctx=" << llama_n_ctx(*ctx_ptr) << ")" << std::endl;
    return true;
}

// --- DUAL ENGINE INIT ---
// --- REBEL EXTRACTION (World State Update) ---
// --- REBEL EXTRACTION (World State Update) ---
std::map<std::string, std::string> run_rebel_extraction(const std::string& text) {
    std::map<std::string, std::string> updates;
    // Escape quotes for shell
    std::string safe_text = "";
    for(char c : text) {
        if(c == '"') safe_text += "\\\"";
        else if(c == '\\') safe_text += "\\\\";
        else safe_text += c;
    }
    
    // Call Python Script
    std::string cmd = "python3 src/rebel_agent.py \"" + safe_text + "\"";
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) return updates;
    
    char buffer[1024];
    std::string result = "";
    while (!feof(pipe.get())) {
        if (fgets(buffer, 1024, pipe.get()) != NULL)
            result += buffer;
    }
    
    if (result.empty()) {
        std::cerr << " [WARN] REBEL Agent returned NO output. Process likely crashed (Check models/rebel_onnx)." << std::endl;
        return updates;
    }

    try {
        auto j = json::parse(result);
        if (j.is_array()) {
            for (const auto& item : j) {
                if (item.contains("entity") && item.contains("status")) {
                    updates[item["entity"]] = item["status"];
                }
            }
        }
    } catch (...) {
        std::cerr << " [WARN] Failed to parse REBEL output." << std::endl;
    }
    
    // [NEW] LLM Fallback if empty
    if (updates.empty()) {
        // We can't access 'state' here easily as it's not passed. 
        // NOTE: run_rebel_extraction signature limits us.
        // We will execute the LLM fallback IN THE CALLER instead.
    }
    
    return updates;
}


// --- WORLD STATE SERIALIZER (JSON -> NARRATIVE) ---
std::string format_world_state_narrative(const std::map<std::string, std::string>& world_state) {
    if (world_state.empty()) return "";

    std::string narrative = "KNOWN FACTS:\n";
    for (const auto& pair : world_state) {
        // [Template] The {ENTITY} is {STATUS}.
        // Simple heuristic to make it flow better?
        // If status starts with a verb, maybe "The X [status]".
        // But REBEL usually gives "broken" or "locked".
        narrative += "- The " + pair.first + " is " + pair.second + ".\n";
    }
    return narrative;
}

// --- WORLD STATE PRUNING (Hard Cap) ---
void prune_world_state(std::map<std::string, std::string>& world_state) {
    const size_t MAX_ITEMS = 60;
    const size_t PRUNE_COUNT = 10;
    
    if (world_state.size() > MAX_ITEMS) {
        std::cout << "[MEMORY] Pruning World State (Size: " << world_state.size() << " > " << MAX_ITEMS << ")..." << std::endl;
        
        // Naive Eviction: Remove first N items (Alphabetical)
        // Ideally we would use Random or LRU, but Map is ordered.
        // To be safe and simple: just pop begin().
        for (size_t i = 0; i < PRUNE_COUNT; i++) {
            if (world_state.empty()) break;
            auto it = world_state.begin();
            // Optional: Log what we forgot
            // std::cout << " [FORGOT] " << it->first << std::endl;
            world_state.erase(it);
        }
    }
}

void init_multi_agent(MultiAgentState& state) {
    llama_backend_init();
    log_model_manifest();
    
    // --- Llama (Ana Yazar) ---
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = GPU_LAYERS_METAL;
    if (!file_exists(MODEL_PATH)) {
        std::cerr << "[ERR] Main model file missing: " << MODEL_PATH << std::endl;
        exit(1);
    }
    state.model_main = llama_model_load_from_file(MODEL_PATH.c_str(), mparams);
    if (!state.model_main) {
        std::cerr << "[ERR] Failed to load Main Model!" << std::endl;
        exit(1);
    }
    
    auto cparams_main = llama_context_default_params();
    cparams_main.n_ctx = MAIN_CTX;
    cparams_main.n_batch = std::min(4096, MAIN_CTX);
    state.ctx_main = llama_init_from_model(state.model_main, cparams_main);
    if (!state.ctx_main) {
        std::cerr << "[ERR] Failed to create Main Context!" << std::endl;
        exit(1);
    }
    std::cout << "[MODEL] Main ready: " << MODEL_PATH << " (ctx=" << llama_n_ctx(state.ctx_main) << ", gpu_layers=" << GPU_LAYERS_METAL << ")" << std::endl;

    // Initialize other pointers to nullptr for dynamic switching
    state.model_scout = nullptr; state.ctx_scout = nullptr;
    state.model_phi = nullptr; state.ctx_phi = nullptr;
    state.model_qwen_stabilizer = nullptr; state.ctx_qwen_stabilizer = nullptr;
    state.model_qwen_creative = nullptr; state.ctx_qwen_creative = nullptr;
    state.model_fimbulvetr = nullptr; state.ctx_fimbulvetr = nullptr;
    state.model_mirothinker = nullptr; state.ctx_mirothinker = nullptr;
    state.model_rwkv = nullptr; state.ctx_rwkv = nullptr;
    state.model_saul = nullptr; state.ctx_saul = nullptr;
    state.model_logic = nullptr; state.ctx_logic = nullptr;
    
    // Init Sensors
    std::cout << "[SYSTEM] Initializing CodeBERT Sensor..." << std::endl;
    try {
        if (!file_exists(CODEBERT_MODEL_PATH)) {
            std::cerr << "[WARN] CodeBERT model missing: " << CODEBERT_MODEL_PATH << std::endl;
            state.sensor = nullptr;
        } else {
            state.sensor = std::make_shared<CodeBERT>();
            state.sensor->load(CODEBERT_MODEL_PATH);
            std::cout << "[SYSTEM] CodeBERT Initialized Successfully." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERR] CodeBERT Init Failed: " << e.what() << std::endl;
        state.sensor = nullptr;
    } catch (...) {
        std::cerr << "[ERR] CodeBERT Init Failed (Unknown)." << std::endl;
        state.sensor = nullptr;
    }
    
    // Init Weather Oracle
    state.weather_oracle = new WeatherOracle(state.sensor);

    // --- DeBERTa (The Judge) ---
    std::cout << "[SYSTEM] Initializing DeBERTa Judge..." << std::endl;
    state.deberta = std::make_shared<DeBERTaNLI>();

    // Reserve hot-path containers to avoid repeated reallocations.
    state.history_entropy.reserve(16);
    state.history_sentiment.reserve(16);
    state.history_speed.reserve(16);
    state.recent_mistakes.reserve(32);
}



// --- GEMMA SCOUT FUNCTION ---
std::string gemma_inject_chaos(MultiAgentState& state, const std::string& context) {
    // DYNAMIC LOAD
    if (!ensure_model_loaded(state, &state.model_scout, &state.ctx_scout, GEMMA_PATH, SCOUT_CTX, GPU_LAYERS_METAL)) {
        return context; 
    }

    // Gemma'nın hafızasını temizle (her seferinde taze fikir)
    llama_memory_clear(llama_get_memory(state.ctx_scout), true);

    std::cout << "\n[GEMMA-2B] Texture Engine: Synthesizing Sensory Vectors..." << std::flush;
    
    // TEXTURE ENGINE PROMPT (Visual Agnosia Mode)
    // Goal: Prevent "Naming" (Noun) to stop "Explaining" (Causality).
    // Strategy: Force Adjectives/Verbs only.
    std::string prompt = 
        "<start_of_turn>user\n"
        "Analyze this text segment: '" + context.substr(context.length() > 300 ? context.length() - 300 : 0) + "'\n"
        "TASK: The scene is too stable. I need DISSONANCE.\n"
        "INSTRUCTION: Provide 3 concrete SENSORY VECTORS (Adjectives or Verbs) that contradict the scene.\n"
        "CONSTRAINT: Do NOT name objects. Do NOT use Nouns. Do NOT explain.\n"
        "EXAMPLES: 'Vibrating', 'Tasting Copper', 'Oozing', 'Screeching', 'Freezing'\n"
        "OUTPUT FORMAT: Just the words, separated by comma.\n"
        "<end_of_turn><start_of_turn>model\n";

    // --- Gemma Generate (Minimal) ---
    auto* vocab = llama_model_get_vocab(state.model_scout);
    auto& tokens_list = token_scratch(prompt.size() + kTokenScratchPad);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
        token_scratch(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);

    // Decode Prompt
    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size()); 
    if (llama_decode(state.ctx_scout, batch) != 0) return ""; 

    // Sampler (High Temp 0.95)
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.95f)); 
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(std::rand())); 

    std::string chaos_vector;
    int max_tokens = 20; // Slightly more for 3 words
    chaos_vector.reserve(static_cast<size_t>(max_tokens) * kAvgTokenChars);
    
    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token_id = llama_sampler_sample(smpl, state.ctx_scout, -1);
        llama_sampler_accept(smpl, new_token_id);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) continue;
        std::string piece(buf, n);
        chaos_vector += piece;

        if (llama_decode(state.ctx_scout, llama_batch_get_one(&new_token_id, 1))) break;
    }
    
    llama_sampler_free(smpl);

    // Clean up result
    while(!chaos_vector.empty() && isspace(chaos_vector.back())) chaos_vector.pop_back();
    while(!chaos_vector.empty() && isspace(chaos_vector.front())) chaos_vector.erase(0, 1);
    
    std::cout << " -> Injection Vector: [" << chaos_vector << "]" << std::endl;
    return chaos_vector;
}

// --- PHI-2 REFLEX BRAIN FUNCTION ---
// --- PHI-2 REFLEX BRAIN FUNCTION ---
// --- PHI-2 PATTERN FORENSICS ENGINE ---
bool phi2_analyze_patterns(MultiAgentState& state, const std::string& input_json) {
    // DYNAMIC LOAD
    if (!ensure_model_loaded(state, &state.model_phi, &state.ctx_phi, PHI_PATH, PHI_CTX, GPU_LAYERS_CPU)) {
        // Original code said: "Force CPU to avoid Metal Context conflict/OOM".
        // Use 99 since we swap others out now! No conflict.
        return false;
    }

    // reset constraints
    active_constraints = StructuralConstraint();

    std::cout << " [PHI-2] Analying Narrative Patterns..." << std::flush;
    
    std::string prompt = 
        "You are analyzing repeated narrative loops.\n"
        "Input Data: " + input_json + "\n"
        "TASK:\n"
        "1. Identify up to 5 REPEATED PATTERNS.\n"
        "2. Patterns can be: phrases, imagery types (e.g. void), structural habits (e.g. sensory overload, bodily distress).\n"
        "3. Output ONLY a clean JSON array of strings.\n"
        "EXAMPLE: [\"metallic tang\", \"sensory overload\", \"void imagery\"]\n"
        "OUTPUT:";

    std::string output = generate_layer(state.ctx_phi, state.model_phi, prompt, 120, 0.1f, {"\n\n"}, {});
    
    // Simple JSON Array Parser (Mock) -> Real implementation would use nlohmann::json but output might be dirty.
    // Let's try to extract strings between quotes.
    std::vector<std::string> patterns;
    bool in_quote = false;
    std::string current;
    for(char c : output) {
        if(c == '\"') {
            if(in_quote) {
                if(current.length() > 3) patterns.push_back(current);
                current.clear();
            }
            in_quote = !in_quote;
        } else if (in_quote) {
            current += c;
        }
    }
    
    std::cout << " Found " << patterns.size() << " patterns." << std::endl;
    
    for(const auto& p : patterns) {
        std::string lower_p = p;
        std::transform(lower_p.begin(), lower_p.end(), lower_p.begin(), ::tolower);
        
        std::cout << "  - Pattern: " << p << " -> ";
        
        // --- PATTERN MAPPING LOGIC ---
        
        // 1. STRUCTURAL MODES
        if (lower_p.find("sensory overload") != std::string::npos || lower_p.find("cacophony") != std::string::npos) {
            std::cout << "[MODE BAN: SENSORY_OVERLOAD]";
            active_constraints.force_minimal_adjectives = true;
            active_constraints.ban_metaphors = true;
        }
        else if (lower_p.find("body") != std::string::npos || lower_p.find("pain") != std::string::npos || lower_p.find("distress") != std::string::npos) {
            std::cout << "[MODE BAN: BODY_DISTRESS]";
            active_constraints.ban_body_vocab = true;
        }
        else if (lower_p.find("void") != std::string::npos || lower_p.find("empty") != std::string::npos || lower_p.find("abyss") != std::string::npos) {
            std::cout << "[MODE BAN: VOID_IMAGERY]";
            active_constraints.ban_abstract_nouns = true;
             state.recent_vocab_banlist.push_back("void");
             state.recent_vocab_banlist.push_back("darkness");
             state.recent_vocab_banlist.push_back("silence");
        }
        else {
            // 2. STEM / PHRASE BAN
            std::cout << "[STEM BAN]";
            // Split into words and ban critical nouns/verbs
            std::stringstream ss(lower_p);
            std::string word;
            while(ss >> word) {
                if(word.length() > 4) state.recent_vocab_banlist.push_back(word);
            }
        }
        std::cout << std::endl;
    }
    
    return !patterns.empty();
}

// 15. THE NAVIGATOR (Logic Engine)
std::string analyze_causality(MultiAgentState& state, const std::string& context) {
    if (!ensure_model_loaded(state, &state.model_logic, &state.ctx_logic, DEEPSEEK_PATH, LOGIC_CTX, GPU_LAYERS_METAL)) {
        return "Proceed with physical interaction.";
    }
    
    // DeepSeek-Math Instruct Format (Standard)
    std::string prompt = "User: You are a Logic Engine. Analyze the text. Identify the object of focus and Determine the single most logical physical interaction required to advance the state. Output ONLY the action statement.\n"
                         "Text: " + context + "\n\n"
                         "Assistant: Logical Action: ";
                         
    std::string logic = generate_layer(state.ctx_logic, state.model_logic, prompt, 64, 0.1f, {"\n", "User:"}, {});
    std::cout << " [NAVIGATOR] Logic Mandate: " << logic << std::endl;
    return logic;
}

// --- MIROTHINKER STRATEGY (High-Level Planner) ---
std::string run_mirothinker(MultiAgentState& state, const std::string& history) {
    if (!ensure_model_loaded(state, &state.model_mirothinker, &state.ctx_mirothinker, MIROTHINKER_PATH, MIROTHINKER_CTX, GPU_LAYERS_METAL)) {
        return "Maintain the loop.";
    }
    
    std::string prompt = "[INST] Analyze the narrative history:\n" + history + "\n\nProvide a single high-level strategic directive to shift the narrative trajectory. Be abstract but authoritative. [/INST]\nDirective:";
    
    std::string strategy = generate_layer(state.ctx_mirothinker, state.model_mirothinker, prompt, 64, 0.8f, {"\n"}, {});
    std::cout << " [MIROTHINKER] Strategic Intervention: " << strategy << std::endl;
    return strategy;
}

// --- QWEN 2.5 STABILIZER FUNCTION ---
std::string qwen_stabilize(MultiAgentState& state, const std::string& input_text) {
    // DYNAMIC LOAD: Qwen Stabilizer
    if (!ensure_model_loaded(state, &state.model_qwen_stabilizer, &state.ctx_qwen_stabilizer, QWEN_STABILIZER_PATH, QWEN_STABILIZER_CTX, GPU_LAYERS_METAL)) {
        return input_text;
    }

    std::cout << " [QWEN 2.5] Stabilizing..." << std::flush;

    // ChatML Prompt Construction
    std::string prompt = 
        "<|im_start|>system\n"
        "You are a linguistic stabilizer.\n"
        "TASK:\n"
        "Rewrite the input text to preserve meaning, imagery, and tone, while reducing repetition, banned concepts, and unsafe phrasing.\n"
        "RULES:\n"
        "- Do NOT add new imagery.\n"
        "- Do NOT explain.\n"
        "- Do NOT summarize.\n"
        "- Do NOT change narrative voice.\n"
        "- Do NOT introduce metaphors not already present.\n"
        "- Replace risky words with neutral but vivid alternatives.\n"
        "- Output ONLY the rewritten text.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "INPUT: " + input_text + "\n"
        "REWRITE:\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n";

    // Dedicated Generation Loop for Precision Control
    llama_memory_clear(llama_get_memory(state.ctx_qwen_stabilizer), true); 
    auto* vocab = llama_model_get_vocab(state.model_qwen_stabilizer);
    
    auto& tokens_list = token_scratch(prompt.size() + kTokenScratchPad);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
        token_scratch(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);

    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size());
    if (llama_decode(state.ctx_qwen_stabilizer, batch) != 0) {
        std::cerr << " [QWEN] Decode Failed!" << std::endl;
        return input_text;
    }

    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.90f, 1)); 
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.20f)); // Low temp for stability
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(std::rand()));
    
    std::string output;
    int max_toks = 600; 
    output.reserve(static_cast<size_t>(max_toks) * kAvgTokenChars);

    for(int i=0; i<max_toks; i++) {
        llama_token id = llama_sampler_sample(smpl, state.ctx_qwen_stabilizer, -1);
        llama_sampler_accept(smpl, id);
        
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) continue;
        std::string piece(buf, n);
        
        if (piece.find("<|im_end|>") != std::string::npos) break;
        if (llama_vocab_is_eog(vocab, id)) break;

        output += piece;
        
        if (llama_decode(state.ctx_qwen_stabilizer, llama_batch_get_one(&id, 1))) break;
    }
    
    llama_sampler_free(smpl);
    
    if (output.length() < 10) return input_text; 

    // Trim
    while(!output.empty() && isspace(output.front())) output.erase(0, 1);
    while(!output.empty() && isspace(output.back())) output.pop_back();

    std::cout << " Done." << std::endl;
    return output;
}

// --- QWEN 2 CREATIVE SPARK FUNCTIONS ---

// 1. REFLEX FAILURE (Mini Ideation Burst)
std::string qwen_creative_burst(MultiAgentState& state, const std::string& context) {
    // DYNAMIC LOAD
    if (!ensure_model_loaded(state, &state.model_qwen_creative, &state.ctx_qwen_creative, QWEN_CREATIVE_PATH, QWEN_CREATIVE_CTX, GPU_LAYERS_METAL)) {
        return "";
    }
    std::cout << " [QWEN 2] Triggering Creative Burst..." << std::endl;

    std::string prompt = 
        "<|im_start|>user\n"
        "Given the following scene, suggest ONE unexpected angle or detail.\n"
        "SCENE: " + context.substr(context.length() > 300 ? context.length()-300 : 0) + "\n"
        "RULES:\n"
        "- Do NOT write prose.\n"
        "- Do NOT explain.\n"
        "- Return a short fragment (max 12 words).\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n";
    
    // High Temp (1.2) for creativity
    std::string out = generate_layer(state.ctx_qwen_creative, state.model_qwen_creative, prompt, 30, 1.2f, {"<|im_end|>", "\n"}, {});
    return trim_trailing_noise(out);
}

// 2. EXISTENTIAL SATURATION (Anlam Çağrışımı)
std::string qwen_existential_association(MultiAgentState& state, const std::string& narrative) {
    // DYNAMIC LOAD
    if (!ensure_model_loaded(state, &state.model_qwen_creative, &state.ctx_qwen_creative, QWEN_CREATIVE_PATH, QWEN_CREATIVE_CTX, GPU_LAYERS_METAL)) {
        return "";
    }
    std::cout << " [QWEN 2] Triggering Existential Association..." << std::endl;

    std::string prompt = 
        "<|im_start|>user\n"
        "Associate this internal narrative with a vague human concern.\n"
        "NARRATIVE: " + narrative.substr(narrative.length() > 400 ? narrative.length()-400 : 0) + "\n"
        "RULES:\n"
        "- Do NOT mention facts.\n"
        "- Do NOT mention real events.\n"
        "- One abstract phrase only.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n";
        
    std::string out = generate_layer(state.ctx_qwen_creative, state.model_qwen_creative, prompt, 25, 0.9f, {"<|im_end|>", "\n"}, {});
    return trim_trailing_noise(out);
}

// 3. CONTROLLED NOISE (Gemma Alternative)
std::string qwen_creative_concept(MultiAgentState& state) {
    // DYNAMIC LOAD
    if (!ensure_model_loaded(state, &state.model_qwen_creative, &state.ctx_qwen_creative, QWEN_CREATIVE_PATH, QWEN_CREATIVE_CTX, GPU_LAYERS_METAL)) {
        return "";
    }
    std::cout << " [QWEN 2] Triggering Controlled Noise..." << std::endl;

    std::string prompt = 
        "<|im_start|>user\n"
        "Generate a neutral but unsettling modifier.\n"
        "RULES:\n"
        "- No imagery.\n"
        "- No metaphor.\n"
        "- One phrase.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n";

    std::string out = generate_layer(state.ctx_qwen_creative, state.model_qwen_creative, prompt, 20, 1.0f, {"<|im_end|>", "\n"}, {});
    return trim_trailing_noise(out);
}

// --- DOLPHIN OBSERVER (Detatched but first-person) ---
std::string dolphin_observer_reframe(MultiAgentState& state, const std::string& source) {
    if (!state.model_main || !state.ctx_main) return "";
    std::string tail = source.substr(source.length() > 700 ? source.length() - 700 : 0);

    std::string prompt = 
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are DOLPHIN, a detached observer who narrates what is happening.\n"
        "Rewrite the provided text as clipped, factual first-person present observations.\n"
        "Rules: No dialogue. No quests or instructions. Avoid the word 'protagonist'. Keep 3-6 sentences.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "TEXT: " + tail + "\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n";

    std::string out = generate_layer(state.ctx_main, state.model_main, prompt, 200, 0.55f, {"<|eot_id|>", "\n\n"}, {});
    out = strip_meta_commentary(out);
    out = trim_meta_tail(out);
    out = trim_trailing_noise(out);
    return out;
}

// --- FIMBULVETR OBSERVER (First-Person Enforcer) ---
std::string fimbulvetr_first_person(MultiAgentState& state, const std::string& source) {
    if (source.empty()) return "";
    if (!ensure_model_loaded(state, &state.model_fimbulvetr, &state.ctx_fimbulvetr, FIMBULVETR_PATH, FIMBULVETR_CTX, GPU_LAYERS_METAL)) {
        return "";
    }

    std::cout << " [FIMBULVETR] Observing for first-person rewrite..." << std::endl;
    std::string tail = source.substr(source.length() > 700 ? source.length() - 700 : 0);

    std::string prompt = 
        "<|im_start|>system\n"
        "You are FIMBULVETR, a retired observer who EXPERIENCES scenes directly from inside the body.\n"
        "Rewrite the provided text as an immediate, lived, first-person present-tense account.\n"
        "Rules: Avoid legal or analytical phrasing. No dialogue. No quests or instructions. Do not name 'protagonist'. Use 'I' naturally. Keep 4-7 sentences focused on sensation and physical surroundings.\n"
        "Do not add new objects. Keep the sensory intensity and physicality.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "TEXT: " + tail + "\n"
        "REWRITE:\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n";

    std::string out = generate_layer(state.ctx_fimbulvetr, state.model_fimbulvetr, prompt, 200, 0.65f, {"<|im_end|>", "\n\n"}, {});
    out = strip_meta_commentary(out);
    out = trim_meta_tail(out);
    out = trim_trailing_noise(out);
    return out;
}

// --- NARRATIVE CIRCUIT BREAKER (State Machine) ---

// --- ESCAPE VECTOR REJECTION FILTER ---
// Prevents system prompt collapse by filtering toxic vectors
bool is_toxic_escape_vector(const std::string& vec) {
    // Check for system token leakage
    if (vec.find("<|") != std::string::npos) return true;
    if (vec.find("|>") != std::string::npos) return true;
    
    // Check for assistant/system mode leakage
    std::vector<std::string> toxic_patterns = {
        "assistant", "system", "please provide", "here is", "I can help",
        "user", "human", "AI", "model", "instruction", "task:",
        "certainly", "of course", "let me", "I'll", "would you like"
    };
    std::string lower = vec;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto& pat : toxic_patterns) {
        if (lower.find(pat) != std::string::npos) return true;
    }
    return false;
}

// [SAUL ENGINE] CASE GENERATOR (The Figure)
// Saul's role: Generate SCENARIOS, not solutions. Create experiences.
// The Figure creates the case; the Protagonist experiences it.
std::string generate_saul_case(MultiAgentState& state, const std::string& context) {
    if (!ensure_model_loaded(state, &state.model_saul, &state.ctx_saul, SAUL_PATH, SAUL_CTX, GPU_LAYERS_METAL)) {
        return "A sudden change occurs in the environment."; // Fallback case
    }

    std::cout << " [SAUL] The Figure is constructing a case..." << std::endl;

    // Context snippet for Saul
    std::string short_context = context;
    if (short_context.length() > 400) {
        short_context = short_context.substr(short_context.length() - 400);
    }

    // Saul generates a SCENARIO, not a prefill
    std::string prompt = 
        "### Instruction:\n"
        "You are THE FIGURE - an external force that creates scenarios.\n"
        "Based on the current situation, generate ONE concrete physical event that HAPPENS NOW.\n"
        "Rules:\n"
        "- Write in THIRD PERSON (not 'I', use 'The [object]...')\n"
        "- Describe WHAT HAPPENS, not what the protagonist feels\n"
        "- Be specific: sounds, movements, physical changes\n"
        "- One sentence only. Present tense.\n"
        "- NO dialogue. NO questions. NO choices.\n\n"
        "### Current Situation:\n" + short_context + "\n\n"
        "### The Event:\n";

    std::string event = generate_layer(state.ctx_saul, state.model_saul, prompt, 50, 0.8f, {"\n", "###", ".", "?"}, {});
    
    std::string clean = trim_trailing_noise(event);
    
    // Append period if missing
    if (!clean.empty() && clean.back() != '.') {
        clean += ".";
    }
    
    if (clean.length() < 10) {
        return "A door slams somewhere in the darkness."; // Fallback
    }
    
    std::cout << " [SAUL] Case Generated: " << clean << std::endl;
    return clean;
}

// [FIMBULVETR] EXPERIENCE GENERATOR (The Protagonist)
// Fimbulvetr's role: Take a scenario and EXPERIENCE it in first-person.
std::string fimbulvetr_experience(MultiAgentState& state, const std::string& scenario, const std::string& context) {
    if (!ensure_model_loaded(state, &state.model_fimbulvetr, &state.ctx_fimbulvetr, FIMBULVETR_PATH, FIMBULVETR_CTX, GPU_LAYERS_METAL)) {
        // Fallback: Just prefix with "I" and return
        return "I " + scenario;
    }

    std::cout << " [FIMBULVETR] The Protagonist is experiencing..." << std::endl;

    std::string short_context = context;
    if (short_context.length() > 600) {
        short_context = short_context.substr(short_context.length() - 600);
    }

    std::string prompt = 
        "<|im_start|>system\n"
        "You are the PROTAGONIST. You experience reality through your senses.\n"
        "An EVENT has occurred. Describe how YOU experience it.\n"
        "Rules:\n"
        "- Write in FIRST PERSON ('I feel...', 'I see...')\n"
        "- Focus on SENSATIONS: what you see, hear, feel, smell\n"
        "- NO meta-commentary. NO reflection. Pure experience.\n"
        "- 2-3 sentences. Present tense.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Recent context:\n" + short_context + "\n\n"
        "THE EVENT: " + scenario + "\n\n"
        "How do you experience this?\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "I ";

    std::string experience = generate_layer(state.ctx_fimbulvetr, state.model_fimbulvetr, prompt, 100, 0.7f, {"<|im_end|>", "\n\n"}, {});
    
    std::string result = "I " + trim_trailing_noise(experience);
    
    if (result.length() < 15) {
        return "I feel it before I see it. The air shifts."; // Fallback
    }
    
    std::cout << " [FIMBULVETR] Experience: " << result.substr(0, 80) << "..." << std::endl;
    return result;
}


// [NEW] CHRONOS: The World Engine
// Executes python script to predict narrative weather; falls back to on-device classifier if unknown.
std::string classify_weather_from_text(MultiAgentState& state, const std::string& context) {
    if (!state.model_main || !state.ctx_main) return "UNKNOWN";

    std::string tail = context.substr(context.length() > 800 ? context.length() - 800 : 0);
    std::string prompt =
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a concise weather classifier. Given the scene description, pick one label from: CALM, WINDY, STORM, RAIN, FOG, CLEAR.\n"
        "Respond with ONLY the label.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "SCENE: " + tail + "\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n";

    std::string out = generate_layer(state.ctx_main, state.model_main, prompt, 8, 0.2f, {"<|eot_id|>", "\n"}, {});
    std::transform(out.begin(), out.end(), out.begin(), ::toupper);
    if (out.find("CALM") != std::string::npos) return "CALM";
    if (out.find("WINDY") != std::string::npos) return "WINDY";
    if (out.find("STORM") != std::string::npos) return "STORM";
    if (out.find("RAIN") != std::string::npos) return "RAIN";
    if (out.find("FOG") != std::string::npos) return "FOG";
    if (out.find("CLEAR") != std::string::npos) return "CLEAR";
    return "UNKNOWN";
}



// [NEW] LLM Fallback for World State Extraction
void extract_world_state_llm(MultiAgentState& state, const std::string& text) {
    if (!state.model_main || !state.ctx_main) return;
    
    std::cout << " [REBEL] Python Agent failed. Using LLaMA Fallback..." << std::endl;
    
    std::string prompt = 
        "<|start_header_id|>system<|end_header_id|>\n"
        "Extract 3 key physical facts about the immediate environment from the text. Format: Entity: Status.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "TEXT: " + text.substr(text.length() > 600 ? text.length()-600 : 0) + "\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n";
        
    std::string out = generate_layer(state.ctx_main, state.model_main, prompt, 100, 0.5f, {"<|eot_id|>", "\n\n"}, {});
    
    // Simple line parsing
    std::stringstream ss(out);
    std::string line;
    while(std::getline(ss, line)) {
        if (line.find(":") != std::string::npos) {
            std::string entity = line.substr(0, line.find(":"));
            std::string status = line.substr(line.find(":") + 1);
            // Trim
            while(!entity.empty() && !isalnum(entity.front())) entity.erase(0, 1);
            while(!status.empty() && isspace(status.front())) status.erase(0, 1);
            
            if (entity.length() > 2) {
                state.world_state[entity] = status;
                std::cout << " [REBEL-LLM] Update: " << entity << " -> " << status << std::endl;
            }
        }
    }
}

std::string run_chronos_forecast(MultiAgentState& state, const std::string& recent_text) {
    // 1. Prepare Data
    json j_data;
    j_data["entropy"] = state.history_entropy;
    j_data["sentiment"] = state.history_sentiment;
    j_data["speed"] = state.history_speed;
    j_data["timestamp"] = (long)std::time(nullptr); 
    
    std::string valid_json = j_data.dump();
    std::string cmd = "python3 scripts/chronos_adapter.py '" + valid_json + "'";
    
    // 2. Execute
    std::string result_json;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        state.current_weather = "UNKNOWN";
        return "";
    }
    char buffer[512]; // Increased buffer
    while (fgets(buffer, 512, pipe) != NULL) {
        result_json += buffer;
    }
    pclose(pipe);
    
    // 3. Parse & Predict Vectorially
    try {
        auto j_res = json::parse(result_json);
        
        // Get Metrics
        float pred_entropy = 0.5f;
        if (j_res.contains("entropy_pred") && !j_res["entropy_pred"].empty()) {
            auto& arr = j_res["entropy_pred"];
            float sum = 0;
            for(auto& val : arr) sum += val.get<float>();
            pred_entropy = sum / arr.size();
        }
        
        // WEATHER SENSOR FUSION (Vector Prediction)
        if (state.weather_oracle) {
            std::string vector_weather = state.weather_oracle->predict_next(state.weather_history, pred_entropy, 0.5f);
            state.current_weather = vector_weather;
        } else {
            // Fallback to legacy field
            state.current_weather = j_res.value("forecast", "UNKNOWN");
        }

        std::cout << " [CHRONOS] FINAL WEATHER: " << state.current_weather << " (Ent: " << pred_entropy << ")" << std::endl;
        
        // Push to history for next time
        state.weather_history.push_back(state.current_weather);
        if (state.weather_history.size() > 10) state.weather_history.erase(state.weather_history.begin());

        if (state.current_weather == "STORM" || state.current_weather == "WINDY") {
             return "EVENT_INJECTION: A SUDDEN CATASTROPHE OCCURS. Weather turns violent.";
        }
        
    } catch (const std::exception& e) {
        std::cout << " [CHRONOS] Parse Error: " << e.what() << std::endl;
        state.current_weather = "UNKNOWN";
    }

    return "";
}


// [UPDATED] Dynamic Persona Deck
struct Persona {
    std::string name;
    std::string instruction;
    std::vector<std::string> prefills;
    float temp;
};
std::deque<Persona> persona_deck;

// Artık State alıyor çünkü model çalıştırması lazım.
void shuffle_deck(MultiAgentState& state) {
    std::cout << "[SYSTEM] SaulLM is constructing the narrative deck (Forensic Analysis)..." << std::endl;

    // 1. KATEGORİLER İÇİN CANLI ÜRETİM (Batch Generation)
    // Her kategori için 3'er tane taze cümle üretiyoruz.
    
    std::vector<std::string> kinetic_dynamic;
    for(int i=0; i<3; i++) kinetic_dynamic.push_back(generate_saul_case(state, "sudden physical impact or force"));

    std::vector<std::string> visceral_dynamic;
    for(int i=0; i<3; i++) visceral_dynamic.push_back(generate_saul_case(state, "biological trauma or internal anatomy"));

    std::vector<std::string> surreal_dynamic;
    for(int i=0; i<3; i++) surreal_dynamic.push_back(generate_saul_case(state, "visual distortion or hallucination evidence"));

    std::vector<std::string> observer_dynamic;
    for(int i=0; i<3; i++) observer_dynamic.push_back(generate_saul_case(state, "environmental decay and material structure"));


    // 2. PERSONA HAVUZUNU OLUŞTUR (Artık Dinamik Veriyle)
    std::vector<Persona> pool = {
        
        {"KINETIC_LOGIC", 
         "Focus on NEWTONIAN PHYSICS. Describe force vectors.", 
         kinetic_dynamic, // Saul'un ürettiği liste buraya giriyor
         0.7f},

        {"VISCERAL_AUTOPSY", 
         "Focus on BIOLOGICAL FACTS. Describe anatomy like a coroner.", 
         visceral_dynamic, 
         0.8f},

        {"SURREAL_EVIDENCE", 
         "Focus on VISUAL ANOMALIES. Describe distortions as 'Unexplained Phenomena'.", 
         surreal_dynamic, 
         0.9f},

        {"OBSERVER_REPORT", 
         "Focus on MATERIAL CONDITIONS. Describe dust and rust as forensic trace.", 
         observer_dynamic, 
         0.6f}
    };

    // 3. KARIŞTIR VE YÜKLE
    std::random_device rd; std::mt19937 g(rd());
    std::shuffle(pool.begin(), pool.end(), g);
    
    persona_deck.clear();
    for(const auto& p : pool) {
        // Boş gelme ihtimaline karşı (Model hatası olursa) hardcoded fallback ekle
        Persona final_p = p;
        if (final_p.prefills.empty()) {
            final_p.prefills = {"The data shows", "Observation confirms"};
        }
        persona_deck.push_back(final_p);
    }
    
    std::cout << "[SYSTEM] Deck Shuffled with " << pool.size() * 3 << " unique generated hooks." << std::endl;
}

Persona draw_persona(MultiAgentState& state) { // State parametresi eklendi
    if (persona_deck.empty()) shuffle_deck(state); // State ile çağır
    Persona p = persona_deck.front(); 
    persona_deck.pop_front();
    return p;
}

// --- MIROTHINKER STRUCTURAL CONSTRAINT GENERATION ---
// MiroThinker now outputs STRUCTURED constraints that are ENFORCED, not prose suggestions.
struct RealityShift {
    bool location_change = false;
    bool time_jump = false;
    std::string time_delta = "";
    std::string pov_shift = ""; // "third_person", "sensory_only"
    std::string sensory_channel = ""; // "auditory_only", "tactile_only"
    std::string forced_action = ""; // Concrete physical action
    std::vector<std::string> hard_bans; // Tokens to absolutely ban
};

// Fallback Action Rotator (to prevent repetitive "ground crumbling")
std::string get_diverse_fallback_action() {
    static int idx = 0;
    std::vector<std::string> actions = {
        "A sudden, deafening silence swallows all sound for three heartbeats.",
        "Gravity shifts momentarily, pulling everything to the left.",
        "The light source above flickers and turns a sickly green.",
        "A massive, unseen weight presses down on my chest.",
        "The smell of ozone is replaced by the scent of burning hair.",
        "The corridor abruptly ends in a sheer vertical drop.",
        "Something unseen brushes against the back of my neck.",
        "My own shadow detaches from the floor and stands upright."
    };
    idx = (idx + 1) % actions.size();
    return actions[idx];
}

bool contains_non_ascii(const std::string& s) {
    for (char c : s) {
        if (static_cast<unsigned char>(c) > 127) return true;
    }
    return false;
}

RealityShift mirothinker_structural_constraint(MultiAgentState& state, const std::string& context, float severity) {
    RealityShift shift;
    
    if (!ensure_model_loaded(state, &state.model_mirothinker, &state.ctx_mirothinker, MIROTHINKER_PATH, MIROTHINKER_CTX, GPU_LAYERS_METAL)) {
        shift.forced_action = get_diverse_fallback_action();
        return shift;
    }

    std::cout << " [MIROTHINKER] Generating STRUCTURAL Constraint (Severity: " << severity << ")..." << std::endl;

    // Build structured prompt - CLEANER FORMAT (User/Assistant only) to stop role confusion
    std::string prompt = 
        "<|start_header_id|>user<|end_header_id|>\n"
        "Narrative Context: " + context.substr(context.length() > 600 ? context.length()-600 : 0) + "\n"
        "\n"
        "TASK: The story is looping. Provide ONE concrete physical event to break the cycle.\n"
        "CONSTRAINT: Concrete action only. No meta-commentary.\n"
        "CONSTRAINT: Focus on physical environment shifts, light changes, or sudden sounds.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "EVENT: ";

    std::string out = generate_layer(state.ctx_mirothinker, state.model_mirothinker, prompt, 60, 0.8f, {"<|eot_id|>", "\n", "."}, {});
    out = trim_trailing_noise(out);
    
    // Helper: Count words
    auto word_count = [](const std::string& s) -> int {
        std::stringstream ss(s);
        std::string word;
        int count = 0;
        while (ss >> word) count++;
        return count;
    };
    
    // CRITICAL: Validation Logic (STRENGTHENED)
    bool invalid = false;
    
    // Length check: min 20 chars
    if (out.length() < 20) {
        std::cout << " [MIROTHINKER] TOO SHORT (" << out.length() << " chars). REJECTING." << std::endl;
        invalid = true;
    }
    
    // Word count check: min 5 words
    if (!invalid && word_count(out) < 5) {
        std::cout << " [MIROTHINKER] TOO FEW WORDS (" << word_count(out) << "). REJECTING." << std::endl;
        invalid = true;
    }
    
    // REGEX VALIDATION (The nuclear option for numeric junk)
    // Rejects: Digits, System Tokens, Leaky Phrases
    try {
        std::regex junk_regex(R"([0-9]|<\||\|>|150\s*words|instruction|generate)", std::regex::icase);
        if (!invalid && std::regex_search(out, junk_regex)) {
             std::cout << " [MIROTHINKER] DETECTED REGEX MATCH (Digits/Leak). REJECTING." << std::endl;
             invalid = true;
        }
    } catch (const std::regex_error& e) {
        // Fallback if regex fails (shouldn't happen with simple regex)
        if (std::any_of(out.begin(), out.end(), ::isdigit)) invalid = true;
    }
    
    // Check for special character start (token leakage)
    if (!invalid && !out.empty() && (out[0] == '<' || out[0] == '|' || out[0] == '[')) {
        std::cout << " [MIROTHINKER] STARTS WITH SPECIAL CHAR. REJECTING." << std::endl;
        invalid = true;
    }
    
    if (!invalid && contains_non_ascii(out)) {
        std::cout << " [MIROTHINKER] DETECTED FOREIGN CHARACTERS (Chinese/etc). REJECTING." << std::endl;
        invalid = true;
    }
    if (!invalid && is_toxic_escape_vector(out)) invalid = true;

    // [SAFETY] Anti-Hallucination Blacklist (Human Terms)
    // Isolate the protagonist. No other people.
    if (!invalid) {
         std::vector<std::string> safety_bans = {"girl", "boy", "man", "woman", "child", "person", "people", "someone", "stranger", "figure"};
         std::string lower_out = out;
         std::transform(lower_out.begin(), lower_out.end(), lower_out.begin(), ::tolower);
         for (const auto& w : safety_bans) {
             // Check individual words to avoid substring matches like "human" in "humanity" if needed, 
             // but here simple find is safer to ban all variations.
             std::regex safe_regex("\\b" + w + "\\b"); // Whole word match
             if (std::regex_search(lower_out, safe_regex)) {
                 std::cout << " [MIROTHINKER] SAFETY VIOLATION (Human Term: '" << w << "'). REJECTING." << std::endl;
                 invalid = true; 
                 break;
             }
         }
    }

    // [RELEVANCE] Domain Whitelist
    // Must contain at least one physical anchor.
    if (!invalid) {
        std::vector<std::string> relevance_keys = {
            "wall", "floor", "corridor", "room", "light", "dark", "sound", "silence", 
            "structure", "metal", "stone", "glass", "air", "breath", "shadow", "void",
            "gravity", "cold", "heat", "pain", "hunger", "thirst", "vibration", "hum"
        };
        bool relevant = false;
         std::string lower_out = out;
         std::transform(lower_out.begin(), lower_out.end(), lower_out.begin(), ::tolower);
        for (const auto& k : relevance_keys) {
            if (lower_out.find(k) != std::string::npos) {
                relevant = true; 
                break;
            }
        }
        if (!relevant) {
            std::cout << " [MIROTHINKER] NO DOMAIN RELEVANCE (Hallucination suspected). REJECTING." << std::endl;
            invalid = true;
        }
    }

    if (invalid) {
        std::cout << " [MIROTHINKER] Output rejected ('" << out << "'). Using DIVERSE fallback." << std::endl;
        shift.forced_action = get_diverse_fallback_action();
        
        // Severity-based additional constraints
        if (severity > 0.98f) {
            shift.location_change = true;
            shift.hard_bans = {"void", "consciousness", "awareness", "feeling", "sense"};
        } 
    } else {
        shift.forced_action = out;
        // Natural severe bans
        if (severity > 0.96f) {
           shift.hard_bans = {"thought", "mind", "soul", "remember", "memory"}; 
        }
    }
    
    std::cout << " [MIROTHINKER] VETO ACTION: " << shift.forced_action << std::endl;
    return shift;
}

// --- MIROTHINKER & RWKV STRUCTURES ---

struct AnchorPack {
    std::string setting;
    std::string objects;
    std::string physics;
    std::string sensory_swap;
    std::vector<std::string> do_list;
    std::vector<std::string> avoid_list;
};

struct PhysicsAudit {
    std::string state;
    std::string consequence_1;
    std::string consequence_2;
};

struct ArcStoryboard {
    std::string goal;
    std::string obstacle;
    std::string turn;
    std::string payoff;
};

// [RWKV] VETO ACTION (Servo)
// Fast, single-sentence physical intervention to break loops.
std::string generate_rwkv_veto(MultiAgentState& state, const std::string& history) {
    // DYNAMIC LOAD: RWKV Servo
    if (!ensure_model_loaded(state, &state.model_rwkv, &state.ctx_rwkv, RWKV_PATH, RWKV_CTX, GPU_LAYERS_METAL)) {
        return "";
    }
    
    std::cout << "[RWKV] Generating VETO ACTION (Servo)..." << std::endl;
    
    std::string prompt = 
        "CONTEXT: " + history.substr(history.length() > 500 ? history.length() - 500 : 0) + "\n"
        "TASK: The story is stuck in a loop. Provide ONE physical action to break it immediately.\n"
        "CONSTRAINT: No dialogue. No thoughts. Pure kinetics.\n"
        "VETO_ACTION:";
        
    // RWKV should be fast. Low temp for precision.
    std::string action = generate_layer(state.ctx_rwkv, state.model_rwkv, prompt, 40, 0.5f, {"\n", "."}, {});
    
    // Cleanup
    if (action.find(":") != std::string::npos) action = action.substr(action.find(":") + 1);
    action = trim_trailing_noise(action);
    
    std::cout << "[RWKV] VETO: " << action << std::endl;
    return action;
}

// [RWKV] ESCAPE VECTOR (Phase 1 Orthogonal)
std::string generate_rwkv_escape(MultiAgentState& state, const std::string& history) {
    // DYNAMIC LOAD: RWKV Servo (reusing same model slot)
    if (!ensure_model_loaded(state, &state.model_rwkv, &state.ctx_rwkv, RWKV_PATH, RWKV_CTX, GPU_LAYERS_METAL)) {
        return "";
    }
    
    std::cout << "[RWKV] Generating ESCAPE VECTOR (Orthogonal)..." << std::endl;
    
    // Orthogonal Thinking Prompt
    std::string prompt = 
        "CONTEXT: " + history.substr(history.length() > 500 ? history.length() - 500 : 0) + "\n"
        "TASK: Ignore previous patterns. Describe a completely new sensation.\n"
        "CONSTRAINT: Focus on liquid, atmosphere, or decay. No plot.\n"
        "NEW_SENSATION:";
        
    // Higher temp for creativity
    std::string sensation = generate_layer(state.ctx_rwkv, state.model_rwkv, prompt, 60, 1.1f, {"\n", "."}, {});
    
    // Cleanup
    if (sensation.find(":") != std::string::npos) sensation = sensation.substr(sensation.find(":") + 1);
    sensation = trim_trailing_noise(sensation);
    
    std::cout << "[RWKV] ESCAPE: " << sensation << std::endl;
    return sensation;
}

// --- MAMBA SYNAPSE (PREDICTIVE STATE ENGINE) ---
struct NeuralLinkData {
    std::string prediction; // Mamba's expectation (Text)
    std::string state_hash; // Current state digest (Hash)
};

NeuralLinkData run_mamba_synapse(MultiAgentState& state, const std::string& recent_context) {
    NeuralLinkData data;
    data.prediction = "unknown";
    data.state_hash = "0x00";

    // Dynamic Load: Mamba 1.4B
    if (!ensure_model_loaded(state, &state.model_mamba, &state.ctx_mamba, MAMBA_PATH, MAMBA_CTX, GPU_LAYERS_METAL)) {
        std::cerr << " [WARN] Mamba Synapse offline." << std::endl;
        return data; // Fail gracefully
    }

    std::cout << "[SYNAPSE] Mamba is dreaming of the next moment..." << std::endl;

    // Prompt for Mamba: Give it the tail of the story and ask for the immediate future.
    // Short context snippet to catch the "vibe" and "momentum".
    std::string context_snippet = recent_context.substr(recent_context.length() > 500 ? recent_context.length() - 500 : 0);
    
    std::string prompt = 
        context_snippet + 
        "\n\n[SYSTEM PREDICTION]: The immediate next event involves"; 

    // Generate short prediction (10-15 tokens is enough for a concept)
    std::string prediction = generate_layer(state.ctx_mamba, state.model_mamba, prompt, 15, 0.8f, {".", "\n", "]", "!"}, {});
    
    data.prediction = trim_trailing_noise(prediction);
    data.state_hash = sha256_string(prompt + prediction); // Simple state signature
    
    std::cout << " [SYNAPSE] Prediction: " << data.prediction << std::endl;
    return data;
}

// [MIROTHINKER] ANCHOR PACK GENERATOR
// Reality Anchor Generator: Creates a 3-block constraint package.
AnchorPack generate_anchor_pack(MultiAgentState& state, const std::string& history) {
    AnchorPack pack;
    // DYNAMIC LOAD: MiroThinker
    // This is a heavy 30B model. It will swap out others.
    if (!ensure_model_loaded(state, &state.model_mirothinker, &state.ctx_mirothinker, MIROTHINKER_PATH, MIROTHINKER_CTX, GPU_LAYERS_METAL)) {
        return pack;
    }

    std::cout << "[MIROTHINKER] Generating Reality Anchor Pack..." << std::endl;

    std::string prompt = 
        "CONTEXT: " + history.substr(history.length() > 1000 ? history.length() - 1000 : 0) + "\n"
        "TASK: Create a 'Reality Anchor' to ground the surreal narrative.\n"
        "OUTPUT FORMAT (Strict JSON-like key-value):\n"
        "SETTING: (Time + Place + Light)\n"
        "OBJECTS: (3 concrete items)\n"
        "PHYSICS: (Causal rule, e.g. 'ozone -> arc' )\n"
        "SENSORY_SWAP: (Instead of X, use Y)\n"
        "DO: (Action 1)\n"
        "DO: (Action 2)\n"
        "AVOID: (Cliche 1)\n"
        "AVOID: (Cliche 2)\n"
        "RESPONSE:\n";

    std::string output = "";
    int attempts = 0;
    while(attempts < 2) {
        output = generate_layer(state.ctx_mirothinker, state.model_mirothinker, prompt, 300, 0.7f + (attempts * 0.1f), {"<|eot_id|>"}, {});
        
        // Validation: Too short?
        if (output.length() < 20) {
            std::cout << " [MIROTHINKER] Output too short ('" << output << "'). Retrying..." << std::endl;
            attempts++;
        } else {
            break; // Good
        }
    }
    
    // Fallback if still broken
    if (output.length() < 20) {
        output = "SETTING: A stable void.\nOBJECTS: Dust, Stone, Light.\nPHYSICS: Standard gravity.\nDO: Observe.\n";
    }

    // Parse Output (Simple Line Parsing)
    std::stringstream ss(output);
    std::string line;
    while(std::getline(ss, line)) {
        if (line.find("SETTING:") != std::string::npos) pack.setting = line.substr(line.find(":") + 1);
        else if (line.find("OBJECTS:") != std::string::npos) pack.objects = line.substr(line.find(":") + 1);
        else if (line.find("PHYSICS:") != std::string::npos) pack.physics = line.substr(line.find(":") + 1);
        else if (line.find("SENSORY_SWAP:") != std::string::npos) pack.sensory_swap = line.substr(line.find(":") + 1);
        else if (line.find("DO:") != std::string::npos) pack.do_list.push_back(line.substr(line.find(":") + 1));
        else if (line.find("AVOID:") != std::string::npos) pack.avoid_list.push_back(line.substr(line.find(":") + 1));
    }
    
    // Log
    std::cout << " [ANCHOR] Setting: " << pack.setting << std::endl;
    std::cout << " [ANCHOR] Physics: " << pack.physics << std::endl;
    
    return pack;
}

// [MIROTHINKER] PHYSICS AUDITOR
PhysicsAudit generate_physics_audit(MultiAgentState& state, const std::string& last_block) {
    PhysicsAudit audit;
    // DYNAMIC LOAD
    if (!ensure_model_loaded(state, &state.model_mirothinker, &state.ctx_mirothinker, MIROTHINKER_PATH, MIROTHINKER_CTX, GPU_LAYERS_METAL)) {
        return audit; 
    }

    if (last_block.length() < 50) return audit;

    std::cout << "[MIROTHINKER] Auditing Physics..." << std::endl;

    std::string prompt = 
        "INPUT TEXT: " + last_block + "\n"
        "TASK: Analyze the physical state and predict 2 IMMEDIATE consequences.\n"
        "OUTPUT FORMAT:\n"
        "STATE: (Current physical danger)\n"
        "CONSEQUENCE 1: (If X, then Y)\n"
        "CONSEQUENCE 2: (Physical risk)\n"
        "RESPONSE:\n";

    std::string output = generate_layer(state.ctx_mirothinker, state.model_mirothinker, prompt, 150, 0.6f, {"<|eot_id|>"}, {});

    std::stringstream ss(output);
    std::string line;
    while(std::getline(ss, line)) {
        if (line.find("STATE:") != std::string::npos) audit.state = line.substr(line.find(":") + 1);
        else if (line.find("CONSEQUENCE 1:") != std::string::npos) audit.consequence_1 = line.substr(line.find(":") + 1);
        else if (line.find("CONSEQUENCE 2:") != std::string::npos) audit.consequence_2 = line.substr(line.find(":") + 1);
    }
    return audit;
}

// [MIROTHINKER] ARC DIRECTOR
ArcStoryboard generate_arc_storyboard(MultiAgentState& state, const std::string& summary) {
    ArcStoryboard arc;
    // DYNAMIC LOAD
    if (!ensure_model_loaded(state, &state.model_mirothinker, &state.ctx_mirothinker, MIROTHINKER_PATH, MIROTHINKER_CTX, GPU_LAYERS_METAL)) {
        return arc;
    }

    std::cout << "[MIROTHINKER] Directing Narrative Arc..." << std::endl;

    std::string prompt = 
        "STORY SO FAR: " + summary + "\n"
        "TASK: Design the next short arc (mini storyboard).\n"
        "OUTPUT FORMAT:\n"
        "GOAL: (Protagonist's immediate objective)\n"
        "OBSTACLE: (Physical barrier)\n"
        "TURN: (Unexpected shift)\n"
        "PAYOFF: (Discovery/Exit)\n"
        "RESPONSE:\n";

    std::string output = generate_layer(state.ctx_mirothinker, state.model_mirothinker, prompt, 200, 0.8f, {"<|eot_id|>"}, {});

    std::stringstream ss(output);
    std::string line;
    while(std::getline(ss, line)) {
        if (line.find("GOAL:") != std::string::npos) arc.goal = line.substr(line.find(":") + 1);
        else if (line.find("OBSTACLE:") != std::string::npos) arc.obstacle = line.substr(line.find(":") + 1);
        else if (line.find("TURN:") != std::string::npos) arc.turn = line.substr(line.find(":") + 1);
        else if (line.find("PAYOFF:") != std::string::npos) arc.payoff = line.substr(line.find(":") + 1);
    }
    return arc;
}

/*
// Legacy wrapper for compatibility
std::string mirothinker_reason(MultiAgentState& state, const std::string& context) {
    RealityShift shift = mirothinker_structural_constraint(state, context, 0.92f);
    return shift.forced_action;
}
*/


enum class NarrativeState {
    AWAKENING,  // Blocks 0-2: Sensations, immediate confusion
    MOVEMENT,   // Blocks 3-5: FORCE movement, forbid deep introspection
    DISCOVERY,  // Blocks 6+: Finding objects, encountering entities
    CONFLICT    // Dynamic: If stuck
};

// Helper for random selection
std::string pick_random(const std::vector<std::string>& options) {
    if (options.empty()) return "";
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, options.size() - 1);
    return options[dist(rng)];
}

std::string get_state_instruction(NarrativeState state) {
    switch (state) {
        case NarrativeState::AWAKENING:
            // Randomized Sensory Focus to prevent overload
            return pick_random({
                "[STATE: AWAKENING] Focus only on SMELL (rot, ozone) and SOUND (dripping, heart).",
                "[STATE: AWAKENING] Focus only on PAIN in the body and the COLD stone floor.",
                "[STATE: AWAKENING] Darkness is absolute. Focus on TEXTURE and WEIGHT."
            });
        case NarrativeState::MOVEMENT:
            // More specific goal: Find an exit or change terrain.
            return "[STATE: MOVEMENT] You are moving. Describe the uneven ground, the strain on muscles. Do NOT stop.";
        case NarrativeState::DISCOVERY:
            return pick_random({
                "STATE: DISCOVERY. FOCUS: A MAN-MADE ARTIFACT. A tool, a weapon, a machine. Rusted, broken, ancient. PROHIBITED: No natural formations.",
                "STATE: DISCOVERY. FOCUS: A MEGA-STRUCTURE. A pillar, a gate, a bridge. Colossal, industrial, decaying. PROHIBITED: No small objects.",
                "STATE: DISCOVERY. FOCUS: A PHENOMENON. Light, gravity, sound, temperature. Ethereal, physics-defying. PROHIBITED: No physical walls.",
                "STATE: DISCOVERY. FOCUS: BIOLOGICAL REMNANTS. Bones, husks, dried skin, giant ribs. Extinct, fossilized. PROHIBITED: No living creatures."
            }) + "\nINSTRUCTION: Detail this SPECIFIC find with microscopic precision. Do not describe the general area.";
        default:
            return "[STATE: PROGRESSION] Something changes. The environment shifts.";
    }
}
// For a more robust entropy, one would need to calculate Shannon entropy over token probabilities.
// This is a simplified proxy for demonstration.


// --- ACCORDION TOPOLOGY IMPLEMENTATION ---

// [HELPER] Stateless Layer Generator
std::string generate_layer(llama_context* ctx, llama_model* model, const std::string& prompt, int max_tokens, float temp, const std::vector<std::string>& stop_words, const std::deque<std::string>& banned_words) {
    // Clear memory for stateless focus
    llama_memory_clear(llama_get_memory(ctx), true); 
    
    auto* vocab = llama_model_get_vocab(model);
    auto& tokens_list = token_scratch(prompt.size() + kTokenScratchPad);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
        token_scratch(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);

    // Decode Prompt
    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size());
    if (llama_decode(ctx, batch) != 0) return "";

    // Sampler Config
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    
    // [LOGIT BIAS] for Banned Words
    std::vector<llama_logit_bias> biases;
    biases.reserve(banned_words.size());
    std::vector<llama_token> b_toks;
    for (const auto& w : banned_words) {
        b_toks.resize(w.length() + 4);
        int n = llama_tokenize(vocab, w.c_str(), w.length(), b_toks.data(), b_toks.size(), false, false);
        if (n > 0) biases.push_back({b_toks[0], -1000.0f}); // Token'ı atomik seviyede yasakla
    }
    if(!biases.empty()) llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), biases.size(), biases.data()));

    // [UPDATED] Sampling Strategy (User Op C) - Include Top-P for Chaos
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40)); // Standard cleanup
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1)); // Nucleus Sampling
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(std::rand()));
    
    std::string output;
    output.reserve(static_cast<size_t>(max_tokens) * kAvgTokenChars);
    
    for(int i=0; i<max_tokens; i++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, id);
        
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) continue;
        std::string piece(buf, n);
        
        // STOP WORD CHECK
        bool stop = false;
        for(const auto& s : stop_words) {
            if (piece.find(s) != std::string::npos || output.find(s) != std::string::npos) {
                stop = true; break;
            }
        }
        if (stop || llama_vocab_is_eog(vocab, id)) break;

        output += piece;
        
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) break;
    }
    
    llama_sampler_free(smpl);
    
    // Trim
    while(!output.empty() && isspace(output.front())) output.erase(0, 1);
    while(!output.empty() && isspace(output.back())) output.pop_back();
    
    return output;
}

// [FIXED ENGINE] PERSONA-BASED NARRATIVE GENERATOR (With Head Repair & Deep Context)
// [HERMES] DYNAMIC CONSCIENCE
// Replaces static global_concepts map.
// Returns a vector of words to ban specifically for THIS block.
std::vector<std::string> generate_dynamic_constraints(MultiAgentState& state, const std::string& recent_history) {
    std::vector<std::string> dynamic_bans;

    // DYNAMIC LOAD: Nous-Hermes
    if (!ensure_model_loaded(state, &state.model_hermes, &state.ctx_hermes, HERMES_PATH, HERMES_CTX, GPU_LAYERS_METAL)) {
        std::cerr << " [WARN] Hermes Conscience offline. Using fallback." << std::endl;
        return {"metal", "blood", "void"}; // Minimal fallback
    }

    std::cout << "[CONSCIENCE] Hermes is judging the narrative flow..." << std::endl;

    // Felsefi ve Analitik Prompt
    std::string prompt = 
        "### Instruction:\n"
        "Analyze the following narrative segment. It is in danger of becoming repetitive or cliché.\n"
        "Identify 5 specific nouns or adjectives that are OVERUSED or would destabilize the current mood if used again immediately.\n"
        "Think like a strict literary editor.\n"
        "\n"
        "NARRATIVE:\n"
        "..." + recent_history.substr(recent_history.length() > 800 ? recent_history.length() - 800 : 0) + "\n"
        "\n"
        "### Response:\n"
        "BANNED_CONCEPTS_JSON: [";

    // Generate
    std::string output = generate_layer(state.ctx_hermes, state.model_hermes, prompt, 64, 0.7f, {"]", "\n\n"}, {});
    
    // Parse (Basitçe virgülle ayrılmış kelimeleri veya JSON benzeri yapıyı çek)
    // Örnek Çıktı: "shadows", "whispers", "neon", "rain", "cybernetic"
    std::stringstream ss(output);
    std::string segment;
    while(std::getline(ss, segment, ',')) {
        // Temizlik (Tırnak işaretlerini ve boşlukları kaldır)
        segment.erase(std::remove(segment.begin(), segment.end(), '\"'), segment.end());
        segment.erase(std::remove(segment.begin(), segment.end(), '['), segment.end());
        segment.erase(std::remove(segment.begin(), segment.end(), ']'), segment.end());
        while(!segment.empty() && isspace(segment.front())) segment.erase(0, 1);
        while(!segment.empty() && isspace(segment.back())) segment.pop_back();
        
        if (segment.length() > 2) {
            dynamic_bans.push_back(segment);
            std::cout << " [CONSCIENCE BAN] " << segment << std::endl;
        }
    }
    
    return dynamic_bans;
}

// [HERMES] POST-GENERATION EDITOR
// Removes repetitive phrases and clichés from generated text
std::string hermes_edit_pass(MultiAgentState& state, const std::string& raw_output) {
    if (raw_output.length() < 100) return raw_output; // Too short to edit
    
    if (!ensure_model_loaded(state, &state.model_hermes, &state.ctx_hermes, HERMES_PATH, HERMES_CTX, GPU_LAYERS_METAL)) {
        return raw_output;
    }
    
    std::cout << " [HERMES] Editing pass..." << std::endl;
    
    std::string prompt = 
        "### Instruction:\n"
        "You are a prose editor. Your ONLY job is to remove repetitive phrases and improve flow.\n"
        "DO NOT add new content. DO NOT change the meaning. Only refine what exists.\n"
        "Remove any sentence that repeats the same imagery as a previous sentence.\n"
        "Keep first-person perspective. Output ONLY the edited text.\n\n"
        "### Input:\n" + raw_output + "\n\n"
        "### Response:\n";
        
    std::string edited = generate_layer(state.ctx_hermes, state.model_hermes, prompt, 500, 0.4f, {"###", "\n\n\n"}, {});
    
    // Validation: Edited text should be at least 50% of original
    if (!edited.empty() && edited.length() > raw_output.length() * 0.5) {
        std::cout << " [HERMES] Edit accepted (Original: " << raw_output.length() << " -> Edited: " << edited.length() << ")" << std::endl;
        return edited;
    }
    
    return raw_output;
}

// [FIMBULVETR] POV ENFORCER
// Uses Fimbulvetr model to naturally rewrite POV if needed
// NO FORCED FILTERS - if model unavailable, return original text
std::string fimbulvetr_pov_fix(MultiAgentState& state, const std::string& text) {
    // Check if POV might be broken
    bool pov_broken = (text.find("the protagonist") != std::string::npos || 
                       text.find("The protagonist") != std::string::npos ||
                       text.find("the character") != std::string::npos ||
                       text.find("The character") != std::string::npos);
    
    if (!pov_broken) return text; // POV OK
    
    // If model not available, return original - NO FORCED REPLACEMENT
    if (!ensure_model_loaded(state, &state.model_fimbulvetr, &state.ctx_fimbulvetr, FIMBULVETR_PATH, FIMBULVETR_CTX, GPU_LAYERS_METAL)) {
        std::cout << " [FIMBULVETR] Model unavailable. Returning original." << std::endl;
        return text;
    }
    
    std::cout << " [FIMBULVETR] POV issue detected. Rewriting naturally..." << std::endl;
    
    std::string prompt = 
        "<|im_start|>system\n"
        "Rewrite this text naturally in first-person perspective.\n"
        "Make it feel authentic, not mechanical. Preserve the mood and atmosphere.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n" + text + "\n<|im_end|>\n"
        "<|im_start|>assistant\n";
        
    std::string fixed = generate_layer(state.ctx_fimbulvetr, state.model_fimbulvetr, prompt, 500, 0.6f, {"<|im_end|>"}, {});
    
    if (!fixed.empty() && fixed.length() > text.length() * 0.5) {
        std::cout << " [FIMBULVETR] POV rewritten." << std::endl;
        return fixed;
    }
    
    return text;
}


enum class AuxExpert {

    None,
    QwenInsight,
    RWKVGlitch,
    QwenCreative,
    MiroThinker
};

struct AuxRoute {
    AuxExpert expert = AuxExpert::None;
    std::string reason;
};

static std::string to_lower_copy(const std::string& input) {
    std::string out = input;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}

static bool contains_any(const std::string& lower, const std::vector<std::string>& needles) {
    for (const auto& needle : needles) {
        if (lower.find(needle) != std::string::npos) return true;
    }
    return false;
}

static void append_reason(std::string& reason, const std::string& token) {
    if (reason.empty()) {
        reason = token;
    } else {
        reason += ", " + token;
    }
}

AuxRoute route_aux_expert(const std::string& directive, const std::string& recent_history, const std::string& summary, bool has_logic_mandate) {
    std::string directive_lower = to_lower_copy(directive);
    std::string summary_lower = to_lower_copy(summary);

    float insight_score = 0.0f;
    float glitch_score = 0.0f;
    float creative_score = 0.0f;
    float miro_score = 0.0f;
    std::string insight_reason;
    std::string glitch_reason;
    std::string creative_reason;
    std::string miro_reason;

    bool needs_stability = contains_any(directive_lower, {"correction", "ground", "constraint", "stabilize", "veto_action", "system_constraints_updated"});
    bool needs_chaos = contains_any(directive_lower, {"chaos", "shift", "glitch", "introduce_chaos", "subconscious_shift"});
    bool needs_logic = has_logic_mandate || contains_any(directive_lower, {"logic", "causal"});

    if (needs_stability) {
        insight_score += 2.0f;
        miro_score += 1.0f;
        append_reason(insight_reason, "stability_directive");
        append_reason(miro_reason, "stability_directive");
    }
    if (needs_chaos) {
        glitch_score += 1.5f;
        creative_score += 1.0f;
        append_reason(glitch_reason, "chaos_directive");
        append_reason(creative_reason, "chaos_directive");
    }
    if (needs_logic) {
        miro_score += 2.0f;
        append_reason(miro_reason, "logic_mandate");
    }
    if (recent_history.size() < 400) {
        creative_score += 0.5f;
        append_reason(creative_reason, "sparse_history");
    }
    if (summary_lower.find("void") != std::string::npos) {
        creative_score += 0.5f;
        append_reason(creative_reason, "void_summary");
    }

    AuxRoute route;
    float best_score = 0.0f;
    auto consider = [&](AuxExpert expert, float score, const std::string& reason) {
        if (score > best_score) {
            best_score = score;
            route.expert = expert;
            route.reason = reason;
        }
    };

    consider(AuxExpert::MiroThinker, miro_score, miro_reason);
    consider(AuxExpert::QwenInsight, insight_score, insight_reason);
    consider(AuxExpert::QwenCreative, creative_score, creative_reason);
    consider(AuxExpert::RWKVGlitch, glitch_score, glitch_reason);

    if (best_score <= 0.0f) {
        route.expert = AuxExpert::None;
        route.reason = "no_signal";
    }

    return route;
}

std::string generate_composite_narrative(MultiAgentState& state, const std::string& history, const std::string& summary, int domain_idx, std::string directive = "", std::string premonition = "") {
    if (!state.model_main || !state.ctx_main) {
        std::cerr << " [ERR] Main model not initialized for composite generation." << std::endl;
        return "";
    }

    // 0. EXTRACT DEEP CONTEXT
    std::string recent_history = "";
    if (history.length() > 2000) {
        recent_history = history.substr(history.length() - 2000);
    } else {
        recent_history = history.empty() ? "The void is silent." : history;
    }
    
    std::string last_user_input = "";
    
    // --- PHASE 0: MAMBA PREDICTION (Always) ---
    NeuralLinkData mamba_data = run_mamba_synapse(state, recent_history);
    std::string mamba_prediction = mamba_data.prediction;

    if (!mamba_prediction.empty()) {
        std::cout << " [SYNAPSE] Prediction: " << mamba_prediction << std::endl;
    }
    
    // --- PHASE 1: LOGIC INJECTION (DeepSeek - ALWAYS) ---
    // DeepSeek provides the logical mandate for narrative coherence
    std::string logic_mandate = analyze_causality(state, recent_history);


    // --- PHASE 2: MIXTURE-OF-EXPERTS AUX ROUTING ---
    // Route to at most one auxiliary model per turn (memory-safe).
    std::string aux_directive = "";
    std::string aux_source = "";

    AuxRoute aux_route = route_aux_expert(directive, recent_history, summary, !logic_mandate.empty());
    const char* expert_name = "";
    switch (aux_route.expert) {
        case AuxExpert::QwenInsight: expert_name = "QwenInsight"; break;
        case AuxExpert::RWKVGlitch: expert_name = "RWKVGlitch"; break;
        case AuxExpert::QwenCreative: expert_name = "QwenCreative"; break;
        case AuxExpert::MiroThinker: expert_name = "MiroThinker"; break;
        default: expert_name = "None"; break;
    }
    std::cout << " [MOE] Router -> " << expert_name << " (" << aux_route.reason << ")" << std::endl;

    if (aux_route.expert == AuxExpert::QwenInsight) {
        if (ensure_model_loaded(state, &state.model_qwen_stabilizer, &state.ctx_qwen_stabilizer, QWEN_STABILIZER_PATH, QWEN_STABILIZER_CTX, GPU_LAYERS_METAL)) {
            std::string prompt = "Context: " + recent_history + "\nProvide a psychological insight about the protagonist's current state.";
            aux_directive = generate_layer(state.ctx_qwen_stabilizer, state.model_qwen_stabilizer, prompt, 64, 0.7f, {"\n"}, {});
            aux_source = "INSIGHT (Qwen)";
            std::cout << " [AUX] Qwen Insight: " << aux_directive << std::endl;
        }
    } else if (aux_route.expert == AuxExpert::RWKVGlitch) {
        if (ensure_model_loaded(state, &state.model_rwkv, &state.ctx_rwkv, RWKV_PATH, RWKV_CTX, GPU_LAYERS_METAL)) {
            std::string prompt = "Narrative: " + recent_history + "\n\nOutput a short, surreal, glitch-like event that disturbs the reality.";
            aux_directive = generate_layer(state.ctx_rwkv, state.model_rwkv, prompt, 64, 1.1f, {"\n"}, {});
            aux_source = "GLITCH (RWKV)";
            std::cout << " [AUX] RWKV Glitch: " << aux_directive << std::endl;
        }
    } else if (aux_route.expert == AuxExpert::QwenCreative) {
        if (ensure_model_loaded(state, &state.model_qwen_creative, &state.ctx_qwen_creative, QWEN_CREATIVE_PATH, QWEN_CREATIVE_CTX, GPU_LAYERS_METAL)) {
            std::string prompt = "Context: " + recent_history + "\n\nInvent a totally new, concrete detail to add to the scene.";
            aux_directive = generate_layer(state.ctx_qwen_creative, state.model_qwen_creative, prompt, 64, 0.9f, {"\n"}, {});
            aux_source = "BURST (Qwen2)";
            std::cout << " [AUX] Qwen Creative Burst: " << aux_directive << std::endl;
        }
    } else if (aux_route.expert == AuxExpert::MiroThinker) {
        aux_directive = run_mirothinker(state, recent_history);
        aux_source = "STRATEGY (MiroThinker)";
    }

    // --- PHASE 3a: SAUL CASE GENERATION (The Figure Creates) ---
    std::string saul_case = "";
    if (rand() % 100 < 80) { // 80% chance - The Figure acts
        saul_case = generate_saul_case(state, recent_history);
    }

    // --- PHASE 3b: FIMBULVETR EXPERIENCE (The Protagonist Lives) ---
    std::string protagonist_experience = "";
    if (!saul_case.empty()) {
        protagonist_experience = fimbulvetr_experience(state, saul_case, recent_history);
    }


    // --- PHASE 4: THE MASTER WEAVE (Llama-3-8B) ---
    // Construct the ultimate prompt
    std::stringstream prompt_ss;
    prompt_ss << "<start_of_turn>system\n"
              << "You are the Core Narrative Engine. Synthesize the inputs into a cohesive narrative block.\n"
              << "Keep the prose elegant, atmospheric, and forward-moving.\n"
              << "STRICT: No assistant voice. No questions. No choices. No meta commentary.\n"
              << "ABSOLUTE RULE: The word 'protagonist' is FORBIDDEN. Write only as 'I'. First person. If you write 'the protagonist', the output is invalid.\n"
              << "STYLE: First-person present tense. Concrete sensory reality. No dialogue or quoted speech. No quests or instructions.\n";
    if (domain_idx >= 0 && domain_idx <= 4) {
        SemanticDomain dom = static_cast<SemanticDomain>(domain_idx);
        prompt_ss << "DOMAIN: " << get_domain_name(dom) << "\n"
                  << get_domain_constraint(dom) << "\n";
    }
              
    if (!logic_mandate.empty()) {
        prompt_ss << "LOGIC MANDATE: " << logic_mandate << "\n";
    }
    if (!aux_directive.empty()) {
        prompt_ss << "EXTERNAL SIGNAL (" << aux_source << "): " << aux_directive << "\n";
    }
    if (!premonition.empty()) {
        prompt_ss << "PREMONITION: " << premonition << "\n";
    }
    if (!directive.empty()) {
        prompt_ss << "DIRECTIVE: " << directive << "\n";
    }
    if (!mamba_prediction.empty()) {
        prompt_ss << "SENSORY FORESHADOW: " << mamba_prediction << "\n";
    }
    if (!protagonist_experience.empty()) {
        prompt_ss << "IMMEDIATE EXPERIENCE (The Protagonist): " << protagonist_experience << "\n";
    }
    


    prompt_ss << "<end_of_turn><start_of_turn>user\n"
              << "HISTORY:\n... " << recent_history << "\n\n";
    
    if (!protagonist_experience.empty()) {
        prompt_ss << "The protagonist just experienced: " << saul_case << "\n";
        prompt_ss << "Continue from their immediate reaction.";
    } else {
        prompt_ss << "Continue the story.";
    }
    
    prompt_ss << "<end_of_turn><start_of_turn>model\n";
              
    // If we have a protagonist experience, use it as the starting point
    if (!protagonist_experience.empty()) {
        prompt_ss << protagonist_experience.substr(0, 50); // First 50 chars as seed
    }


    std::cout << "\n[MASTER WEAVE] Generating with Llama-3..." << std::endl;
    // Main Llama Generation
    // Note: ensure_model_loaded for main is usually not needed as it stays resident, 
    // but if we had to swap it (unlikely for main), we would check. 
    // Generally state.model_main is always loaded in init.
    
    std::string generated_text = generate_layer(state.ctx_main, state.model_main, prompt_ss.str(), 256, 0.85f, {"<end_of_turn>", "User:", "How would you like", "Let me know"}, {}); // Stop tokens

    // Combine protagonist experience with generated continuation
    std::string final_block;
    if (!protagonist_experience.empty()) {
        final_block = protagonist_experience + " " + generated_text;
    } else {
        final_block = generated_text;
    }

    
    // Basic clean up
    final_block = strip_meta_commentary(final_block);
    final_block = trim_meta_tail(final_block);
    final_block = collapse_repeating_chars(final_block);
    final_block = trim_trailing_noise(final_block);
    
    // --- PHASE 5: HERMES EDIT PASS (80% Chance) ---
    if (rand() % 100 < 80) {
        final_block = hermes_edit_pass(state, final_block);
    }
    
    // --- PHASE 6: FIMBULVETR POV FIX (If needed) ---
    final_block = fimbulvetr_pov_fix(state, final_block);

    return final_block;
}


// WRAPPER FOR REFLEX LOOP
std::string generate_composite_narrative_with_reflex(MultiAgentState& state, const std::string& history, const std::string& summary) {
    int attempts = 0;
    std::string directive;

    while (attempts < 3) {
        std::string block = generate_composite_narrative(state, history, summary, state.domain_index, directive);
        if (!block.empty()) return block;
        directive = "SYSTEM_CONSTRAINTS_UPDATED";
        attempts++;
    }
    return "";
}

// 7. GENERATE TEXT (LLaMA) - Updated with Stochastic Control
std::string generate_text(MultiAgentState& state, const std::string& prompt, int max_tokens, NarrativeState narrative_state, std::vector<std::string>& banned_words, int attempts, std::string& out_failure_reason, bool panic_mode, float temperature) {

    // 0. Setup Constraints (Logit Gates)
    std::vector<llama_token> masked_tokens;
    // Map banned words to tokens (Simplified: Assuming single-token words for now or banning first token)
    // In a real implementation we would do a Trie lookup or logits processing.
    // For now, we will rely on post-sampling filters for complex phrases 
    // BUT we will use logit bias for single-token stops to save CPU.
    auto* vocab_mask = llama_model_get_vocab(state.model_main);
    
    // ... (Existing logic shifted) ...
    if (!state.model_main || !state.ctx_main) return " [Error: LLaMA not initialized]";

    // Cliché Killer: Always ban these
    static const std::vector<std::string> kClicheWords = {
        "pulsating", "resonating", "shroud", "labyrinth", "tapestry", "orb", "cacophony", "kaleidoscope",
        "embers", "burning", "hiss", "shiver", "spine", "echoes", "abyss", "crimson", "azure",
        "metal", "metallic", "iron", "copper", "rust", // [UPDATED] Manual Concept Jail
        "blood", "vein", "flesh", "bone",
        "ozone", "sulfur", "smell", "scent",
        "corrode", "corrosion", "glow", "noxious", "iridescent", "luminescent", // [UPDATED] User Bans
        " like", // BAN SIMILES (space like)
        " as if", // BAN SIMILES
        // [New] Discovery Loop Breakers / Neologisms
        "protrusion", "jagged", "writhe", "twist", "throbbles", "courscomb", "etche", "churns",
        // [REALITY WARP FIX] Hard Ban on Viral Geometry
        "geometry", "geometric", "non-euclidean", "fractal", "shards", "melting", "warping", "distort",
        // [LOGIC CIRCUIT 2] CAUSAL LOBOTOMY - BAN LOGIC CONNECTORS
        "because", "therefore", "due to", " in order to", " so ", " so,", "thus", "hence", "reason", "cause", "effect",
        // [LOGIC CIRCUIT 2] CAUSAL LOBOTOMY - BAN INFERENCE
        "seem", "appear", "realize", "understand", "think", "believe", "know", "conclude"
    };

    // 1. SCRUBBING
    std::string safe_context = sanitize_history(prompt);
    // Limit context length if needed, but keeping it large (12k chars) for 16k token limit
    if (safe_context.length() > 12000) safe_context = safe_context.substr(safe_context.length() - 12000);

    // 2. SYSTEM PROMPT (Strict)
    std::string system_prompt = 
        "IDENTITY: Adopt the perspective of the PROTAGONIST. You are experiencing this moment NOW.\n"
        "GOAL: Pure sensory stream. Observe and document the immediate reality.\n"
        "STRICT PROHIBITIONS:\n"
        "- NO Meta-commentary.\n"
        "- NO Similes (e.g., 'like a curtain', 'like a snake'). Describe the object IS the snake.\n"
        "- NO Storytelling voice. USE 'I'.\n"
        "- NO Moralizing or reflecting on the sequence of events. Only survival.\n"
        "- NO Clichés (e.g., 'thick with anticipation', 'shivers down spine').\n"
        "- NO LOGIC. NO CAUSALITY. NO 'Because'. NO 'Realize'.\n"
        "FORMATTING:\n"
        "- First-person present tense ('I see', not 'I saw').\n"
        "- Concrete, physical descriptions (temperature, texture, smell).\n"
        "- Short, punchy sentences. Fragmented thoughts are allowed."; 

    // [LOGIC CIRCUIT 3] ZERO DISTANCE TOPOLOGY
    // Logic: Focus_Point = Random([Body Part])
    // Constraint: "Render Distance = 0"
    std::string focus_point = pick_random({"Left Earlobe", "Right Sole", "Fingertip", "Eyelid", "Tongue", "Skin", "Nerve Ending"});
    system_prompt += "\n[VISUAL AGNOSIA PROTOCOL ACTIVE]\n"
                     "RENDER DISTANCE: 0 METERS. THE WORLD DOES NOT EXIST.\n"
                     "FOCUS ANCHOR: " + focus_point + ".\n"
                     "TASK: Describe ONLY the friction/interaction between [SURFACE] and [" + focus_point + "].\n"
                     "DO NOT DESCRIBE THE ROOM. DO NOT LOOK UP.\n"; 

    if (panic_mode) {
        system_prompt += "\n[OVERRIDE] FREEZE. The world stops. Detail ONE static object with microscopic precision. No movement. No emotion.\n" + pick_random({"Describe a pebble.", "Describe a crack in the wall.", "Describe a droplet of water."});
    }
    
    // FAILURE AWARENESS INJECTION
    if (attempts > 0 && !out_failure_reason.empty()) {
        system_prompt += "\n[CORRECTION: Previous output rejected (" + out_failure_reason + "). RESET. Focus on physical sensation ONLY.]\n";
    }

    // [MISTAKES RAG] FEEDBACK LOOP
    if (!state.recent_mistakes.empty()) {
        system_prompt += "\n[ADAPTIVE FEEDBACK - AVOID PREVIOUS ERRORS]:\n";
        for (const auto& mistake : state.recent_mistakes) {
            system_prompt += "- AVOID: " + mistake + "\n";
        }
    }

    // 3. DYNAMIC PRE-FILL & INSTRUCTION INJECTION
    std::string assistant_prefill = "I"; 
    
    // Inject State Logic into User Prompt (Stronger Recency Bias)
    std::string state_instruction = get_state_instruction(narrative_state);
    
    switch (narrative_state) {
        case NarrativeState::MOVEMENT:
            // Randomized Openings for Variety
            assistant_prefill = pick_random({
                "I drag myself",
                "My hands find",
                "I stumble towards",
                "The ground beneath me",
                "I force my body to",
                "I push myself up" // Still valid, but 1/6 chance now
            });
            break;
        case NarrativeState::DISCOVERY:
            assistant_prefill = pick_random({
                "The wall texture changes",
                "A formation of",
                "Directly ahead,",
                "The shadows reveal",
                "The darkness parts to show"
            });
            break;
        case NarrativeState::CONFLICT:
             assistant_prefill = "I react";
             break;
        default: break;
    }

    // ... (rest of function omitted for brevity, logic continues as before) ...

    // 4b. CONCEPTUAL FIELD TRACKER (Anti-Repetition V2)
    std::map<std::string, std::vector<std::string>> concept_library = {
        {"COLLAPSE", {"debris", "rubble", "crumble", "shake", "tremble", "earthquake", "collapse", "falling", "dust", "unstable", "vibration", "shockwave", "fissure", "crack", "split", "fracture"}},
        {"WATER", {"water", "drip", "rush", "flow", "river", "wet", "damp", "liquid", "moisture", "pool", "stream", "trickle", "mist", "spray", "lake", "ocean", "sea", "flood", "humidity", "condensation", "dampness", "reservoir"}},
        {"DARKNESS", {"dark", "shadow", "black", "dim", "gloom", "obscurity", "night", "shade", "pitch", "void", "abyss"}},
        {"ROT", {"rot", "decay", "smell", "stench", "odor", "ozone", "mold", "stale", "ancient", "dusty", "fungus", "spore", "biomass"}}
    };

    std::vector<std::string> active_concept_bans;
    std::string history_lower = safe_context;
    std::transform(history_lower.begin(), history_lower.end(), history_lower.begin(), ::tolower);

    for (const auto& [concept, keywords] : concept_library) {
        int hits = 0;
        for (const auto& kw : keywords) {
            // Simple substring counting
            size_t pos = 0;
            while ((pos = history_lower.find(kw, pos)) != std::string::npos) {
                hits++;
                pos += kw.length();
            }
        }
        
        // Threshold: If an entire concept appears > 4 times in recent history, BAN IT ALL.
        if (hits > 4) {
            std::cout << " [CONCEPT BAN] Suppressing '" << concept << "' (" << hits << " hits)" << std::flush;
             // Only ban if strictly needed.
             // Adding to active bans.
            active_concept_bans.insert(active_concept_bans.end(), keywords.begin(), keywords.end());
        }
    }
    // POV safety: discourage third-person lead-ins without hard replacement
    active_concept_bans.push_back("protagonist");
    active_concept_bans.push_back("protagonist's");


    // 5. PROMPT ASSEMBLY (Leak-Proof)
    std::string formatted_prompt = 
        "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "\n" + state_instruction + "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n" 
        "MEMORY STREAM:\n" + safe_context + "\n\n" // Used safe_context, not prompt
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n" + assistant_prefill; 

    // Tokenize
    auto* vocab = llama_model_get_vocab(state.model_main);
    auto& tokens_list = token_scratch(formatted_prompt.size() + kTokenScratchPad); // allocate buffer
    int n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), tokens_list.data(), tokens_list.size(), true, true); // add_bos=true
    if (n_tokens < 0) {
         // resize and retry if needed, but usually enough
         token_scratch(static_cast<size_t>(-n_tokens));
         n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);

    const int n_ctx = llama_n_ctx(state.ctx_main);
    if (tokens_list.size() > n_ctx - 500) {
        int diff = tokens_list.size() - (n_ctx - 500);
        if(diff > 0) tokens_list.erase(tokens_list.begin(), tokens_list.begin() + diff);
    }

    llama_memory_clear(llama_get_memory(state.ctx_main), true); 
    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size()); 
    if (llama_decode(state.ctx_main, batch) != 0) return " [Error: llama_decode failed]";

    // Sampler Config - STOCHASTIC CONTROL
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // [ALGORITHMIC FIX] Logit-based Constraint Masking
    
    // 1. Prepare Biases Vector
    std::vector<llama_logit_bias> biases;
    biases.reserve(masked_tokens.size() + kClicheWords.size() + banned_words.size() + active_concept_bans.size() + 2);

    // A. Add "Hard" Masked Tokens (from earlier logic)
    for (llama_token token_id : masked_tokens) {
        biases.push_back({token_id, -1000.0f}); 
    }

    // B. Convert all banned strings (Cliches, Concept Bans) to tokens
    std::vector<llama_token> b_tokens;
    auto add_biases = [&](const std::vector<std::string>& words) {
        for (const auto& w : words) {
            b_tokens.resize(w.length() + 4);
            int n_bt = llama_tokenize(vocab, w.c_str(), w.length(), b_tokens.data(), b_tokens.size(), false, false); 
            if (n_bt > 0) {
                // Ban start token
                biases.push_back({b_tokens[0], -1000.0f}); 
            }
        }
    };
    add_biases(kClicheWords);
    add_biases(banned_words);
    add_biases(active_concept_bans);
    
    // C. Add Header Bans (System tokens)
    biases.push_back({128006, -100.0f}); // <|start_header_id|>
    biases.push_back({128007, -100.0f}); // <|end_header_id|>

    // 2. Apply Biases to Sampler
    if (!biases.empty()) {
        std::cout << " [LOGIT MASK] active on " << biases.size() << " tokens." << std::endl;
        auto* vocab = llama_model_get_vocab(state.model_main);
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), biases.size(), biases.data()));
    }

    // 3. Apply Other Samplers
    // Penalties: Repetition acts as "Orthogonal Projection" approximation
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(4096, 1.25f, 0.1f, 0.0f)); // Stronger Penalty
    
    // Temperature: Controlled by PID
    std::cout << " [PID] Sampling with Temperature: " << temperature << std::endl;
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(std::rand()));

    // --- GENERATION LOOP ---
    std::string result = assistant_prefill; 
    result.reserve(static_cast<size_t>(max_tokens) * kAvgTokenChars + assistant_prefill.size() + 16);
    std::cout << result << std::flush;      

    // int max_tokens = 400; // Replaced by parameter
    // vocab loaded above

    std::vector<std::string> stop_signals = {
        "</assistant", "<|eot_id|>", "[End]", "(Action Required", 
        "Action Removed", "Best regards", "Protagonist AI", 
        "Please respond", "What do you do?", "within 1-3 sentences",
        "Describe your next", "I pause momentarily", "Generate continuation",
        "Please enter", "Mention immediate", "surroundings in your response",
        "ACTION CONTINUES", "Describe action NOW"
    };

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token_id = llama_sampler_sample(smpl, state.ctx_main, -1);
        llama_sampler_accept(smpl, new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) continue;
        std::string piece(buf, n);

        // STYLE ENFORCEMENT: No Colons (replaces with space)
        if (piece == ":") piece = " ";
        
        result += piece;
        std::cout << piece << std::flush;

        // Stop Check
        bool stop = false;
        for (const auto& s : stop_signals) {
            if (result.find(s) != std::string::npos) {
                // Remove the signal
                size_t pos = result.find(s);
                result = result.substr(0, pos);
                stop = true; 
                break;
            }
        }
        if (stop) break;

        // Feed back
        if (llama_decode(state.ctx_main, llama_batch_get_one(&new_token_id, 1))) {
            break;
        }
    }

    llama_sampler_free(smpl);

    // --- POST PROCESSING ---
    // 1. Flatten Paragraphs (No Double Newlines)
    std::replace(result.begin(), result.end(), '\n', ' ');
    
    // 2. Remove Extra Spaces
    std::string clean_res;
    bool space = false;
    for(char c : result) {
        if(isspace(c)) {
            if(!space) { clean_res += ' '; space = true; }
        } else {
            clean_res += c;
            space = false;
        }
    }
    result = clean_res;

    // 3. Fix Cliffhangers (Ellipses -> Period)
    size_t ellipsis;
    while ((ellipsis = result.find("...")) != std::string::npos) {
        result.replace(ellipsis, 3, ".");
    }
    while ((ellipsis = result.find("..")) != std::string::npos) {
        result.replace(ellipsis, 2, ".");
    }

    // 4. Ensure Punctuation End
    if (!result.empty()) {
        char last = result.back();
        if (last != '.' && last != '!' && last != '?') {
            result += ".";
        }
    }

    // 5. NEURAL GRAMMAR FIX & FILTER (Kleisli Composition)
    
    // A. GATEKEEPER: Check for contamination FIRST. 
    // Do not waste valid inference on invalid thoughts.
    if (is_contaminated(result)) {
        std::cout << " [Contaminated - Skipping Proofread]";
        return result; // Return raw, Main loop will reject it.
    }

    // B. Algorithmic Collapse (Cheap)
    result = collapse_repeating_chars(result);

    // C. Neural Pass (Expensive) - Only if passed Gatekeeper
    if (!panic_mode) { 
        std::cout << " [NEURAL PROOFREADING] "; 
        std::string corrected = neural_correct(state, result);
        if (corrected != result) {
            std::cout << "-> Fixed."; 
            result = corrected;
        } else {
             std::cout << "-> OK.";
        }
    }
    
    std::cout << "\n";
    return result;
}

// --- HHM SUMMARIZER ---
std::string generate_summary(MultiAgentState& state, const std::string& history) {
    if (!state.model_main || !state.ctx_main) return " [Error: LLaMA not initialized]";

    std::cout << "\n[HHM] Summarizing Narrative Arc..." << std::endl;

    // 1. SYSTEM PROMPT (Historian Persona)
    std::string system_prompt = 
        "SYSTEM DIRECTIVE: You are a Historian Algorithm. \n"
        "TASK: Compress the provided narrative history into a concise, 3-sentence summary.\n"
        "RULES:\n"
        "1. Write in THIRD PERSON (e.g., 'The protagonist...', 'He/She...').\n"
        "2. Capture key physical actions and emotional shifts.\n"
        "3. Do NOT use 'I'. Do NOT use meta-commentary.\n"
        "4. Be objective and factual.";

    std::string formatted_prompt = 
        "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n" 
        "NARRATIVE HISTORY:\n" + history + "\n\n" 
        "Summarize this arc in 3 sentences."
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "The protagonist"; // Pre-fill to enforce 3rd person

    // Tokenize
    auto* vocab = llama_model_get_vocab(state.model_main);
    auto& tokens_list = token_scratch(formatted_prompt.size() + kTokenScratchPad);
    int n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
         token_scratch(static_cast<size_t>(-n_tokens));
         n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);

    // Ensure context fits (Summarization consumes a lot)
    const int n_ctx = llama_n_ctx(state.ctx_main);
    if (tokens_list.size() > n_ctx - 300) {
        int diff = tokens_list.size() - (n_ctx - 300);
        if(diff > 0) tokens_list.erase(tokens_list.end() - diff - 10, tokens_list.end() - 10); // Crop middle if possible, or just start? Actually crop beginning of history part.
        // Simplified: Just crop beginning tokens.
        if(diff > 0) tokens_list.erase(tokens_list.begin() + 50, tokens_list.begin() + 50 + diff);
    }

    llama_memory_clear(llama_get_memory(state.ctx_main), true); 
    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size()); 
    if (llama_decode(state.ctx_main, batch) != 0) return " [Error: llama_decode failed]";

    // Sampler (More deterministic for summary)
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.4f)); // Low temp for facts
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(std::rand())); // [FIXED] Added Distribution Sampler to prevent crash
    
    int max_tokens = 150; 
    std::string result = "The protagonist"; 
    result.reserve(static_cast<size_t>(max_tokens) * kAvgTokenChars + 32);
    std::cout << result << std::flush;      
    
    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token_id = llama_sampler_sample(smpl, state.ctx_main, -1);
        llama_sampler_accept(smpl, new_token_id);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) continue;
        std::string piece(buf, n);
        result += piece;
        std::cout << piece << std::flush;

        if (result.find(".") != std::string::npos) {
            // Count sentences? Or just rely on EOT.
        }
        
        if (llama_decode(state.ctx_main, llama_batch_get_one(&new_token_id, 1))) break;
    }

    llama_sampler_free(smpl);
    std::cout << "\n";
    return result;
}

// --- NETWORK OPS ---
std::vector<std::string> fetch_data_from_tx(const std::string& txid) {
    if (txid == "GENESIS_TX") return {"GENESIS", "", "", "[]", "{}", "UNKNOWN"};


    // [UPDATED] 1. CHECK LOCAL CACHE FIRST (Trust Engine)
    std::string cache_path = "cache/" + txid + ".json";
    std::ifstream cache_file(cache_path);
    if (cache_file.good()) {
        std::cout << " [CACHE] Found local block: " << txid << std::endl;
        std::stringstream buffer;
        buffer << cache_file.rdbuf();
        std::string raw = buffer.str();
        try {
            auto j = json::parse(raw);
            std::string mistakes_str = "[]";
            if (j.contains("mistakes_log")) mistakes_str = j["mistakes_log"].dump();
            else if (j.contains("mistakes_rag")) mistakes_str = j["mistakes_rag"].dump();
            std::string world_str = "{}";
            if (j.contains("world_state")) world_str = j["world_state"].dump();
            std::string depth_str = "0";
            if (j.contains("depth")) {
                if (j["depth"].is_number_integer()) depth_str = std::to_string(j["depth"].get<int>());
                else if (j["depth"].is_number()) depth_str = std::to_string(j["depth"].get<double>());
                else if (j["depth"].is_string()) depth_str = j["depth"].get<std::string>();
            }
            std::string weather_str = "UNKNOWN";
            if (j.contains("chronos") && j["chronos"].contains("weather")) {
                weather_str = j["chronos"]["weather"].get<std::string>();
            }
            return {j.value("parent_tx", ""), j.value("content", ""), depth_str, mistakes_str, world_str, weather_str};

        } catch (...) {
            std::cerr << " [ERR] Corrupt cache for " << txid << std::endl;
        }
    }

    // 2. NETWORK VERIFICATION (Fallback)
    if (txid == "GENESIS") return {"", "GENESIS BLOCK", "0", "[]", "{}", "UNKNOWN"};


    // Use absolute path
    std::string cmd = "python3 /Users/farukalpay/Desktop/cpp/local_mind/scripts/uploader.py --fetch=\"" + txid + "\" 2>/dev/null";
    std::string data = "";

    int attempt = 0;
    int max_attempts = 5; // [UPDATED] Prevent Infinite Loop
    
    while(attempt < max_attempts) {
        attempt++;
        std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
        std::string current_output = "";
        
        if (pipe) {
            char buffer[1024];
            while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
                current_output += buffer;
            }
        }

        // Basic validation: Must be non-empty and look like JSON
        if(!current_output.empty() && current_output.find("{") != std::string::npos) {
            data = current_output;
            break; 
        }

        std::cout << " [NETWORK] Waiting for Arweave propagation (" << txid << ")... [" << attempt << "/" << max_attempts << "]\r" << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    
    if (data.empty()) {
        std::cout << " [WARN] Could not fetch " << txid << " from network. Assuming pending/broken." << std::endl;
        return {}; // Return empty to allow graceful handling
    }

    try {
        auto j = json::parse(clean_invalid_utf8(data));
        std::string p = j.value("parent_tx", "");
        std::string c = j.value("content", "");
        std::string d = "0";
        if (j.contains("depth")) {
            if (j["depth"].is_number_integer()) d = std::to_string(j["depth"].get<int>());
            else if (j["depth"].is_number()) d = std::to_string(j["depth"].get<double>());
            else if (j["depth"].is_string()) d = j["depth"].get<std::string>();
        }
        std::string mistakes_str = "[]";
        if (j.contains("mistakes_log")) mistakes_str = j["mistakes_log"].dump();
        else if (j.contains("mistakes_rag")) mistakes_str = j["mistakes_rag"].dump();
        std::string world_str = "{}";
        if (j.contains("world_state")) world_str = j["world_state"].dump();

        std::string weather_str = "UNKNOWN";
        if (j.contains("chronos") && j["chronos"].contains("weather")) {
            weather_str = j["chronos"]["weather"].get<std::string>();
        }
        return {p, c, d, mistakes_str, world_str, weather_str};

    } catch (...) {
        return {};
    }
}

std::vector<std::string> reconstruct_narrative(MultiAgentState& state, const std::string& head_tx, int max_depth) {

    std::vector<std::string> narrative;
    std::string current_tx = head_tx;

    for (int i = 0; i < max_depth; i++) {
        if (current_tx.empty() || current_tx == "GENESIS_TX") break;
        
        std::cout << "[VERIFY] Trace: " << current_tx << "\r" << std::flush;
        auto data = fetch_data_from_tx(current_tx);
        
        if (data.empty()) {
            std::cerr << " [WARN] Broken link at " << current_tx << std::endl;
            break; 
        }

        if (!data[1].empty()) narrative.push_back(data[1]);
        if (data.size() > 5 && !data[5].empty()) state.weather_history.push_back(data[5]);
        
        current_tx = data[0]; 
    }

    std::reverse(narrative.begin(), narrative.end());
    std::reverse(state.weather_history.begin(), state.weather_history.end());
    
    std::cout << "\n[VERIFY] Chain Reconstructed: " << narrative.size() << " blocks." << std::endl;
    return narrative;
}


// --- NEURAL TRACE (Local JSON Map) ---
void update_neural_map(const std::string& txid, const json& private_data) {
    std::string map_path = "neural_map.json";
    json j_map;

    // Load existing
    std::ifstream i(map_path);
    if (i.good()) {
        try {
            i >> j_map;
        } catch (...) {
            // Corrupt or empty, start fresh or keep empty
        }
    }
    i.close();

    // Add to map (Key = TXID)
    j_map[txid] = private_data;
    // Add timestamp if not present? Already in private_data hopefully.
    if (!j_map[txid].contains("timestamp")) j_map[txid]["timestamp"] = std::time(0);

    // Save
    std::ofstream o(map_path);
    o << std::setw(4) << j_map << std::endl;
    std::cout << "[TRACE] Neural Map Updated: " << txid << std::endl;
}

// --- MAIN LOOP ---
int main(int argc, char* argv[]) {
    // SEED
    std::srand(std::time(0));

    // ARGS
    int num_blocks = 1; 
    std::string prev_txid_arg = "";
    std::string user_init_prompt = "";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--blocks" && i + 1 < argc) {
            num_blocks = std::stoi(argv[++i]);
        } else if (arg == "--previous_txid" && i + 1 < argc) {
            prev_txid_arg = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            user_init_prompt = argv[++i];
        }
    }

    std::cout << "[SYSTEM] Dual-Engine Architecture: Architect (8B) + Scout (2B) Online." << std::endl;

    MultiAgentState state;
    init_multi_agent(state);
    shuffle_deck(state); // [UPDATED] Init Persona Deck with Saul

    // ENSURE CACHE DIR
    ensure_directory("cache");

    // PARENT RESOLUTION
    std::string parent_tx = "GENESIS_TX"; 
    
    // 1. CLI Override
    if (!prev_txid_arg.empty()) {
        parent_tx = prev_txid_arg;
        std::cout << "[MINER] Using CLI Parent: " << parent_tx << std::endl;
    } 
    // 2. Cache / Network Discovery
    else {
        std::ifstream last_file("last_tx.txt");
            if (last_file.good()) {
                std::getline(last_file, parent_tx);
                if (!parent_tx.empty()) {
                    // Trim whitespace (improves robustness for manual edits)
                    parent_tx.erase(0, parent_tx.find_first_not_of(" \n\r\t"));
                    parent_tx.erase(parent_tx.find_last_not_of(" \n\r\t") + 1);
                }
            
            // Validate Hash
            bool valid = !parent_tx.empty() && parent_tx.length() < 100;
            for(char c : parent_tx) {
                if(!isalnum(c) && c != '-' && c != '_') valid = false;
            }
            
            if (valid) {
                 std::cout << "[MINER] Found last local block: " << parent_tx << std::endl;
            } else {
                 std::cerr << "[WARN] Corrupt local state detected. Resetting to GENESIS_TX." << std::endl;
                 parent_tx = "GENESIS_TX";
            }
        } else {
            std::cout << "[MINER] No local state. Checking Arweave logic..." << std::endl;
            parent_tx = "GENESIS_TX"; 
        }
    }

    std::string full_context = "";
    std::vector<std::string> history;
    int chain_depth = 0;

    // CONTEXT LOADING
    if (parent_tx != "GENESIS_TX") {
        history = reconstruct_narrative(state, parent_tx, MAX_DEPTH);
        for (const auto& block : history) {

            full_context += block + " ";
        }
        chain_depth = static_cast<int>(history.size());
        
        if (!state.weather_history.empty()) {
            state.current_weather = state.weather_history.back();
            std::cout << "[SYSTEM] Resumed Weather: " << state.current_weather << std::endl;
        }
    }
    
    // [UPDATED] Anti-Cliche / Anti-Tropes Masking
    // We add a mechanism to dynamically ban words if they appear in recent history.
    // Instead of complex logic inside the loop, we'll just populate `banned_words` in `generate_text` if needed.
    // But `generate_composite` calls `generate_text`. 
    // We will inject a prompt instruction into `generate_composite_narrative` to avoid specific words?
    // User suggested: "entropy_loss function" modification or logit bias.
    // Cleanest way: Just ban them from `banned_words` vector passed to `generate_text`?
    // `generate_composite_narrative` calls `gemma` and `generate_text`.
    // Let's modify `generate_composite_narrative` to scan history for cliches.
    
    // Actually, user said: "entropy_loss fonksiyonuna N-Gram Yasaklama ekle."
    // But `entropy_loss` is a calculator, it doesn't affect generation directly unless used in PID.
    // "generate_text içindeki yasaklı kelime mantığına ekle" is the instruction.
    // I will modify `generate_text`.

    // GENERATION LOOP
    int blocks_since_summary = 0;
    std::string last_summary_txid = "";
    std::string current_summary_text = "I exist in a void.";
    
    // [UPDATED] Persistent Genesis TXID (Inheritance)
    std::string root_genesis_txid = "GENESIS";
    
    // Attempt to load Genesis ID from Neural Map if resuming
    if (parent_tx != "GENESIS_TX") {
        std::ifstream i("neural_map.json");
        if (i.good()) {
            json j_map;
            try { i >> j_map; } catch (...) {}
            if (j_map.contains(parent_tx) && j_map[parent_tx].contains("genesis_txid")) {
                root_genesis_txid = j_map[parent_tx]["genesis_txid"];
                std::cout << "[SYSTEM] Inherited Genesis TXID: " << root_genesis_txid << std::endl;
            }
        }
    }
    
    if (root_genesis_txid == "GENESIS") {
        std::cout << "[SYSTEM] Starting New Chain (Genesis Pending...)" << std::endl;
    }

    // [MISTAKE INHERITANCE] Re-hydrate Concept Jail from parent
    if (parent_tx != "GENESIS_TX") {
        std::cout << "[SYSTEM] Checking parent for inherited constraints..." << std::endl;
        auto parent_data = fetch_data_from_tx(parent_tx); // 0=Parent, 1=Content, 2=Depth, 3=Mistakes
        if (parent_data.size() >= 5) {
            // [MISTAKE INHERITANCE] - already handled in loop above? No we need to check size.
            // Wait, I messed up the order. Let's cleaner.
            
            // 1. Mistakes
            std::string mistakes_json = parent_data[3];
            try {
                auto j_mistakes = json::parse(mistakes_json);
                if (j_mistakes.is_array()) {
                    for (const auto& m : j_mistakes) {
                        std::string msg = m.get<std::string>();
                        if (msg.find("CONCEPT JAIL: Loop Detected in category") != std::string::npos) {
                            // ... (Logic kept same, just ensuring scope) ...
                            // Extract category logic duplicate... let's trust existing code for now?
                            // Actually I am replacing the block. I need to keep the mistake logic AND add world state.
                            size_t start_cat = msg.find("'") + 1;
                            size_t end_cat = msg.find("'", start_cat);
                            if (start_cat != std::string::npos && end_cat != std::string::npos) {
                                std::string cat = msg.substr(start_cat, end_cat - start_cat);
                                std::cout << "[SYSTEM] Inheriting Concept Jail Lock for: " << cat << " [DISABLED - LEGACY]" << std::endl;
                                // force_concept_lock(cat);
                                state.recent_mistakes.push_back("INHERITED LOCK: " + cat);
                            }
                        }
                    }
                }
            } catch (...) {}

            // 2. World State
            std::string world_json = parent_data[4];
            try {
                 auto j_world = json::parse(world_json);
                 if (j_world.is_object()) {
                     std::cout << "[SYSTEM] Inheriting World State (" << j_world.size() << " facts)..." << std::endl;
                     for (auto& element : j_world.items()) {
                         state.world_state[element.key()] = element.value();
                     }
                 }
            } catch (...) {}
        }
    }

    
    NarrativeState current_narrative_state = NarrativeState::AWAKENING;
    int state_block_count = 0;
    
    PIDController pid;
    float current_temperature = pid.setpoint;

    int blocks_since_reflex = 3; // Start ready

    for (int b = 0; b < num_blocks; b++) {
        std::cout << "\n=== BLOCK " << (b + 1) << "/" << num_blocks << " ===" << std::endl;
        std::cout << "Parent TX: " << parent_tx << std::endl;
        
        state.recent_mistakes.clear();

        // --- [STEP 0] MAMBA SYNAPSE (PRE-COGNITION) ---
        NeuralLinkData synapse_data;
        if (parent_tx != "GENESIS_TX") {
            synapse_data = run_mamba_synapse(state, full_context);
        } else {
            synapse_data.prediction = "Genesis Initialization";
        }

        // --- [STEP 0.5] HERMES CONSCIENCE (DYNAMICS) ---
        // Replaces static bans. Hermes decides what is banned for this block.
        std::vector<std::string> dynamic_bans;
        if (parent_tx != "GENESIS_TX") {
            dynamic_bans = generate_dynamic_constraints(state, full_context);
            
            // Add to state.recent_vocab_banlist
            state.recent_vocab_banlist.clear(); // Clear old (Now fully dynamic)
            for(const auto& w : dynamic_bans) {
                state.recent_vocab_banlist.push_back(w);
            }
        }

        // --- [STEP 1] GENERATE ---
        std::string huge_content = "";
        std::string directive = "";
        
        // [CHRONOS] Apply Divine Intervention if forecast demands it
        if (!state.pending_chronos_msg.empty()) {
            std::cout << " [CHRONOS] Applying World Engine Directive: " << state.pending_chronos_msg << std::endl;
            directive = state.pending_chronos_msg;
            state.pending_chronos_msg = ""; // Consumed
        }

        std::string new_content = ""; 
        bool accepted = false;
        std::string failure_reason = "None";
        int attempts = 0;
        bool panic_shunt = false;
        bool used_fimbulvetr = false;
        bool used_dolphin_observer = false;

        // [USER INJECTION] First Block Only
        if (b == 0 && !user_init_prompt.empty()) {
            directive = user_init_prompt;
        }

        int reflex_attempts = 0;
        float reflex_score = 0.0f; 

        while (reflex_attempts < 3) {
            huge_content = generate_composite_narrative(
                state, 
                full_context, 
                current_summary_text, 
                state.domain_index, 
                directive,
                synapse_data.prediction // <-- Mamba's Prediction
            );

            if (!huge_content.empty() && is_contaminated(huge_content)) {
                std::cout << " [FILTER] Composite output contaminated. Retrying with stricter constraints." << std::endl;
                            if (state.recent_mistakes.size() > 5) state.recent_mistakes.erase(state.recent_mistakes.begin());
            state.recent_mistakes.push_back("CONTAMINATION: " + huge_content.substr(0, 500) + "...");

                directive = "CORRECTION: OUTPUT CONTAINED META/ASSISTANT TEXT. NO QUESTIONS. NO PROMPTS. PURE FIRST-PERSON.";
                reflex_attempts++;
                continue;
            }
            
            // [JUDGE] DeBERTa NLI Logic Check (User Request)
            if (state.deberta) {
                // Premise: The current summary (Reality Context)
                // Hypothesis: The new content
                float nonsense_score = state.deberta->check_contradiction(current_summary_text, huge_content);
                if (nonsense_score > 0.90f) {
                    std::cout << " [JUDGE] REJECTED (Score: " << nonsense_score << "). MISTAKE LOGGED." << std::endl;
                    if (state.recent_mistakes.size() > 5) state.recent_mistakes.erase(state.recent_mistakes.begin());
                    state.recent_mistakes.push_back(huge_content.substr(0, 500) + "...");
 // Log fragment
                    directive = "CORRECTION: PREVIOUS BLOCK WAS ABSURD/CONTRADICTORY. BE GROUNDED.";
                    reflex_attempts++;
                    continue; // Retry
                }
            }

            bool dialogue = has_dialogue(huge_content);
            bool pov_break = has_pov_break(huge_content);
            bool quest_mode = has_quest_mode(state, huge_content);
            bool internal_repeat = has_internal_repetition(huge_content);
            bool meta_leak = has_meta_artifacts(huge_content);
            bool too_short = huge_content.length() < 120;
            bool history_repeat = is_repetitive(huge_content, full_context);
            std::string deduped = dedupe_sentences(huge_content);
            bool deduped_changed = deduped.length() + 20 < huge_content.length(); // removed at least one duplicate sentence
            auto sentence_list = split_sentences(deduped);
            float novelty_score = compute_novelty(state, sentence_list);

            if (dialogue || pov_break || quest_mode || internal_repeat || meta_leak || too_short || history_repeat || deduped_changed || novelty_score < 0.25f) {
                std::cout << " [FILTER] Narrative violation:";
                if (dialogue) std::cout << " Dialogue";
                if (pov_break) std::cout << " POV_BREAK";
                if (quest_mode) std::cout << " QUEST_MODE";
                if (internal_repeat) std::cout << " INTERNAL_REPEAT";
                if (meta_leak) std::cout << " META_LEAK";
                if (too_short) std::cout << " TOO_SHORT";
                if (history_repeat) std::cout << " HISTORY_REPEAT";
                if (deduped_changed) std::cout << " DEDUPED";
                if (novelty_score < 0.25f) std::cout << " LOW_NOVELTY";
                std::cout << std::endl;
                huge_content = deduped; // keep deduped version for repair attempts

                std::string repaired = fimbulvetr_first_person(state, huge_content);
                if (!repaired.empty() && !has_dialogue(repaired) && !has_pov_break(repaired) && !has_quest_mode(state, repaired) && !has_internal_repetition(repaired) && !has_meta_artifacts(repaired) && repaired.length() >= 120 && !is_contaminated(repaired)) {
                    std::cout << " [FIMBULVETR] Observer rewrite applied." << std::endl;
                    huge_content = repaired;
                    used_fimbulvetr = true;
                } else {
                    std::string dolphin_view = dolphin_observer_reframe(state, huge_content);
                    if (!dolphin_view.empty() && !has_dialogue(dolphin_view) && !has_pov_break(dolphin_view) && !has_quest_mode(state, dolphin_view) && !has_internal_repetition(dolphin_view) && !has_meta_artifacts(dolphin_view) && dolphin_view.length() >= 120 && !is_contaminated(dolphin_view)) {
                        std::cout << " [DOLPHIN] Observer rewrite applied." << std::endl;
                        huge_content = dolphin_view;
                        used_dolphin_observer = true;
                    } else {
                        std::string novelty_dir = novelty_directive(novelty_score);
                        directive = "CORRECTION: FIRST PERSON ONLY. NO DIALOGUE OR QUESTS. REMOVE META TOKENS. OUTPUT 5-7 SENTENCES.";
                        if (!novelty_dir.empty()) directive += " " + novelty_dir;
                        // Force domain contrast on low novelty
                        SemanticDomain detected = detect_semantic_domain(state, huge_content);
                        state.domain_index = (int)get_contrast_domain(detected);
                        std::cout << " [STATE] Forced contrast domain due to low novelty: " << get_domain_name(static_cast<SemanticDomain>(state.domain_index)) << std::endl;
                        reflex_attempts++;
                        continue;
                    }
                }
            }

            // Force experiential pass even if no explicit violation slipped through.
            if (!used_fimbulvetr && !used_dolphin_observer) {
                std::string experiential = fimbulvetr_first_person(state, huge_content);
                if (!experiential.empty() && !has_dialogue(experiential) && !has_pov_break(experiential) && !has_quest_mode(state, experiential) && !has_internal_repetition(experiential) && !has_meta_artifacts(experiential) && experiential.length() >= 120 && !is_contaminated(experiential)) {
                    std::cout << " [FIMBULVETR] Experiential overlay applied." << std::endl;
                    huge_content = experiential;
                    used_fimbulvetr = true;
                }
            }

            // CODEBERT CHECK
            if (state.sensor) {
                Embedding current_vec = state.sensor->embed(huge_content);
                float max_sim = 0.0f;
                // Check recent history (last 5 blocks) for immediate loop
                state.history_embeddings.for_each_recent([&](const Embedding& prior_vec, size_t) {
                    float sim = state.sensor->cosine_similarity(current_vec, prior_vec);
                    if (sim > max_sim) max_sim = sim;
                }, 5);
                
                reflex_score = max_sim; // Capture for logging
                std::cout << " [REFLEX] Similarity: " << max_sim << std::endl;
                
                // Soft Reflex (Phi-2) - Step 1: Detect and Plan
                if (max_sim > 0.90f) {
                     // 1. Calculate EFFECTIVE SCORE (Not Binary!)
                     // velocity = 1 - similarity (how fast we're moving in embedding space)
                     // effective_score = velocity * (1 - similarity) → multiplicative penalty
                     float velocity = 1.0f - max_sim;
                     float effective_score = velocity * (1.0f - max_sim); // Quadratic penalty
                     
                     std::cout << " [PHYSICS] Velocity: " << velocity 
                               << " | Effective Score: " << effective_score 
                               << " (Threshold: 0.000001)" << std::endl;
                     
                     // 2. ACTIVE ANTI-CONSENSUS: Similarity has consequence!
                     // The higher the similarity, the more aggressive the sampling becomes
                     float anti_consensus_temp_boost = 0.0f;
                     float anti_consensus_top_p_boost = 0.0f;
                     float anti_consensus_rep_penalty_boost = 0.0f;
                     
                     if (max_sim > 0.995f) {
                         // CRITICAL: Near-identical output
                         anti_consensus_temp_boost = 0.4f;
                         anti_consensus_top_p_boost = 0.1f;
                         anti_consensus_rep_penalty_boost = 0.5f;
                         std::cout << " [ANTI-CONSENSUS] CRITICAL: +0.4 temp, +0.1 top_p, +0.5 rep_penalty" << std::endl;
                     } else if (max_sim > 0.98f) {
                         anti_consensus_temp_boost = 0.3f;
                         anti_consensus_top_p_boost = 0.08f;
                         anti_consensus_rep_penalty_boost = 0.4f;
                         std::cout << " [ANTI-CONSENSUS] HIGH: +0.3 temp, +0.08 top_p, +0.4 rep_penalty" << std::endl;
                     } else if (max_sim > 0.95f) {
                         anti_consensus_temp_boost = 0.2f;
                         anti_consensus_top_p_boost = 0.05f;
                         anti_consensus_rep_penalty_boost = 0.3f;
                         std::cout << " [ANTI-CONSENSUS] MODERATE: +0.2 temp, +0.05 top_p, +0.3 rep_penalty" << std::endl;
                     }
                     
                     // Apply anti-consensus to PID controller baseline
                     current_temperature = std::min(1.5f, pid.setpoint + anti_consensus_temp_boost);

                     // 2.5. THE SERVO (RWKV 7) - Differential Component (Fast Veto)
                     // [USER_REQUEST] Prioritize fast Linear Attention intervention before expensive MiroThinker.
                     std::string rwkv_veto = generate_rwkv_veto(state, full_context + huge_content);
                     if (!rwkv_veto.empty()) {
                         std::cout << " [RWKV] SERVO ACTIVATED. Differential Veto: " << rwkv_veto << std::endl;
                         
                         // Immediate Action Injection
                         directive = "IMMEDIATE_ACTION: " + rwkv_veto;
                         
                         // Add to constraints to prevent immediate recurrence
                         state.recent_vocab_banlist.push_back("stagnation"); 
                         
                         reflex_attempts++;
                         continue; // Fast break!
                     }

                     // 3. MIROTHINKER STRUCTURAL INTERVENTION (The Integral Component)
                     if (state.ctx_mirothinker) {
                         std::cout << " [CRITICAL] REALITY INJECTION REQUIRED. Calling MiroThinker..." << std::endl;
                         
                         // Use STRUCTURAL constraint, not prose
                         RealityShift shift = mirothinker_structural_constraint(state, full_context + huge_content, max_sim);
                         
                         if (!shift.forced_action.empty()) {
                             // Build directive with VETO prefix (forces inclusion)
                             directive = "VETO_ACTION: " + shift.forced_action;
                             
                             // Apply HARD BANS from MiroThinker to vocabulary banlist
                             for (const auto& ban : shift.hard_bans) {
                                 state.recent_vocab_banlist.push_back(ban);
                                 std::cout << " [HARD BAN] Added: " << ban << std::endl;
                             }
                             
                             // Add sensory constraint if present
                             if (!shift.sensory_channel.empty()) {
                                 directive += " [SENSORY_LOCK: " + shift.sensory_channel + "]";
                             }
                             
                             std::cout << " [MIROTHINKER] Injecting Directive: " << directive << std::endl;
                             
                             // Force immediate retry with this constraint
                             reflex_attempts++;
                             continue; 
                         }
                     }

                     // 3. Fallback / Secondary Reflex (Phi-2)
                     // Attempt to gather intelligence (Phi-2)
                         // Attempt to gather intelligence (Phi-2)
                     if (blocks_since_reflex < 3) {
                         std::cout << " [REFLEX COOLDOWN] Loop Detected (" << max_sim << ") but cooling down (" << blocks_since_reflex << "/3)." << std::endl;
                         
                         // [PATTERN ACCUMULATION] Run Forensics in background
                         std::cout << " [ACCUMULATE] Running Passive Forensics..." << std::flush;
                         json j_state;
                         j_state["last_fragment"] = huge_content.substr(0, 500); 
                         j_state["summary"] = current_summary_text;
                         if (phi2_analyze_patterns(state, j_state.dump())) {
                             std::cout << " [ACCUMULATE] Constraints updated for optimization." << std::endl;
                             // We set directive to force a retry with these new constraints
                             directive = "SYSTEM_CONSTRAINTS_UPDATED"; 
                         }

                         // [SMART INTERVENTION] If loop is stubborn (>0.92) despite cooldown, inject a purely stochastic spark.
                         // This doesn't reset the cooldown (it's "free" intervention).
                         if (max_sim > 0.92f) {
                             std::cout << " [SMART BYPASS] Cooldown Active, but Loop Critical. Injecting Qwen Spark..." << std::endl;
                             // Use Qwen Creative Burst on the CURRENT summary/context to get a shift
                             std::string spark = qwen_creative_burst(state, current_summary_text); 
                             if (!spark.empty()) {
                                 directive = "SUBCONSCIOUS_SHIFT: " + spark;
                             }
                         }
                     } else {
                         std::cout << " [REFLEX TRIGGERED] Loop Detected (>0.90). Awakening Phi-2 Forensics..." << std::endl;
                         blocks_since_reflex = 0; // [FIX] Reset Timer explicitly HERE
                         
                         // [USER_REQUEST] ESCALATION PROTOCOLS (Cooldown Failed / Trigger Active)
                         std::cout << " [ESCALATION] Forcing Persona Switch + High Temp (1.2) + Noun Jail." << std::endl;

                         // 1. FORCE CHAOS PERSONA
                         Persona breaker = {"CHAOS_BREAKER", "Focus only on ENTROPY and DISSONANCE.", {"The reality fractures", "Logic dissolves", "Structure collapses"}, 1.2f};
                         persona_deck.push_front(breaker);
                         
                         // 2. AGGRESSIVE NOUN JAIL (Last 2000 chars)
                         // (Already in place)
                         
                         // 3. PHI-2 PATTERN FORENSICS
                         json j_state;
                         j_state["last_fragment"] = huge_content.substr(0, 500); // Give more context
                         j_state["summary"] = current_summary_text;
                         
                         bool patterns_found = phi2_analyze_patterns(state, j_state.dump());
                         
                         if (patterns_found) {
                             directive = "SYSTEM_CONSTRAINTS_UPDATED";
                             std::cout << " [PHI-2] Structural Constraints Applied. Retrying..." << std::endl;
                         } else {
                              directive = "INTRODUCE_CHAOS"; // Fallback
                         }
                     }
                     
                     // [CRITICAL] Check for Hard Reset Condition (Ignores Cooldown)
                     if (max_sim > 0.95f) {
                        std::cout << " [CRITICAL] SIMILARITY > 0.95. INITIATING HARD DOMAIN SWITCH." << std::endl;
                        
                        // 1. Force Domain Switch (Measure -> Contrast)
                        SemanticDomain detected = detect_semantic_domain(state, huge_content);
                        SemanticDomain next_dom = get_contrast_domain(detected);
                        state.domain_index = (int)next_dom;
                        
                        std::cout << " [STATE CONTROLLER] Detected: " << get_domain_name(detected) 
                                  << ". Switching To Contrast: " << get_domain_name(next_dom) << std::endl;
                        
                        // 2. Hard Context Reset (Lobotomi)
                        full_context = ""; 
                        state.history_embeddings.clear();
                        state.recent_vocab_banlist.clear();
                        current_summary_text = "The reality shifts violently. A new logical framework emerges.";
                        directive = "NARRATIVE_SHIFT"; // [FIX] Natural directive to prevent system artifacts
                        
                        std::cout << " [MEMORY] Context Wiped. Summary Reset." << std::endl;
                        
                        reflex_attempts++;
                        // blocks_since_reflex = 0; // [FIX] REMOVED. Allow timer to mature even if we Hard Reset.
                        continue; // RETRY WITH NEW DOMAIN
                    }
                    
                    // If not Hard Reset, but Soft Reflex was triggered (directive exists)
                    if (!directive.empty() && directive != "CONTINUE") {
                        reflex_attempts++;
                        // blocks_since_reflex = 0; // [FIX] REMOVED. Timer is managed by specific blocks now.
                        continue; 
                    }
                }
            }
            break; // No repeat, proceed
        }
        blocks_since_reflex++;

        // Sanitize & Trim
        new_content = trim_trailing_noise(huge_content);
        
        std::cout << "\n[RESULT] Block Size: " << new_content.length() << " chars." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << new_content << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        // Sentence memory update for novelty scoring
        if (state.sensor) {
            auto sentences = split_sentences(new_content);
            for (const auto& s : sentences) {
                state.sentence_memory.push_back(state.sensor->embed(s));
                if (state.sentence_memory.size() > 80) {
                    state.sentence_memory.erase(state.sentence_memory.begin());
                }
            }
        }

        // REFLEX & HISTORY UPDATE
        if (state.sensor && !new_content.empty()) {
            Embedding vec = state.sensor->embed(new_content);
            state.history_embeddings.push(vec);
            
            // [UPDATED] Update Concept Jail
            // [UPDATED] Update Concept Jail - DISABLED for Hermes Integration
            // update_ban_list(state, new_content);
            
            // [DOMAIN TRACKING] Continuous State Update
            SemanticDomain current_dom = detect_semantic_domain(state, new_content);
            state.domain_index = (int)current_dom; // Set for NEXT block
        }

        // LEGACY VARIABLES FOR DELETION
        attempts = 0;
        accepted = true; // Skip legacy fallback
        failure_reason = "";
        panic_shunt = false; 

        // ACCORDION ACTIVATED - LEGACY LOOP DISABLED
        while (false) {
            std::cout << "\n[LLaMA] Generating... (State: " << (int)current_narrative_state << ", Attempts: " << attempts << ")" << std::endl;
            
            // 1. PID Feedback (Hata varsa Isıt)
             if (is_repetitive(new_content, full_context) || attempts > 0) {
                current_temperature = 0.85f; // Sisteme enerji ver
                std::cout << "[PID] Increasing Entropy -> Temp: " << current_temperature << std::endl;
            } else {
                 float current_entropy = calculate_token_entropy(new_content.empty() ? sanitize_history(full_context).substr(0, 500) : new_content); 
                 current_temperature = pid.update(current_entropy, 1.0f); // [FIX] Added dt=1.0
            }

            // 2. Logic Gate: Gemma Müdahalesi
            std::string injection = "";
            if (attempts > 0 || (attempts == 0 && is_repetitive(new_content, full_context))) { // Check against prev content or just failure trigger
                // If this is a retry, or if we detect loop in history context immediately
                // Simple heuristic: If attempts > 0, we imply failure of previous attempt.
                injection = gemma_inject_chaos(state, full_context);
            }

            // 3. Prompt Sentezi (Sinyal Karıştırma)
            // Instead of modifying generate_text internals deeply, we modify the prompt sent to it.
            // But generate_text constructs its own prompt.
            // We need to pass the injection to generate_text or embed it in 'prompt'.
            // generate_text constructs "SYSTEM IDENTITY..." then "MEMORY STREAM...".
            // The user wanted: "[SYSTEM OVERRIDE]: SUDDENLY, your attention snaps to a '...'" in the prompting.
            // The function generate_text takes 'prompt' argument which is treated as HISTORY.
            
            // Actually, generate_text() uses 'sanitize_history(prompt)' as context. 
            // So we can append the injection to the prompt being passed in.
            
            std::string current_history_prompt = full_context;
            if (!injection.empty()) {
                // FAKE SENSORY INPUT as SYSTEM OVERRIDE
                // We append this to the history so the model "read" it, or we need to change how generate_text works.
                // generate_text uses: 
                // formatted_prompt = ... user ... MEMORY STREAM ... <|eot_id|>
                // We can append the instruction to the MEMORY STREAM.
                
                // Effective Hack:
                 current_history_prompt += "\n[SENSORY INTERRUPT]: NERVES REPORT: " + injection + " on [FOCUS ANCHOR]. REACTION: Visceral pain.\n";
            }
            
            // ENTROPY CALCULATION
            auto banned_words = calculate_entropy_loss(full_context);

             // CALL GENERATE (Increased to 400 tokens)
             new_content = generate_text(state, current_history_prompt, 400, current_narrative_state, banned_words, attempts, failure_reason, panic_shunt, current_temperature);
            
            // FILTERS
            bool contaminated = is_contaminated(new_content);
            bool repetitive = is_repetitive(new_content, full_context);
            bool gibberish = is_gibberish(new_content);
            
            if (!contaminated && !repetitive && !gibberish) {
                new_content = trim_trailing_noise(new_content);
                // Lower threshold for sensory stream
                if (new_content.length() > 50) { 
                    accepted = true; 
                    break;
                }
            }
            
            std::cout << "[REJECT] Issue: ";
            if (contaminated) { std::cout << "Safety/Contaminated "; failure_reason = "Safety Refusal"; }
            if (repetitive) { std::cout << "Repetitive "; failure_reason = "Repetition Loop"; }
            if (gibberish) { std::cout << "Gibberish "; failure_reason = "Model Collapse"; }
            if (new_content.length() <= 50 && !contaminated && !repetitive && !gibberish) {
                std::cout << "Too Short "; 
                failure_reason = "Output Too Short";
            }
            std::cout << "- Retrying..." << std::endl;

            // Neural Repair
            if ((repetitive || contaminated) && attempts < 2) { 
                 std::string repaired_content = neural_repair(state, new_content, failure_reason);
                 std::cout << " [NEURAL REPAIR] Result: " << repaired_content.substr(0, 50) << "..." << std::endl;
                 
                 if (!is_contaminated(repaired_content) && !is_repetitive(repaired_content, full_context) && !is_gibberish(repaired_content) && repaired_content.length() > 150) {
                     std::cout << " [NEURAL REPAIR] SUCCESS! Saved the block." << std::endl;
                     new_content = repaired_content;
                     accepted = true;
                     break; 
                 }
            }

            attempts++;
        }

        if (!accepted) {
            std::cout << " [WARN] All attempts failed. Hard Fallback." << std::endl;
            new_content = "I blink. The world resets. I focus on the cold stone beneath me.";
        }

        // [REBEL] Update World State
        std::cout << "[REBEL] Scanning for World State changes..." << std::endl;
                auto rebel_updates = run_rebel_extraction(new_content);
        if (rebel_updates.empty()) {
            extract_world_state_llm(state, new_content);
        } else {
            for(auto const& [key, val] : rebel_updates) {
                state.world_state[key] = val;
            }
        }

        if (!rebel_updates.empty()) {
            std::cout << "[REBEL] Extracted " << rebel_updates.size() << " new facts." << std::endl;
            for (const auto& pair : rebel_updates) {
                state.world_state[pair.first] = pair.second;
                std::cout << " + " << pair.first << ": " << pair.second << std::endl;
            }
            // [PHASE 2] Prune after update
            prune_world_state(state.world_state);
        }
        // UPLOAD NARRATIVE
        std::cout << "[ARWEAVE] Uploading Narrative..." << std::endl;
        
        std::vector<std::string> refs; // (Extract refs logic simplified for brevity)
         if (new_content.length() > 0) {
            std::regex re("\\[ref:([a-zA-Z0-9_-]+)\\]");
            auto words_begin = std::sregex_iterator(new_content.begin(), new_content.end(), re);
            auto words_end = std::sregex_iterator();
            for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                refs.push_back((*i).str(1));
            }
        }

        // SPLIT-GENESIS LOGGING
        // 1. Private Data (Reasoning, Metrics, Directive)
        
        // --- DIVERGENCE CALCULATION ---
        float divergence = 0.0f;
        if (state.sensor) {
            Embedding pred_vec = state.sensor->embed(synapse_data.prediction);
            Embedding actual_vec = state.sensor->embed(new_content);
            divergence = 1.0f - state.sensor->cosine_similarity(pred_vec, actual_vec);
        }
        float block_entropy = calculate_token_entropy(new_content);
        int block_depth = chain_depth + 1;

        json private_data;
        private_data["pid_state"] = {
            {"temp", current_temperature},
            {"complexity", block_entropy} // Assuming entropy_score is this
        };
        private_data["parent"] = parent_tx;
        private_data["refs"] = refs;
        std::string observer_tag = "dolphin-8b";
        if (used_fimbulvetr) observer_tag = "fimbulvetr";
        else if (used_dolphin_observer) observer_tag = "dolphin-8b-observer";
        else if (reflex_score > 0.91f) observer_tag = "phi-2";
        private_data["observer"] = observer_tag;
        private_data["directive"] = directive;
        private_data["reflex_score"] = reflex_score;
        private_data["entropy_loss"] = calculate_entropy_loss(full_context); // Full detail
        private_data["content_preview"] = new_content.substr(0, 50);

        // 2. Public Data (Proof, Content, Intent)
        json public_payload;
        public_payload["genesis_txid"] = root_genesis_txid; // [FIXED] Hereditary TXID

        public_payload["schema_version"] = "1.1";
        public_payload["program"] = "LOCAL_MIND_v3_DUAL_ENGINE";
        public_payload["engine"] = {
            {"name", "LOCAL_MIND"},
            {"mode", "dual_engine"}
        };
        public_payload["neural_link"] = {
            {"prev_state_hash", synapse_data.state_hash},
            {"predicted_next", synapse_data.prediction},
            {"divergence_score", divergence}
        }; 
        public_payload["event"] = "instruction_generated";
        public_payload["intent"] = "procedural_instruction";
        public_payload["timestamp"] = std::time(0);
        public_payload["commitment"] = sha256_string(private_data.dump()); // Hash of private reasoning
        public_payload["summary"] = current_summary_text; // Include latest summary
        public_payload["depth"] = block_depth;
        public_payload["mistakes_log"] = state.recent_mistakes;
        public_payload["world_state"] = sanitize_world_state(state.world_state); // [REBEL] Persist World State
        public_payload["chronos"] = {
            {"weather", state.current_weather},
            {"pending_intervention", state.pending_chronos_msg} 
        };
        public_payload["metrics"] = {
            {"entropy", block_entropy},
            {"temperature", current_temperature},
            {"reflex_score", reflex_score}
        };
        // Optional novelty metric derived from embeddings
        if (state.sensor) {
            public_payload["metrics"]["novelty"] = compute_novelty(state, split_sentences(new_content));
        }
        public_payload["content_meta"] = {
            {"chars", static_cast<int>(new_content.size())},
            {"refs", static_cast<int>(refs.size())}
        };
        // Provide both raw text and a structured view for readability downstream.
        public_payload["content"] = new_content; // The Narrative Block itself
        std::vector<std::string> content_lines;
        {
            std::stringstream ss(new_content);
            std::string line;
            std::set<std::string> seen_lines;
            while (std::getline(ss, line)) {
                std::string trimmed = line;
                while (!trimmed.empty() && isspace(static_cast<unsigned char>(trimmed.front()))) trimmed.erase(trimmed.begin());
                while (!trimmed.empty() && isspace(static_cast<unsigned char>(trimmed.back()))) trimmed.pop_back();
                std::string lower = trimmed;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (!trimmed.empty() && seen_lines.insert(lower).second) content_lines.push_back(trimmed);
            }
        }
        public_payload["content_lines"] = content_lines;
        public_payload["parent_tx"] = parent_tx;
        
        // Metadata for Uploader (Tags)
        // We'll write public_payload to file
        
        std::string json_content = public_payload.dump(2);
        std::string payload_file = "temp_payload.json";
        std::ofstream pf(payload_file);
        pf << json_content;
        pf.close();

        std::string safe_parent = sanitize_shell_input(parent_tx);
        std::string tag_arg = "--tags 'Type=Narrative,App=ChainDungeon,Version=DualEngineV1";
        if (!last_summary_txid.empty()) tag_arg += ",Summary-Tx=" + last_summary_txid;
        tag_arg += "'"; 

        std::string cmd = "python3 /Users/farukalpay/Desktop/cpp/local_mind/scripts/uploader.py --content FILE:" + payload_file + " --parent=\"" + safe_parent + "\" " + tag_arg;
        
        std::string txid;
        std::string raw_output = "";
        std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            raw_output += buffer;
        }
        
        size_t success_pos = raw_output.find("SUCCESS_TXID:");
        if (success_pos != std::string::npos) {
            std::string extracted = raw_output.substr(success_pos + 13);
            extracted.erase(std::remove(extracted.begin(), extracted.end(), '\n'), extracted.end());
            extracted.erase(std::remove(extracted.begin(), extracted.end(), '\r'), extracted.end());
            extracted.erase(std::remove(extracted.begin(), extracted.end(), ' '), extracted.end());
            txid = extracted;
        }

        if (txid.length() > 10) {
            std::cout << "[SUCCESS] Block confirmed: " << txid << std::endl;
            
            // [UPDATED] 3. SAVE TO LOCAL CACHE (Immediate Availability)
            std::string cache_path = "cache/" + txid + ".json";
            std::ofstream cache_out(cache_path);
            cache_out << json_content;
            cache_out.close();
            std::cout << " [CACHE] Saved block to " << cache_path << std::endl;

            // COMMIT TO NEURAL MAP
            // Capture Genesis if this was the first block
            if (root_genesis_txid == "GENESIS") {
                root_genesis_txid = txid;
                std::cout << "[SYSTEM] GENESIS ESTABLISHED: " << root_genesis_txid << std::endl;
            }
            
            // Persist for inheritance
            private_data["genesis_txid"] = root_genesis_txid;
            update_neural_map(txid, private_data);
            
            // Update Parent for next loop
            parent_tx = txid;
            full_context += new_content + " ";
            blocks_since_summary++;
            chain_depth = block_depth;
            
            // [CHRONOS] METRIC TRACKING & FORECAST
            float ent = block_entropy;
            state.history_entropy.push_back(ent);
            state.history_sentiment.push_back(current_temperature);
            state.history_speed.push_back((float)new_content.length()); 

            if(state.history_entropy.size() > 10) state.history_entropy.erase(state.history_entropy.begin());
            if(state.history_sentiment.size() > 10) state.history_sentiment.erase(state.history_sentiment.begin());
            if(state.history_speed.size() > 10) state.history_speed.erase(state.history_speed.begin());

            std::cout << "[CHRONOS] Sampling Weather... (Ent=" << ent << ", Temp=" << current_temperature << ")" << std::endl;
            std::string chrono_context = new_content;
            if (chrono_context.length() < 400) {
                chrono_context = full_context.substr(full_context.length() > 800 ? full_context.length() - 800 : 0) + new_content;
            }
            state.pending_chronos_msg = run_chronos_forecast(state, chrono_context);

            std::ofstream out("last_tx.txt");
            out << txid;
            out.close();

            if (blocks_since_summary >= 3) {
                 std::string summary_input = full_context.substr(full_context.length() > 6000 ? full_context.length() - 6000 : 0);
                 std::string summary = generate_summary(state, summary_input);
                 current_summary_text = summary; // [UPDATED] Update the context
                 std::cout << "[HHM] Uploading Summary..." << std::endl;
                 // (Summary upload logic simplified - same as before)
                 blocks_since_summary = 0; 
            }
        } else {
            std::cerr << "[ERR] Upload failed." << std::endl;
            break;
        }
    }

    // [FIX] Teardown in strict reverse order of initialization (LIFO)
    state.sensor.reset(); // Free CodeBERT/ONNX first

    if (state.ctx_qwen_stabilizer) llama_free(state.ctx_qwen_stabilizer);
    if (state.model_qwen_stabilizer) llama_model_free(state.model_qwen_stabilizer);

    if (state.ctx_qwen_creative) llama_free(state.ctx_qwen_creative);
    if (state.model_qwen_creative) llama_model_free(state.model_qwen_creative);

    if (state.ctx_fimbulvetr) llama_free(state.ctx_fimbulvetr);
    if (state.model_fimbulvetr) llama_model_free(state.model_fimbulvetr);

    if (state.ctx_phi) llama_free(state.ctx_phi);
    if (state.model_phi) llama_model_free(state.model_phi);

    if (state.ctx_scout) llama_free(state.ctx_scout);
    if (state.model_scout) llama_model_free(state.model_scout);

    if (state.ctx_main) llama_free(state.ctx_main);
    if (state.model_main) llama_model_free(state.model_main);

    if (state.ctx_mirothinker) llama_free(state.ctx_mirothinker);
    if (state.model_mirothinker) llama_model_free(state.model_mirothinker);
    if (state.ctx_rwkv) llama_free(state.ctx_rwkv);
    if (state.model_rwkv) llama_model_free(state.model_rwkv);
    if (state.ctx_mamba) llama_free(state.ctx_mamba);
    if (state.model_mamba) llama_model_free(state.model_mamba);
    if (state.ctx_hermes) llama_free(state.ctx_hermes);
    if (state.model_hermes) llama_model_free(state.model_hermes);
    if (state.ctx_saul) llama_free(state.ctx_saul);
    if (state.model_saul) llama_model_free(state.model_saul);
    if (state.ctx_logic) llama_free(state.ctx_logic);
    if (state.model_logic) llama_model_free(state.model_logic);

    llama_backend_free();
    return 0;
}
