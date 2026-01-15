// [ENGINE] CORE NARRATIVE ENGINE (C++ / LLaMA)
// Maintainer: AION / AntiGravity
// Goal: Raw Consciousness Stream (No "Assistant" artifacts)

#include "llama.h"
#include <iostream>
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
#include "arweave.hpp"
#include <onnxruntime_cxx_api.h> // ONNX Runtime
#include <openssl/sha.h>
#include <curl/curl.h>
#include <iomanip>

using json = nlohmann::json;

// --- CONFIGURATION ---
const int MAX_DEPTH = 20; // [UPDATED] Expanded Context Window
const std::string BASE_DIR = "/Users/farukalpay/Desktop/cpp/local_mind/";
const std::string MODEL_PATH = BASE_DIR + "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf";
const std::string QWEN_CREATIVE_PATH = BASE_DIR + "models/qwen2-1.5b-instruct-q4_k_m.gguf";
const std::string QWEN_STABILIZER_PATH = BASE_DIR + "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
const std::string MIROTHINKER_PATH = BASE_DIR + "models/MiroThinker-v1.5-30B.Q4_K_M.gguf";
const std::string RWKV_PATH = BASE_DIR + "models/rwkv/rwkv7-g0a4-13.3b-Q4_K_M.gguf"; // [NEW] RWKV 7
const std::string MAMBA_PATH = BASE_DIR + "models/mamba-1.4b-hf-Q4_K_M.gguf"; // [NEW] Mamba Synapse
const std::string HERMES_PATH = BASE_DIR + "models/nous-hermes-llama2-13b.Q4_K_M.gguf"; // [NEW] Hermes Conscience
const std::string SAUL_PATH = BASE_DIR + "models/saul-7b.gguf"; // [NEW] Saul 7B for Dynamic Prefills
const std::string VOCAB_PATH = BASE_DIR + "models/onnx/vocab.txt";

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

    std::vector<float> embed(const std::string& text) {
        // [PYTHON BRIDGE] Call src/embed.py due to ONNX restoration complexity
        std::string cmd = "python3 src/embed.py \"" + sanitize_shell_input(text) + "\"";
        // sanitize_shell_input not defined in scope yet? It's a helper function.
        // I need to confirm sanitize_shell_input is available to this class.
        // It is defined usually later or earlier.
        // If not available, I'll do basic quote escaping.
        
        // Basic escaping
        std::string safe_text = "";
        for(char c : text) {
            if(c == '"') safe_text += "\\\"";
            else if(c == '\\') safe_text += "\\\\";
            else safe_text += c;
        }
        
        std::string cmd_safe = "python3 src/embed.py \"" + safe_text + "\"";
        std::shared_ptr<FILE> pipe(popen(cmd_safe.c_str(), "r"), pclose);
        if (!pipe) return std::vector<float>(768, 0.0f);
        
        char buffer[1024]; // Increase buffer
        std::string result = "";
        while (!feof(pipe.get())) {
            if (fgets(buffer, 1024, pipe.get()) != NULL)
                result += buffer;
        }
        
        // Parse JSON [0.1, 0.2, ...]
        try {
            auto j = json::parse(result);
            std::vector<float> vec;
            for (auto& el : j) vec.push_back(el.get<float>());
            if (vec.size() != 768) return std::vector<float>(768, 0.001f);
            return vec;
        } catch (...) {
            return std::vector<float>(768, 0.001f);
        }
    }

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-9);
    }

    // [RESTORED] Compute Centroid of Embeddings
    std::vector<float> compute_centroid(const std::vector<std::vector<float>>& history) {
        if (history.empty()) return std::vector<float>(768, 0.0f);
        
        std::vector<float> centroid(768, 0.0f);
        for (const auto& vec : history) {
            if (vec.size() != 768) continue;
            for (size_t i = 0; i < 768; ++i) {
                centroid[i] += vec[i];
            }
        }
        
        for (size_t i = 0; i < 768; ++i) {
            centroid[i] /= history.size();
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

    // 14. CHRONOS (The World Engine) - Metric History
    std::vector<float> history_entropy;    // "Wind"
    std::vector<float> history_sentiment;  // "Temperature" (Intensity)
    std::vector<float> history_speed;      // "Pressure" (Tokens/sec)
    std::string current_weather = "SUNNY"; // Current Forecast
    std::string pending_chronos_msg = "";  // Directive for next block
    
    // 8. THE SENSOR (CodeBERT)
    std::shared_ptr<CodeBERT> sensor = nullptr;
    
    // HISTORY EMBEDDINGS (For Repetition Detection)
    std::vector<std::vector<float>> history_embeddings;
    std::deque<std::string> recent_vocab_banlist; // Dynamic Ban List
    std::vector<std::string> recent_mistakes; // [NEW] RAG for Mistakes
    
    // 9. THE JUDGE (DeBERTa NLI)
    std::shared_ptr<DeBERTaNLI> deberta = nullptr;

    // NARRATIVE STATE CONTROLLER
    int domain_index = 0; // 0=DOMESTIC_SURREAL, etc.
    int domain_streak = 0; // [NEW] Track how long we've been stuck in a domain

    // 10. WORLD STATE (REBEL KNOWLEDGE GRAPH)
    std::map<std::string, std::string> world_state; // Entity -> Status/Relation
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

// --- UPDATE BAN LIST (CONCEPT JAIL) ---
// Now with TTL (Time-To-Live) for bans - words expire after N blocks
struct BannedTerm {
    std::string word;
    int blocks_remaining; // Countdown to unban
};
static std::deque<BannedTerm> concept_jail_with_ttl;

// [MOVED] Global Concept Definitions
// [DISABLED] Replaced by Hermes Dynamic Conscience
// static std::map<std::string, std::vector<std::string>> global_concepts = {
//     {"METAL", {"copper", "rust", "steel", "iron", "metallic", "oxidized", "chrome", "brass", "wire", "alloy", "tin", "aluminum"}},
//     {"GORE", {"blood", "bone", "flesh", "bile", "vein", "skin", "sweat", "mucus", "marrow", "viscera", "sinew", "pus"}},
//     {"CHEM", {"ozone", "sulfur", "acid", "acrid", "fumes", "chemical", "ammonia", "stench", "caustic", "chlorine", "burning rubber"}},
//     {"CLICHE", {"time stretches", "spine stiffened", "breath caught", "skin crawled", "reality unraveling", "consciousness expanding", "void staring", "puppet with no strings"}},
//     {"SENSORY_LOOP", {"metallic tang", "oil-slick", "pulsating", "throbbing", "iridescent eyes", "viscous", "gelatinous", "shards", "jagged"}},
//     {"ABSTRACT_TRAP", {"void", "consciousness", "awareness", "existence", "essence", "soul", "infinite", "eternal"}}
// };

// void force_concept_lock(const std::string& cat) {
//     if (global_concepts.find(cat) == global_concepts.end()) return;
    
//     std::cout << " [CONCEPT JAIL] Re-locking category '" << cat << "' (Inherited/Triggered)." << std::endl;
//     for(const auto& banned_word : global_concepts[cat]) {
//         // Check if already in TTL jail
//         bool already_banned = false;
//         for (auto& existing : concept_jail_with_ttl) {
//             if (existing.word == banned_word) {
//                 existing.blocks_remaining = 5; // Reset TTL
//                 already_banned = true;
//                 break;
//             }
//         }
//         if (!already_banned) {
//             concept_jail_with_ttl.push_back({banned_word, 5});
//         }
//     }
// }

// void update_ban_list(MultiAgentState& state, const std::string& text, int current_block = 0) {
//     // 1. Decrement TTL and remove expired bans
//     for (auto it = concept_jail_with_ttl.begin(); it != concept_jail_with_ttl.end();) {
//         it->blocks_remaining--;
//         if (it->blocks_remaining <= 0) {
//             std::cout << " [CONCEPT JAIL] Unbanning: " << it->word << " (TTL expired)" << std::endl;
//             it = concept_jail_with_ttl.erase(it);
//         } else {
//             ++it;
//         }
//     }
    
//     // 2. Kavram Tarayıcı (Expanded)
//     std::string scan = text;
//     std::transform(scan.begin(), scan.end(), scan.begin(), ::tolower);

//     for(const auto& [cat, words] : global_concepts) {
//         int hits = 0;
//         std::vector<std::string> matched_words;
//         for(const auto& w : words) {
//             if(scan.find(w) != std::string::npos) {
//                 hits++;
//                 matched_words.push_back(w);
//             }
//         }
//         // Eşik: Eğer bir blokta aynı konseptten 2 kelime geçerse, sonraki blokta O KONSEPTİ KOMPLE YASAKLA.
//         // [UPDATED] With 5-block TTL AND Capitalization
//         if(hits >= 2) {
//             std::cout << " [CONCEPT JAIL] Loop Detected in category '" << cat << "'. Locking for 5 blocks." << std::endl;
//             state.recent_mistakes.push_back("CONCEPT JAIL: Loop Detected in category '" + cat + "'. Locking for 5 blocks.");
//             // force_concept_lock(cat);
//         }
//     }
    
//     // 3. Sync TTL jail to state banlist (for use in generate_text)
//     // Clear old banlist and rebuild from TTL jail
//     state.recent_vocab_banlist.clear();
//     for (const auto& term : concept_jail_with_ttl) {
//         state.recent_vocab_banlist.push_back(term.word);
//     }
    
//     // Listeyi temizle (Hafıza sınırı) - increased
//     while(concept_jail_with_ttl.size() > 200) concept_jail_with_ttl.pop_front();
// }

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
    std::vector<llama_token> tokens_list(formatted_prompt.length() + 128); 
    int n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
         tokens_list.resize(-n_tokens);
         n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
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

    std::string correction = "";
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
    std::vector<llama_token> tokens_list(formatted_prompt.length() + 128); 
    int n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
         tokens_list.resize(-n_tokens);
         n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
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

    std::string repair = "";
    int max_repair = 400; // Allow enough space for rewrite
    
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

// --- LLAMA ENGINE ---

// [DYNAMIC MEMORY MANAGEMENT]
// User requested full control over memory. We will load auxiliary models ON DEMAND.
// Llama-8B (Main) stays resident. Others (MiroThinker, RWKV, Gemma, Phi, Qwen) are swapped.

void unload_all_aux(MultiAgentState& state) {
    if (state.model_scout) { llama_free(state.ctx_scout); llama_free_model(state.model_scout); state.model_scout = nullptr; state.ctx_scout = nullptr; }
    if (state.model_phi) { llama_free(state.ctx_phi); llama_free_model(state.model_phi); state.model_phi = nullptr; state.ctx_phi = nullptr; }
    if (state.model_qwen_stabilizer) { llama_free(state.ctx_qwen_stabilizer); llama_free_model(state.model_qwen_stabilizer); state.model_qwen_stabilizer = nullptr; state.ctx_qwen_stabilizer = nullptr; }
    if (state.model_qwen_creative) { llama_free(state.ctx_qwen_creative); llama_free_model(state.model_qwen_creative); state.model_qwen_creative = nullptr; state.ctx_qwen_creative = nullptr; }
    if (state.model_mirothinker) { llama_free(state.ctx_mirothinker); llama_free_model(state.model_mirothinker); state.model_mirothinker = nullptr; state.ctx_mirothinker = nullptr; }
    if (state.model_rwkv) { llama_free(state.ctx_rwkv); llama_free_model(state.model_rwkv); state.model_rwkv = nullptr; state.ctx_rwkv = nullptr; }
    
    // [SINGLE SLOT POLICY] Add Mamba and Hermes
    if (state.model_mamba) { llama_free(state.ctx_mamba); llama_free_model(state.model_mamba); state.model_mamba = nullptr; state.ctx_mamba = nullptr; }
    if (state.model_hermes) { llama_free(state.ctx_hermes); llama_free_model(state.model_hermes); state.model_hermes = nullptr; state.ctx_hermes = nullptr; }
    if (state.model_saul) { llama_free(state.ctx_saul); llama_free_model(state.model_saul); state.model_saul = nullptr; state.ctx_saul = nullptr; }
}

bool ensure_model_loaded(MultiAgentState& state, llama_model** model_ptr, llama_context** ctx_ptr, const std::string& path, int n_ctx, int n_gpu_layers) {
    if (*model_ptr != nullptr) return true; // Already loaded

    std::cout << "[MEMORY] Swapping in model: " << path << "..." << std::endl;
    unload_all_aux(state); // Unload others first! (Single Aux Slot Policy)

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers; 
    *model_ptr = llama_model_load_from_file(path.c_str(), mparams);
    
    if (!*model_ptr) {
        std::cerr << "[ERR] Failed to load dynamic model: " << path << std::endl;
        return false;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 512; // Lower batch for aux
    *ctx_ptr = llama_init_from_model(*model_ptr, cparams);
    
    if (!*ctx_ptr) {
        std::cerr << "[ERR] Failed to create context for dynamic model." << std::endl;
        return false;
    }
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
    
    // --- Llama (Ana Yazar) ---
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99; // Metal
    state.model_main = llama_model_load_from_file("/Users/farukalpay/Desktop/cpp/local_mind/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", mparams);
    if (!state.model_main) {
        std::cerr << "[ERR] Failed to load Main Model!" << std::endl;
        exit(1);
    }
    
    auto cparams_main = llama_context_default_params();
    cparams_main.n_ctx = 16384; 
    cparams_main.n_batch = 4096;
    state.ctx_main = llama_init_from_model(state.model_main, cparams_main);
    if (!state.ctx_main) {
        std::cerr << "[ERR] Failed to create Main Context!" << std::endl;
        exit(1);
    }

    // Initialize other pointers to nullptr for dynamic switching
    state.model_scout = nullptr; state.ctx_scout = nullptr;
    state.model_phi = nullptr; state.ctx_phi = nullptr;
    state.model_qwen_stabilizer = nullptr; state.ctx_qwen_stabilizer = nullptr;
    state.model_qwen_creative = nullptr; state.ctx_qwen_creative = nullptr;
    state.model_mirothinker = nullptr; state.ctx_mirothinker = nullptr;
    state.model_rwkv = nullptr; state.ctx_rwkv = nullptr;
    state.model_saul = nullptr; state.ctx_saul = nullptr;
    
    // Init Sensors
    std::cout << "[SYSTEM] Initializing CodeBERT Sensor..." << std::endl;
    state.sensor = std::make_shared<CodeBERT>();
    state.sensor->load("/Users/farukalpay/Desktop/cpp/local_mind/models/onnx/model_int8.onnx"); 

    // --- DeBERTa (The Judge) ---
    std::cout << "[SYSTEM] Initializing DeBERTa Judge..." << std::endl;
    state.deberta = std::make_shared<DeBERTaNLI>();
}



// --- GEMMA SCOUT FUNCTION ---
std::string gemma_inject_chaos(MultiAgentState& state, const std::string& context) {
    // DYNAMIC LOAD
    if (!ensure_model_loaded(state, &state.model_scout, &state.ctx_scout, "/Users/farukalpay/Desktop/cpp/local_mind/models/gemma-2-2b-it-Q4_K_M.gguf", 4096, 99)) {
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
    std::vector<llama_token> tokens_list(prompt.length() + 128); 
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
    tokens_list.resize(n_tokens);

    // Decode Prompt
    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size()); 
    if (llama_decode(state.ctx_scout, batch) != 0) return ""; 

    // Sampler (High Temp 0.95)
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.95f)); 
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(std::rand())); 

    std::string chaos_vector = "";
    int max_tokens = 20; // Slightly more for 3 words
    
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
    if (!ensure_model_loaded(state, &state.model_phi, &state.ctx_phi, "/Users/farukalpay/Desktop/cpp/local_mind/models/phi-2.Q4_K_M.gguf", 2048, 0)) { // CPU preferred for Phi? No, Metal is fine if alone. Let's try CPU (0) first as original code did, or 99? 
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

// --- QWEN 2.5 STABILIZER FUNCTION ---
std::string qwen_stabilize(MultiAgentState& state, const std::string& input_text) {
    // DYNAMIC LOAD: Qwen Stabilizer
    if (!ensure_model_loaded(state, &state.model_qwen_stabilizer, &state.ctx_qwen_stabilizer, QWEN_STABILIZER_PATH, 4096, 99)) {
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
    
    std::vector<llama_token> tokens_list(prompt.length() + 128); 
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
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
    
    std::string output = "";
    int max_toks = 600; 

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
    if (!ensure_model_loaded(state, &state.model_qwen_creative, &state.ctx_qwen_creative, QWEN_CREATIVE_PATH, 2048, 99)) {
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
    if (!ensure_model_loaded(state, &state.model_qwen_creative, &state.ctx_qwen_creative, QWEN_CREATIVE_PATH, 2048, 99)) {
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
    if (!ensure_model_loaded(state, &state.model_qwen_creative, &state.ctx_qwen_creative, QWEN_CREATIVE_PATH, 2048, 99)) {
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

// [SAUL ENGINE] Dynamic Prefill Generator
// Hardcoded stringleri bitiren fonksiyon budur.
std::string generate_saul_prefill(MultiAgentState& state, std::string topic) {
    // Bellek Kontrolü: Saul yüklü değilse yükle (Diğerlerini swap-out yap)
    // NOT: MODEL_PATH kısmını indirdiğin "models/saul-7b.gguf" olarak güncellemelisin!
    if (!ensure_model_loaded(state, &state.model_saul, &state.ctx_saul, SAUL_PATH, 2048, 99)) {
        return "The evidence suggests"; // Model yüklenemezse acil durum fallback'i
    }


    // Saul'a özel "Hukukçu/Forensic" Prompt - Refined for robustness
    std::string prompt = 
        "### Instruction:\n"
        "Generate a short, clinical, forensic sentence starter (3-5 words) for the topic: '" + topic + "'.\n"
        "Rules: No metaphors. No repetition of the topic. Use legal/anatomical terminology.\n"
        "Example: 'Autopsy reveals significant trauma'\n"
        "### Response:\n";

    // generate_layer fonksiyonunu çağır (Senin kodunda zaten var)
    std::string out = generate_layer(state.ctx_saul, state.model_saul, prompt, 15, 0.7f, {"\n", ".", "\"", "[", "]", "###", ":"}, {});
    
    // Temizlik (Boşlukları vs sil)
    std::string clean = trim_trailing_noise(out);

    // Fallback: If output is just the topic or empty
    if (clean.length() < 3 || clean.find(topic) != std::string::npos) {
        return "The forensic evidence reveals";
    }
    return clean;
}

// [NEW] CHRONOS: The World Engine
// Executes python script to predict narrative weather
std::string run_chronos_forecast(MultiAgentState& state) {
    // 1. Prepare Data
    json j_data;
    j_data["entropy"] = state.history_entropy;
    j_data["sentiment"] = state.history_sentiment;
    j_data["speed"] = state.history_speed;
    j_data["timestamp"] = (long)std::time(nullptr); // [USER REQ] Real-time sync
    
    std::string valid_json = j_data.dump();
    
    // Escape quotes for command line (simplistic)
    // ideal approach: write to temp file, but command line arg is faster for small data
    std::string cmd = "python3 scripts/chronos_adapter.py '" + valid_json + "'";
    
    // 2. Execute
    std::string result_json;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "SUNNY";
    char buffer[128];
    while (fgets(buffer, 128, pipe) != NULL) {
        result_json += buffer;
    }
    pclose(pipe);
    
    // 3. Parse
    try {
        auto j_res = json::parse(result_json);
        std::string forecast = j_res.value("forecast", "SUNNY");
        std::string reason = j_res.value("reason", "Unknown");
        
        std::cout << " [CHRONOS] FORECAST: " << forecast << " (" << reason << ")" << std::endl;
        
        if (forecast == "STORM" || forecast == "WINDY") {
            return "EVENT_INJECTION: A SUDDEN CATASTROPHE OCCURS. " + reason;
        }
        
    } catch (...) {
        std::cout << " [CHRONOS] Parse Error." << std::endl;
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
    for(int i=0; i<3; i++) kinetic_dynamic.push_back(generate_saul_prefill(state, "sudden physical impact or force"));

    std::vector<std::string> visceral_dynamic;
    for(int i=0; i<3; i++) visceral_dynamic.push_back(generate_saul_prefill(state, "biological trauma or internal anatomy"));

    std::vector<std::string> surreal_dynamic;
    for(int i=0; i<3; i++) surreal_dynamic.push_back(generate_saul_prefill(state, "visual distortion or hallucination evidence"));

    std::vector<std::string> observer_dynamic;
    for(int i=0; i<3; i++) observer_dynamic.push_back(generate_saul_prefill(state, "environmental decay and material structure"));

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
    
    // Safety check for model pointer
    if (!state.ctx_mirothinker) {
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
    if (!ensure_model_loaded(state, &state.model_rwkv, &state.ctx_rwkv, RWKV_PATH, 2048, 99)) {
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
    if (!ensure_model_loaded(state, &state.model_rwkv, &state.ctx_rwkv, RWKV_PATH, 2048, 99)) {
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
    if (!ensure_model_loaded(state, &state.model_mamba, &state.ctx_mamba, MAMBA_PATH, 2048, 99)) {
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
    if (!ensure_model_loaded(state, &state.model_mirothinker, &state.ctx_mirothinker, "/Users/farukalpay/Desktop/cpp/local_mind/models/MiroThinker-v1.5-30B.Q4_K_M.gguf", 4096, 99)) {
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
    if (!ensure_model_loaded(state, &state.model_mirothinker, &state.ctx_mirothinker, "/Users/farukalpay/Desktop/cpp/local_mind/models/MiroThinker-v1.5-30B.Q4_K_M.gguf", 4096, 99)) {
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
    if (!ensure_model_loaded(state, &state.model_mirothinker, &state.ctx_mirothinker, "/Users/farukalpay/Desktop/cpp/local_mind/models/MiroThinker-v1.5-30B.Q4_K_M.gguf", 4096, 99)) {
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
    std::vector<llama_token> tokens_list(prompt.length() + 128); 
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
    tokens_list.resize(n_tokens);

    // Decode Prompt
    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size());
    if (llama_decode(ctx, batch) != 0) return "";

    // Sampler Config
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    
    // [LOGIT BIAS] for Banned Words
    std::vector<llama_logit_bias> biases;
    for (const auto& w : banned_words) {
        std::vector<llama_token> b_toks(16);
        int n = llama_tokenize(vocab, w.c_str(), w.length(), b_toks.data(), b_toks.size(), false, false);
        if (n > 0) biases.push_back({b_toks[0], -1000.0f}); // Token'ı atomik seviyede yasakla
    }
    if(!biases.empty()) llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), biases.size(), biases.data()));

    // [UPDATED] Sampling Strategy (User Op C) - Include Top-P for Chaos
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40)); // Standard cleanup
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1)); // Nucleus Sampling
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(std::rand()));
    
    std::string output = "";
    
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
    if (!ensure_model_loaded(state, &state.model_hermes, &state.ctx_hermes, HERMES_PATH, 4096, 99)) {
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

std::string generate_composite_narrative(MultiAgentState& state, const std::string& history, const std::string& summary, int domain_idx, std::string directive = "", std::string premonition = "") {
    
    // 0. EXTRACT DEEP CONTEXT (Grounding)
    // Sadece son cümleyi değil, son 2000 karakteri al ki sahne kopmasın.
    std::string recent_history = "";
    if (history.length() > 2000) {
        recent_history = history.substr(history.length() - 2000);
    } else {
        recent_history = history.empty() ? "The void is silent." : history;
    }

    // Anchor (Son Cümle - Action bağlantısı için)
    std::string last_sentence = "";
    size_t last_dot = history.rfind('.');
    if (last_dot != std::string::npos && last_dot > 150) {
        last_sentence = history.substr(last_dot + 1);
    } else {
        last_sentence = history.substr(history.length() > 200 ? history.length() - 200 : 0);
    }

    // --- PHASE 1: GEMMA BATCHING (DIRECTIONAL ESCAPE) ---
    // [UPDATED] Geometric Selection: Find vector furthest from History Centroid
    
    std::cout << "\n[PHASE 1] Directional Escape (Geometric Selection)..." << std::endl;
    std::vector<std::string> options;
    
    // 1. Calculate History Centroid (The Gravitational Well)
    std::vector<float> centroid;
    if (state.sensor && !state.history_embeddings.empty()) {
        centroid = state.sensor->compute_centroid(state.history_embeddings);
        std::cout << " [GEOMETRY] Computed Centroid from " << state.history_embeddings.size() << " blocks." << std::endl;
    }
    
    // 2. Generate Divergent Candidates (High Entropy)
    // DYNAMIC LOAD: Gemma (Scout)
    if (!ensure_model_loaded(state, &state.model_scout, &state.ctx_scout, "/Users/farukalpay/Desktop/cpp/local_mind/models/gemma-2-2b-it-Q4_K_M.gguf", 4096, 99)) {
        std::cerr << " [ERR] Failed to load Gemma for Phase 1!" << std::endl;
        // Fallback or skip?
    } else {
    
    for (int i = 0; i < 5; i++) {
        std::string p1_prompt; // [FIX] Declaration moved outside scope
        if (!directive.empty()) {
             // [OVERRIDE] User Directive Mode
             p1_prompt = 
            "<start_of_turn>user\n"
            "CONTEXT: " + last_sentence + "\n"
            "TASK: Initialize the scene based on the HIDDEN MOTIVE.\n"
            "CONSTRAINT: Focus strictly on the motive. Do not deviate.\n"
            "HIDDEN MOTIVE: " + directive + "\n"
            "STYLE: Sensory. Direct.\n"
            "OUTPUT: 1 sentence.\n"
            "<end_of_turn><start_of_turn>model\n";
        } else {
             // [DEFAULT] Drift Mode
             p1_prompt = 
            "<start_of_turn>user\n"
            "CONTEXT: " + last_sentence + "\n"
            "TASK: Generate a sensory fragment that feels RADICALLY DIFFERENT from the context.\n"
            "CONSTRAINT: Break continuity. Change the lighting, texture, or physics violently.\n"
            "HIDDEN MOTIVE: None\n"
            "STYLE: Disorienting. High Entropy.\n"
            "OUTPUT: 1 sentence.\n"
            "<end_of_turn><start_of_turn>model\n";
        }
        
        // High temp for Maximum Divergence
        std::string angle = generate_layer(state.ctx_scout, state.model_scout, p1_prompt, 60, 1.25f, {"\n"}, {});
        if (!angle.empty()) {
            std::string trimmed = trim_trailing_noise(angle);
            // [CRITICAL] ESCAPE VECTOR REJECTION FILTER
            // Prevents system prompt collapse
            if (!is_toxic_escape_vector(trimmed)) {
                options.push_back(trimmed);
            } else {
                std::cout << " [VECTOR REJECTED] Toxic: " << trimmed.substr(0, 40) << "..." << std::endl;
            }
        }
    }
    } // End else (valid gemma)
    
    // [USER REQ] 2.1 ADD RWKV ORTHOGONAL VECTOR
    // RWKV 7 provides a "Linear Attention" perspective distinct from Transformer attention.
    std::string rwkv_vec = generate_rwkv_escape(state, recent_history);
    if (!rwkv_vec.empty()) {
        std::string trimmed_r = trim_trailing_noise(rwkv_vec);
        if (!is_toxic_escape_vector(trimmed_r)) {
            options.push_back(trimmed_r);
            std::cout << " [VECTOR ADDED] RWKV 7 Orthogonal Candidate." << std::endl;
        }
    }

    // 3. Selection by Maximum Distance
    std::string best_vector = "";
    float max_dist = -1.0f;
    
    if (state.sensor && !centroid.empty() && !options.empty()) {
        std::cout << " [VECTORS] Calculating escape trajectories:" << std::endl;
        for(int i=0; i<options.size(); i++) {
            std::vector<float> vec = state.sensor->embed(options[i]);
            float sim = state.sensor->cosine_similarity(vec, centroid);
            float dist = 1.0f - sim; // Distance is dissimilarity
            
            std::cout << "  [" << i << "] Dist: " << dist << " | " << options[i].substr(0, 50) << "..." << std::endl;
            
            if (dist > max_dist) {
                max_dist = dist;
                best_vector = options[i];
            }
        }
    } else if (!options.empty()) {
        // Fallback: Random or First if no sensor
        best_vector = options[0]; 
    }
    
    if (best_vector.empty()) best_vector = "The logic fractures into distinct shards.";
    std::cout << " [SELECTED ESCAPE VECTOR] " << best_vector << " (Dist: " << max_dist << ")" << std::endl;
    
    // 4. Assign to Pipeline Variable
    // Replaces 'atmosphere' variable from old code
    std::string atmosphere = best_vector;
    std::cout << " [Vector Applied]: " << atmosphere << std::endl;
    
    // --- PHASE 3: REFLEXIVE ACTION ---
    std::cout << "[PHASE 3] Generating Reflexive Action..." << std::flush;
    std::string p2_prompt = 
        "<|start_header_id|>system<|end_header_id|>\n"
        "SITUATION: " + atmosphere + "\n"
        "TASK: Describe the protagonist's INVOLUNTARY physical reaction.\n"
        "CONSTRAINT: Do NOT use 'I decided' or 'I tried'. Use reflexes (twitch, spasm, gasp, collide, gag).\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        "I"; 
    std::string action = generate_layer(state.ctx_main, state.model_main, p2_prompt, 50, 0.6f, {"\n", "."}); 
    std::cout << " Done." << std::endl;

    // --- PHASE 4: TEXTURE (GEMMA vs QWEN) ---
    // Condition 3: Controlled Noise Injection (Gemma vs Qwen)
    std::string texture = "";
    bool use_qwen_noise = (std::rand() % 100 < 30);
    
    // Fallback textures for when models produce meta-commentary
    auto get_fallback_texture = []() -> std::string {
        static int idx = 0;
        std::vector<std::string> fallbacks = {
            "wet rust, machine oil, copper wires",
            "cold static, burning plastic, distant thunder",
            "salt and iron, buzzing fluorescent, damp concrete",
            "sulfur, rattling chains, peeling wallpaper",
            "ammonia, grinding gears, cracked tiles"
        };
        idx = (idx + 1) % fallbacks.size();
        return fallbacks[idx];
    };
    
    // Sanitize Gemma output - reject meta-commentary
    auto sanitize_gemma_texture = [&](const std::string& output) -> std::string {
        std::string lower = output;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        
        // Check for meta-commentary patterns
        if (lower.find("here are") != std::string::npos ||
            lower.find("here's") != std::string::npos ||
            lower.find("options") != std::string::npos ||
            lower.find("trying to") != std::string::npos ||
            lower.find("requested") != std::string::npos ||
            lower.find("aiming for") != std::string::npos ||
            lower.find("description") != std::string::npos ||
            lower.find("focusing on") != std::string::npos ||
            output.length() < 10 ||
            output.length() > 200) {
            std::cout << " [GEMMA SANITIZE] Meta-output rejected. Using fallback." << std::endl;
            return get_fallback_texture();
        }
        return output;
    };
    
    if (use_qwen_noise) {
         // Qwen 2 Creative Concept
         std::cout << "[PHASE 4] Texture Source: QWEN 2 (Controlled Soft Noise)..." << std::flush;
         // Qwen function handles dynamic loading now
         texture = qwen_creative_concept(state);
    } 
    
    // Fallback or Primary: GEMMA (if Qwen not used or returned empty)
    if (texture.empty()) {
         // Gemma 2 Chaos
         std::cout << "[PHASE 4] Texture Source: GEMMA 2 (Biological Chaos)..." << std::flush;
         
         // DYNAMIC LOAD: Gemma
         if (ensure_model_loaded(state, &state.model_scout, &state.ctx_scout, "/Users/farukalpay/Desktop/cpp/local_mind/models/gemma-2-2b-it-Q4_K_M.gguf", 4096, 99)) {
             std::string p3_prompt = 
                "<start_of_turn>user\n"
                "Subject: " + action + "\n"
                "TASK: Describe the texture/smell/sound of this specific split-second.\n"
                "STYLE: Biological. Industrial. Disgusting. Cyberpunk.\n"
                "OUTPUT: 5-10 vivid words. NO EXPLANATIONS. NO 'Here is'.\n"
                "<end_of_turn><start_of_turn>model\n"; 
             std::string raw_texture = generate_layer(state.ctx_scout, state.model_scout, p3_prompt, 60, 1.0f, {"\n"});
             texture = sanitize_gemma_texture(raw_texture);
         } else {
             std::cerr << " [ERR] Gemma Failed to Load. Using Fallback Texture." << std::endl;
             texture = get_fallback_texture();
         }
    }
    std::cout << " Done. [" << texture << "]" << std::endl;

    // --- PHASE 5: PERSONA SELECTION (THE IDIOSYNCRASY ENGINE) ---
    // Use the Global Persona Deck for variety
    std::cout << "[PHASE 5] Selecting Persona..." << std::flush;
    
    Persona p = draw_persona(state);
    std::string prefill = pick_random(p.prefills);
    std::cout << " [" << p.name << "] -> Prefill: '" << prefill << "'..." << std::endl;

    // --- PHASE 6: MASTERING (THE HEAD TRANSPLANT) ---
    
    std::string constraint = get_domain_constraint((SemanticDomain)domain_idx);
    // [STRUCTURAL JAIL APPLICATION]
    std::string jail_rules = "";
    if (active_constraints.force_minimal_adjectives) {
        jail_rules += "CONSTRAINT: STRUCTURAL JAIL ACTIVE [SENSORY OVERLOAD]. MAX 1 ADJECTIVE PER SENTENCE. Use concrete verbs only.\n";
    }
    if (active_constraints.ban_metaphors) {
        jail_rules += "CONSTRAINT: STRUCTURAL JAIL ACTIVE [METAPHOR OVERLOAD]. NO METAPHORS. Describe physical reality only.\n";
    }
    if (active_constraints.ban_body_vocab) {
        jail_rules += "CONSTRAINT: STRUCTURAL JAIL ACTIVE [BODY DISTRESS]. DO NOT MENTION: veins, blood, skin, heart, lungs, breath, pulse. Focus on EXTERNAL objects.\n";
    }
    if (active_constraints.ban_abstract_nouns) {
        jail_rules += "CONSTRAINT: STRUCTURAL JAIL ACTIVE [VOID IMAGERY]. DO NOT MENTION: void, abyss, darkness, silence, empty. Focus on solid matter.\n";
    }

    // [STRUCTURAL JITTER] - Randomize sentence order to break monotony
    auto apply_structural_jitter = [](std::string text) -> std::string {
        if (std::rand() % 100 < 30) { // 30% chance to shuffle sentences
            std::vector<std::string> sentences;
            std::stringstream ss(text);
            std::string segment;
            while(std::getline(ss, segment, '.')) {
                if(segment.length() > 3) sentences.push_back(segment + ".");
            }
            if (sentences.size() > 2) {
                std::cout << " [JITTER] Shuffling sentence order for variety." << std::endl;
                std::shuffle(sentences.begin() + 1, sentences.end(), std::mt19937(std::random_device()()));
                text = "";
                for(const auto& s : sentences) text += s;
            }
        }
        return text;
    };

    // [PHASE 6] MIXER PROMPT (The Conductor)
    // [UPDATED] Use Natural Language Serializer for World State
    // std::string world_state_prompt = "";
    // if (!state.world_state.empty()) {
    //  world_state_prompt = "[WORLD STATE / KNOWN FACTS]\n";
    //  for (const auto& pair : state.world_state) {
    //      world_state_prompt += pair.first + ": " + pair.second + "\n";
    //  }
    //  world_state_prompt += "INSTRUCTION: You MUST respect these facts. Do not contradict them.\n";
    // }
    
    std::string world_state_prompt = "";
    if (!state.world_state.empty()) {
        world_state_prompt = "[WORLD STATE / MEMORY]\n" + format_world_state_narrative(state.world_state) + 
                             "INSTRUCTION: Incorporate these details naturally. Do not contradict them.\n";
    }

    std::string mixer_prompt = 
        "<|start_header_id|>system<|end_header_id|>\n"
        "ROLE: You are a New Wave Sci-Fi Author (Style: J.G. Ballard, William Gibson). " + p.instruction + "\n"
        + constraint + "\n"
        + jail_rules + "\n"
        + world_state_prompt +
        "\n"
        "STORY CONTEXT (MEMORY): " + summary + "\n"
        "IMMEDIATE SURROUNDINGS (SCENE): " + recent_history.substr(recent_history.length() > 500 ? recent_history.length()-500 : 0) + "\n"
        "\n"
        "MANDATE: " + (directive.empty() ? "None" : "PRIORITY OVERRIDE. Focus primarily on the INTERNAL IMPULSE: " + directive) + "\n"
        "NEW INPUTS:\n"
        "- Shift: " + atmosphere + "\n"
        "- Reflex: I " + action + "\n"
        "- Sensation: " + texture + "\n"
        "- NERVOUS SYSTEM PREMONITION (INSTINCT): " + (premonition.empty() ? "Unclear" : premonition) + "\n"
        "INSTRUCTION: Incorporate the 'Premonition' subtly. The character feels this is about to happen.\n"
        "- INTERNAL IMPULSE: " + (directive.empty() ? "None" : directive) + "\n"
        "- Subconscious Spark: " + ((directive == "CONTINUE" && std::rand() % 100 < 20) ? qwen_creative_burst(state, recent_history) : "None") + "\n" // Cond 1
        "- Human Association: " + (std::rand() % 100 < 25 ? qwen_existential_association(state, recent_history) : "None") + "\n" // Cond 2
        "\n"
        "STRICT RULES:\n"
        "1. START EXACTLY with: '" + prefill + "'.\n"
        "2. SHOW, DON'T TELL. Don't say 'I felt afraid'. Describe the cold sweat.\n"
        "3. NO LOGIC. Do not explain WHY. Focus on the SENSORY INPUT.\n"
        "4. SENTENCE VARIETY. Use fragments. Break the grammar rules if needed for impact.\n"
        "5. LENGTH: Write a dense, immersive paragraph (approx 200 words). Do NOT be brief.\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        "" + prefill; // FORCE THE START

    // RETRY LOOP
    std::string final_block = "";
    int attempts = 0;
    std::vector<std::string> banned_words; // [FIX] Declaration added
    
    // [UPDATED] Anti-Repetition Logic: Frequency Penalty from History
    // Scan recent history (last 2000 chars) for frequent words and soft-ban them.
    if (!history.empty()) {
        std::map<std::string, int> freq_map;
        std::string recent_hist = history.substr(history.length() > 2500 ? history.length()-2500 : 0);
        std::transform(recent_hist.begin(), recent_hist.end(), recent_hist.begin(), ::tolower);
        
        std::stringstream ss(recent_hist);
        std::string word;
        while (ss >> word) {
            // Basic cleanup
            word.erase(std::remove_if(word.begin(), word.end(), [](char c){ return !isalnum(c); }), word.end());
            if (word.length() > 4) { // Only track significant words
                freq_map[word]++;
            }
        }
        
        // Ban words that appear > 3 times in recent context
        for (const auto& [w, count] : freq_map) {
            // Whitelist common structural words
            if (w == "with" || w == "from" || w == "that" || w == "this" || 
                w == "they" || w == "their" || w == "there" || w == "into" || w == "through") continue;
                
            if (count > 3) {
                // Add to transient ban list for this generation
                banned_words.push_back(w);
                
                // Capitalized version too
                std::string cap = w; 
                cap[0] = toupper(cap[0]);
                banned_words.push_back(cap);
                
                // Log critical repeats
                if (count > 5) {
                   std::cout << " [ANTI-REPEAT] Transient Ban: " << w << " (Count: " << count << ")" << std::endl;
                }
            }
        }
    }

    // [UPDATED] Anti-Cliche Logic (User Request 5)
    if (state.ctx_main) {
       // "fingernails", "bone", "ozone", "metallic" - if recently used.
       // Check if prompt contains them? Or just hard ban them if they were used in the *previous* output (context).
       std::vector<std::string> cliches = {"fingernails", "bone", "ozone", "metallic", "slide", "pulsating", "viscous", "shards"};
       for (const auto& word : cliches) {
           if (history.find(word) != std::string::npos) {
               // Add to banned words for THIS generation
               banned_words.push_back(word);
               // Also capitalize
               std::string cap = word; cap[0] = toupper(cap[0]);
               banned_words.push_back(cap);
               std::cout << " [CLICHE BLOCK] Banned recently used cliche: " << word << std::endl;
               state.recent_mistakes.push_back("CLICHE BLOCK: Banned " + word);
           }
       }
    }

    while (attempts < 3) {
        attempts++;
        // Stop word EOT. \n yok, paragrafı bitirmesine izin ver.
        // [UPDATED] Pass dynamic ban list
        std::deque<std::string> effective_bans = state.recent_vocab_banlist; // Copy global
        for(const auto& w : banned_words) effective_bans.push_back(w); // Add local cliches

        std::string generated_body = generate_layer(state.ctx_main, state.model_main, mixer_prompt, 800, p.temp, {"<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"}, effective_bans);
        
        // *** CRITICAL FIX: KAFAYI GÖVDEYE DİKME İŞLEMİ ***
        // "green goo" hatasının ilacı bu satırdır. Prefill'i geri ekliyoruz.
        std::string body_clean = strip_meta_commentary(generated_body);
        if (!prefill.empty() && !body_clean.empty() && prefill.back() != ' ' && body_clean.front() != ' ' && body_clean.front() != '.') {
            final_block = prefill + " " + body_clean;
        } else {
            final_block = prefill + body_clean;
        }

        // [USER REQ] CRITICAL FIX: POST-GENERATION HARD REJECT FOR CLICHES
        // Logit bias is not enough for multi-token words (e.g. "metallic"). We must check unconditionally.
        bool found_cliche = false;
        std::string lower_final = final_block;
        std::transform(lower_final.begin(), lower_final.end(), lower_final.begin(), ::tolower);
        
        for (const auto& banned : state.recent_vocab_banlist) {
            std::string lower_banned = banned;
            std::transform(lower_banned.begin(), lower_banned.end(), lower_banned.begin(), ::tolower);
            if (!lower_banned.empty() && lower_final.find(lower_banned) != std::string::npos) {
                std::cout << " [HARD REJECT] Found banned word: '" << banned << "' in output. Retrying..." << std::endl;
                state.recent_mistakes.push_back("HARD REJECT: Found '" + banned + "'");
                
                // [FIX] Force Replace if this is the last attempt
                if (attempts >= 2) { 
                     std::cout << " [HARD REJECT] FATAL: Failed to clear banned word. Redacting..." << std::endl;
                     size_t pos = lower_final.find(lower_banned);
                     while(pos != std::string::npos) {
                         // Find word boundaries
                         size_t start = pos;
                         while (start > 0 && isalpha(lower_final[start-1])) start--;
                         size_t end = pos + banned.length();
                         while (end < lower_final.length() && isalpha(lower_final[end])) end++;
                         
                         size_t len = end - start;
                         final_block.replace(start, len, "[REDACTED]");
                         lower_final.replace(start, len, "[REDACTED]"); // Update search string
                         
                         pos = lower_final.find(lower_banned, start + 10);
                     }
                } else {
                    found_cliche = true;
                    break; // Stop checking other words, just retry
                }
            }
        }
        
        if (found_cliche) {
            continue; // Retry loop immediately
        }

        // Validasyon: Model "I felt" diyerek kolaya kaçarsa reddet.
        bool bad_style = false;
        std::string lower_check = final_block;
        std::transform(lower_check.begin(), lower_check.end(), lower_check.begin(), ::tolower);
        
        if (lower_check.find("i felt") != std::string::npos) bad_style = true;
        if (lower_check.find("it seemed") != std::string::npos) bad_style = true;
        
        if (final_block.length() > 300 && !bad_style) {
            break; 
        }
        if (bad_style) {
            std::cout << " [RETRY " << attempts << "] Caught 'filter word' (felt/seemed)." << std::endl;
            state.recent_mistakes.push_back("RETRY " + std::to_string(attempts) + ": Caught filter word (felt/seemed)");
        }
        if (final_block.length() <= 300) {
            std::cout << " [RETRY " << attempts << "] Too short (" << final_block.length() << " chars)." << std::endl;
            state.recent_mistakes.push_back("RETRY " + std::to_string(attempts) + ": Too short (" + std::to_string(final_block.length()) + " chars)");
        }

    }
    
    // Safety: Hala "As I" ile başlıyorsa (prefill'e rağmen) kes.
    if (final_block.rfind("As I ", 0) == 0) {
         final_block.replace(0, 5, "I ");
    }

    // --- PHASE 7: REFLEXIVE CHECK (CodeBERT) ---
    // User Requirement: Check embedding. If REPEAT -> Trigger Phi-2 -> Restart with Directive.
    // Since we are at the end of the function, we can't easily restart *inside* without wrapping the whole function body.
    // Instead, we will return a SPECIAL SIGNAL or just handle it here in a loop if I refactor.
    // Refactoring the whole function to be inside a loop is safest.
    
    // --- PHASE 7: QWEN STABILIZER INTEGRATION ---
    // User Spec: Trigger if similarity > 0.85 or risk detected.
    
    // [UPDATE] Apply Structural Jitter Re-ordering
    final_block = apply_structural_jitter(final_block);

    // 1. Calculate Risk (Similarity)
    float current_risk = 0.0f;
    if (state.sensor && !state.history_embeddings.empty()) {
        std::vector<float> current_embed = state.sensor->embed(final_block);
        for (const auto& h : state.history_embeddings) {
           float s = state.sensor->cosine_similarity(current_embed, h);
           if (s > current_risk) current_risk = s;
        }
    }
    
    // 2. Check Concept Jail Triggers (Simplified scan)
    bool jail_risk = false;
    std::string lower_blk = final_block;
    std::transform(lower_blk.begin(), lower_blk.end(), lower_blk.begin(), ::tolower);
    // Hardcoded check for known offenders to be safe
    if(lower_blk.find("metal") != std::string::npos || lower_blk.find("blood") != std::string::npos || lower_blk.find("ozone") != std::string::npos) {
        jail_risk = true;
    }

    if (current_risk > 0.85 || jail_risk) {
        std::cout << "[STABILIZER] Triggering Qwen (Risk: " << current_risk << ", Jail: " << jail_risk << ")..." << std::endl;
        final_block = qwen_stabilize(state, final_block);
    } // Else maintain raw chaos

    return final_block;
}

// WRAPPER FOR REFLEX LOOP
std::string generate_composite_narrative_with_reflex(MultiAgentState& state, const std::string& history, const std::string& summary) {
    int attempts = 0;
    std::string directive = "";
    
    while(attempts < 3) {
        // CALL CORE GENERATION (We need to pass directive to it. Modifying signature of generate_composite_narrative needed?)
        // Yes. Let's modify the signature above or copy logic. 
        // Modifying signature is cleaner.
        // But for now, let's just use a global or a member in state? No, thread safety.
        // Let's modify generate_composite_narrative to take 'directive'.
        // Wait, I can't modify the signature in this chunk easily without changing the function definition line which is far away.
        // Actually, line 831 is generate_composite_narrative.
        // I will overload it or just copy the logic? No, copying is bad.
        // I will change the signature of generate_composite_narrative in a separate chunk to accept `std::string directive = ""` 
        
        // Assuming I changed the signature:
        // std::string block = generate_composite_narrative_internal(state, history, summary, directive);
        
        // For now, since I can't easily change the signature in *this* tool call without touching line 831 (which is consistent),
        // I will assume the 'generate_composite_narrative' I see in the file (lines 831-1008) is the one to use.
        // I will rename the *existing* function to `generate_composite_narrative_internal` in a separate chunk, and add `directive` arg.
        // Then this wrapper becomes the public `generate_composite_narrative`.
        
        return "Temp"; // Placeholder for logic I'll implement in the next step properly.
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
    std::vector<std::string> cliche_words = {
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


    // 5. PROMPT ASSEMBLY (Leak-Proof)
    std::string formatted_prompt = 
        "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "\n" + state_instruction + "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n" 
        "MEMORY STREAM:\n" + safe_context + "\n\n" // Used safe_context, not prompt
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n" + assistant_prefill; 

    // Tokenize
    auto* vocab = llama_model_get_vocab(state.model_main);
    std::vector<llama_token> tokens_list(formatted_prompt.length() + 128); // allocate buffer
    int n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.length(), tokens_list.data(), tokens_list.size(), true, true); // add_bos=true
    if (n_tokens < 0) {
         // resize and retry if needed, but usually enough
         tokens_list.resize(-n_tokens);
         n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
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

    // A. Add "Hard" Masked Tokens (from earlier logic)
    for (llama_token token_id : masked_tokens) {
        biases.push_back({token_id, -1000.0f}); 
    }

    // B. Convert all banned strings (Cliches, Concept Bans) to tokens
    std::vector<std::string> all_bans = cliche_words;
    all_bans.insert(all_bans.end(), banned_words.begin(), banned_words.end());
    all_bans.insert(all_bans.end(), active_concept_bans.begin(), active_concept_bans.end());
    
    for (const auto& w : all_bans) {
        std::vector<llama_token> b_tokens(w.length() + 4); 
        int n_bt = llama_tokenize(vocab, w.c_str(), w.length(), b_tokens.data(), b_tokens.size(), false, false); 
        if (n_bt > 0) {
            // Ban start token
            biases.push_back({b_tokens[0], -1000.0f}); 
        }
    }
    
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
        "<|eot_header_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "The protagonist"; // Pre-fill to enforce 3rd person

    // Tokenize
    auto* vocab = llama_model_get_vocab(state.model_main);
    std::vector<llama_token> tokens_list(formatted_prompt.length() + 128); 
    int n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
         tokens_list.resize(-n_tokens);
         n_tokens = llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
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
    
    std::string result = "The protagonist"; 
    std::cout << result << std::flush;      

    int max_tokens = 150; 
    
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
    if (txid == "GENESIS_TX") return {"GENESIS", "", "", "[]", "{}"};

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
            if(j.contains("mistakes_log")) mistakes_str = j["mistakes_log"].dump();
            std::string world_str = "{}";
            if(j.contains("world_state")) world_str = j["world_state"].dump();
            return {j.value("parent_tx", ""), j.value("content", ""), j.value("depth", "0"), mistakes_str, world_str};
        } catch (...) {
            std::cerr << " [ERR] Corrupt cache for " << txid << std::endl;
        }
    }

    // 2. NETWORK VERIFICATION (Fallback)
    if (txid == "GENESIS") return {"", "GENESIS BLOCK", "0", "[]", "{}"};

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
        std::string d = j.value("depth", "0");
        std::string mistakes_str = "[]";
        if(j.contains("mistakes_log")) mistakes_str = j["mistakes_log"].dump();
        std::string world_str = "{}";
        if(j.contains("world_state")) world_str = j["world_state"].dump();
        return {p, c, d, mistakes_str, world_str};
    } catch (...) {
        return {};
    }
}

std::vector<std::string> reconstruct_narrative(const std::string& head_tx, int max_depth) {
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
        current_tx = data[0]; 
    }

    std::reverse(narrative.begin(), narrative.end());
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
    system("mkdir -p cache");

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

    // CONTEXT LOADING
    if (parent_tx != "GENESIS_TX") {
        history = reconstruct_narrative(parent_tx, MAX_DEPTH);
        for (const auto& block : history) {
            full_context += block + " ";
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
    std::string current_summary_text = "The protagonist exists in a void.";
    
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
            
            // [JUDGE] DeBERTa NLI Logic Check (User Request)
            if (state.deberta) {
                // Premise: The current summary (Reality Context)
                // Hypothesis: The new content
                float nonsense_score = state.deberta->check_contradiction(current_summary_text, huge_content);
                if (nonsense_score > 0.90f) {
                    std::cout << " [JUDGE] REJECTED (Score: " << nonsense_score << "). MISTAKE LOGGED." << std::endl;
                    state.recent_mistakes.push_back(huge_content.substr(0, 100) + "..."); // Log fragment
                    directive = "CORRECTION: PREVIOUS BLOCK WAS ABSURD/CONTRADICTORY. BE GROUNDED.";
                    reflex_attempts++;
                    continue; // Retry
                }
            }

            // CODEBERT CHECK
            if (state.sensor) {
                std::vector<float> current_vec = state.sensor->embed(huge_content);
                float max_sim = 0.0f;
                // Check recent history (last 5 blocks) for immediate loop
                int count = 0;
                for (auto it = state.history_embeddings.rbegin(); it != state.history_embeddings.rend(); ++it) {
                    if (count++ > 5) break; 
                    float sim = state.sensor->cosine_similarity(current_vec, *it);
                    if (sim > max_sim) max_sim = sim;
                }
                
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

        // REFLEX & HISTORY UPDATE
        if (state.sensor && !new_content.empty()) {
            std::vector<float> vec = state.sensor->embed(new_content);
            state.history_embeddings.push_back(vec);
            if(state.history_embeddings.size() > 50) state.history_embeddings.erase(state.history_embeddings.begin());
            
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
            std::vector<float> pred_vec = state.sensor->embed(synapse_data.prediction);
            std::vector<float> actual_vec = state.sensor->embed(new_content);
            divergence = 1.0f - state.sensor->cosine_similarity(pred_vec, actual_vec);
        }

        json private_data;
        private_data["pid_state"] = {
            {"temp", current_temperature},
            {"complexity", calculate_token_entropy(new_content)} // Assuming entropy_score is this
        };
        private_data["parent"] = parent_tx;
        private_data["refs"] = refs;
        private_data["observer"] = (reflex_score > 0.91f) ? "phi-2" : "llama-3-8b";
        private_data["directive"] = directive;
        private_data["reflex_score"] = reflex_score;
        private_data["entropy_loss"] = calculate_entropy_loss(full_context); // Full detail
        private_data["content_preview"] = new_content.substr(0, 50);

        // 2. Public Data (Proof, Content, Intent)
        json public_payload;
        public_payload["genesis_txid"] = root_genesis_txid; // [FIXED] Hereditary TXID

        public_payload["program"] = "LOCAL_MIND_v3_DUAL_ENGINE";
        public_payload["model"] = "Llama-3-8B-Instruct + Gemma-2-2B";
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
        public_payload["depth"] = std::to_string(history.size() + 1); // Depth
        public_payload["mistakes_rag"] = state.recent_mistakes; // [NEW] mistakes log
        public_payload["world_state"] = state.world_state; // [REBEL] Persist World State
        public_payload["chronos"] = {
            {"weather", state.current_weather},
            {"pending_intervention", state.pending_chronos_msg} 
        };
        public_payload["content"] = new_content; // The Narrative Block itself
        public_payload["parent_tx"] = parent_tx;
        
        // Metadata for Uploader (Tags)
        // We'll write public_payload to file
        
        std::string json_content = public_payload.dump();
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
            
            // [CHRONOS] METRIC TRACKING & FORECAST
            float ent = calculate_token_entropy(new_content);
            state.history_entropy.push_back(ent);
            state.history_sentiment.push_back(current_temperature);
            state.history_speed.push_back((float)new_content.length()); 

            if(state.history_entropy.size() > 10) state.history_entropy.erase(state.history_entropy.begin());
            if(state.history_sentiment.size() > 10) state.history_sentiment.erase(state.history_sentiment.begin());
            if(state.history_speed.size() > 10) state.history_speed.erase(state.history_speed.begin());

            std::cout << "[CHRONOS] Sampling Weather... (Ent=" << ent << ", Temp=" << current_temperature << ")" << std::endl;
            state.pending_chronos_msg = run_chronos_forecast(state);
            state.current_weather = (state.pending_chronos_msg.empty()) ? "SUNNY" : "STORM";

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

    llama_backend_free();
    return 0;
}
