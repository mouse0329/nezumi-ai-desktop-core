#include "llama_wrapper.h"
#include "llama.cpp/include/llama.h"
#include <cstring>
#include <string>
#include <vector>

struct NezumiLlamaState {
    llama_model* model   = nullptr;
    llama_context* ctx     = nullptr;
    llama_sampler* sampler = nullptr;
    const llama_vocab* vocab   = nullptr;
};

// ログを完全に黙らせる
void dummy_log_callback(ggml_log_level level, const char* text, void* user_data) {
    (void)level; (void)text; (void)user_data;
}

static size_t find_stop_pos(const std::string &text, size_t start_pos) {
    static const char *const stop_patterns[] = {
        "<end_of_turn>",
        "<start_of_turn>",
        "\n\n",
        "```",
        "you>",
        "You>",
    };

    size_t best = std::string::npos;
    for (const char *pattern : stop_patterns) {
        size_t pos = text.find(pattern, start_pos);
        if (pos != std::string::npos && (best == std::string::npos || pos < best)) {
            best = pos;
        }
    }
    return best;
}

NezumiLlamaState* nezumi_llama_load(const char* model_path, int32_t n_ctx, int32_t n_gpu_layers) {
    llama_log_set(dummy_log_callback, nullptr);
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) return nullptr;

    auto cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(n_ctx > 0 ? n_ctx : 2048);
    cparams.n_batch = 512;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return nullptr;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(-1, 1.18f, 0.2f, 0.3f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto* state = new NezumiLlamaState();
    state->model = model;
    state->ctx = ctx;
    state->sampler = sampler;
    state->vocab = llama_model_get_vocab(model);
    return state;
}

int nezumi_llama_generate(NezumiLlamaState* state, const char* prompt, int32_t max_tokens, float temperature, NezumiTokenCallback cb, void* user_data) {
    if (!state || !state->ctx) return -1;
    const llama_vocab* vocab = state->vocab;

    const int prompt_len = static_cast<int>(strlen(prompt));
    std::vector<llama_token> tokens(prompt_len + 16);
    int n_tokens = llama_tokenize(vocab, prompt, prompt_len, tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, prompt, prompt_len, tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
    }
    if (n_tokens < 0) return -2;
    tokens.resize(static_cast<size_t>(n_tokens));

    llama_sampler_reset(state->sampler);

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(state->ctx, batch) != 0) return -3;

    const int32_t limit = max_tokens > 0 ? max_tokens : 512;
    char piece_buf[256];
    std::string generated;
    generated.reserve(1024);
    size_t emitted_len = 0;

    for (int32_t i = 0; i < limit; ++i) {
        llama_token token_id = llama_sampler_sample(state->sampler, state->ctx, -1);
        llama_sampler_accept(state->sampler, token_id);

        // 終了判定
        if (llama_vocab_is_eog(vocab, token_id)) break;

        int piece_len = llama_token_to_piece(vocab, token_id, piece_buf, static_cast<int32_t>(sizeof(piece_buf)) - 1, 0, false);
        if (piece_len < 0) continue;
        piece_buf[piece_len] = '\0';

        generated.append(piece_buf, static_cast<size_t>(piece_len));

        size_t stop_pos = find_stop_pos(generated, emitted_len);
        if (stop_pos != std::string::npos) {
            if (stop_pos > emitted_len) {
                std::string out = generated.substr(emitted_len, stop_pos - emitted_len);
                if (cb && cb(out.c_str(), user_data) != 0) break;
            }
            break;
        }

        if (generated.size() > emitted_len) {
            std::string out = generated.substr(emitted_len);
            if (cb && cb(out.c_str(), user_data) != 0) break;
            emitted_len = generated.size();
        }

        llama_batch next = llama_batch_get_one(&token_id, 1);
        if (llama_decode(state->ctx, next) != 0) break;
    }
    return 0;
}

void nezumi_llama_free(NezumiLlamaState* state) {
    if (!state) return;
    if (state->sampler) llama_sampler_free(state->sampler);
    if (state->ctx) llama_free(state->ctx);
    if (state->model) llama_model_free(state->model);
    llama_backend_free();
    delete state;
}