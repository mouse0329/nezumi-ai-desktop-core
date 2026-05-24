#include "llama_wrapper.h"
#include "llama.cpp/include/llama.h"

#include <cstring>
#include <string>
#include <vector>

struct NezumiLlamaState {
    llama_model*       model   = nullptr;
    llama_context*     ctx     = nullptr;
    llama_sampler*     sampler = nullptr;
    const llama_vocab* vocab   = nullptr;
};

NezumiLlamaState* nezumi_llama_load(const char* model_path, int32_t n_ctx, int32_t n_gpu_layers) {
    llama_backend_init();

    auto mparams         = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) return nullptr;

    auto cparams    = llama_context_default_params();
    cparams.n_ctx   = static_cast<uint32_t>(n_ctx > 0 ? n_ctx : 2048);
    cparams.n_batch = 512;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return nullptr;
    }

    // default sampler chain: top-k(40) -> top-p(0.9) -> temp(0.8) -> dist
    auto sparams   = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto* state    = new NezumiLlamaState();
    state->model   = model;
    state->ctx     = ctx;
    state->sampler = sampler;
    state->vocab   = llama_model_get_vocab(model);
    return state;
}

int nezumi_llama_generate(
    NezumiLlamaState*   state,
    const char*         prompt,
    int32_t             max_tokens,
    float               temperature,
    NezumiTokenCallback cb,
    void*               user_data
) {
    if (!state || !state->ctx) return -1;

    const llama_vocab* vocab = state->vocab;

    // tokenize
    const int prompt_len = static_cast<int>(strlen(prompt));
    std::vector<llama_token> tokens(prompt_len + 16);
    int n_tokens = llama_tokenize(vocab, prompt, prompt_len,
                                  tokens.data(), static_cast<int32_t>(tokens.size()),
                                  true, false);
    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, prompt, prompt_len,
                                  tokens.data(), static_cast<int32_t>(tokens.size()),
                                  true, false);
    }
    if (n_tokens < 0) return -2;
    tokens.resize(static_cast<size_t>(n_tokens));

    // clear KV cache and reset sampler
    llama_memory_clear(llama_get_memory(state->ctx), false);
    llama_sampler_reset(state->sampler);

    // process prompt batch
    {
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        if (llama_decode(state->ctx, batch) != 0) return -3;
    }

    // generation loop
    const int32_t limit = max_tokens > 0 ? max_tokens : 512;
    char piece_buf[256];

    for (int32_t i = 0; i < limit; ++i) {
        llama_token token_id = llama_sampler_sample(state->sampler, state->ctx, -1);
        llama_sampler_accept(state->sampler, token_id);

        if (llama_vocab_is_eog(vocab, token_id)) break;

        int piece_len = llama_token_to_piece(vocab, token_id,
                                             piece_buf, static_cast<int32_t>(sizeof(piece_buf)) - 1,
                                             0, false);
        if (piece_len < 0) continue;
        piece_buf[piece_len] = '\0';

        if (cb && cb(piece_buf, user_data) != 0) break;

        llama_batch next = llama_batch_get_one(&token_id, 1);
        if (llama_decode(state->ctx, next) != 0) break;
    }

    return 0;
}

void nezumi_llama_free(NezumiLlamaState* state) {
    if (!state) return;
    if (state->sampler) llama_sampler_free(state->sampler);
    if (state->ctx)     llama_free(state->ctx);
    if (state->model)   llama_model_free(state->model);
    llama_backend_free();
    delete state;
}
