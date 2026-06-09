#include "llama_wrapper.h"
#include "llama.cpp/include/llama.h"
#include <atomic>
#include <cstring>
#include <string>
#include <vector>

static std::atomic<int> g_backend_refs{0};

static void backend_init_once()
{
    if (g_backend_refs.fetch_add(1) == 0)
    {
        llama_backend_init();
    }
}

static void backend_free_once()
{
    if (g_backend_refs.fetch_sub(1) == 1)
    {
        llama_backend_free();
    }
}

struct StoredChatMessage
{
    std::string role;
    std::string content;
};

struct NezumiLlamaState
{
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_sampler *sampler = nullptr;
    const llama_vocab *vocab = nullptr;
    int32_t n_ctx = 2048;
    float temperature = 0.8f;
    // Text that has actually been decoded into ctx, including generated tokens.
    std::string last_prompt;
    std::string chat_template;
    std::vector<StoredChatMessage> chat_history;
    std::vector<llama_chat_message> chat_messages;
};

static llama_sampler *create_sampler(float temperature)
{
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(-1, 1.18f, 0.2f, 0.3f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return sampler;
}

// ���O�����S�ɖق点��
void dummy_log_callback(ggml_log_level level, const char *text, void *user_data)
{
    (void)level;
    (void)text;
    (void)user_data;
}

static size_t find_stop_pos(const std::string &text, size_t start_pos)
{
    static const char *const stop_patterns[] = {
        "<end_of_turn>",
        "<start_of_turn>",
        "<|im_end|>",
    };

    size_t best = std::string::npos;
    for (const char *pattern : stop_patterns)
    {
        size_t pos = text.find(pattern, start_pos);
        if (pos != std::string::npos && (best == std::string::npos || pos < best))
        {
            best = pos;
        }
    }
    return best;
}

static constexpr size_t STOP_LOOKBEHIND = 32;

static size_t utf8_truncate_to_boundary(const std::string &text, size_t max_len)
{
    if (max_len >= text.size())
    {
        return text.size();
    }

    while (max_len > 0 && (static_cast<unsigned char>(text[max_len]) & 0xC0) == 0x80)
    {
        --max_len;
    }
    return max_len;
}

static const char *normalize_chat_role(const char *role)
{
    if (role && std::strcmp(role, "model") == 0)
    {
        return "assistant";
    }
    return role ? role : "";
}

static bool apply_chat_template(
    const NezumiLlamaState *state,
    const std::vector<llama_chat_message> &chat,
    bool add_ass,
    std::string &out)
{
    const char *tmpl = state->chat_template.empty() ? nullptr : state->chat_template.c_str();

    size_t est = 256;
    for (const auto &msg : chat)
    {
        est += msg.role ? std::strlen(msg.role) : 0;
        est += msg.content ? std::strlen(msg.content) : 0;
    }
    est *= 4; // 日本語等マルチバイト文字（1文字最大4バイト）を考慮

    std::vector<char> buf(est);
    int32_t len = llama_chat_apply_template(
        tmpl,
        chat.data(),
        chat.size(),
        add_ass,
        buf.data(),
        static_cast<int32_t>(buf.size()));
    if (len < 0)
    {
        return false;
    }
    if (len > static_cast<int32_t>(buf.size()))
    {
        buf.resize(static_cast<size_t>(len));
        len = llama_chat_apply_template(
            tmpl,
            chat.data(),
            chat.size(),
            add_ass,
            buf.data(),
            len);
        if (len < 0)
        {
            return false;
        }
    }

    out.assign(buf.data(), static_cast<size_t>(len));
    return true;
}

static void update_chat_history(NezumiLlamaState *state, const NezumiChatMessage *messages, size_t n_messages)
{
    state->chat_history.clear();
    state->chat_messages.clear();
    state->chat_history.reserve(n_messages);
    state->chat_messages.reserve(n_messages);

    for (size_t i = 0; i < n_messages; ++i)
    {
        state->chat_history.push_back({
            messages[i].role ? messages[i].role : "",
            messages[i].content ? messages[i].content : "",
        });
        const auto &stored = state->chat_history.back();
        state->chat_messages.push_back({
            normalize_chat_role(stored.role.c_str()),
            stored.content.c_str(),
        });
    }
}

static bool reset_llama_context(NezumiLlamaState *state)
{
    if (state->ctx)
    {
        llama_free(state->ctx);
        state->ctx = nullptr;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(state->n_ctx > 0 ? state->n_ctx : 2048);
    cparams.n_batch = cparams.n_ctx;

    state->ctx = llama_init_from_model(state->model, cparams);
    state->last_prompt.clear();
    return state->ctx != nullptr;
}

extern "C" NezumiLlamaState *nezumi_llama_load(const char *model_path, int32_t n_ctx, int32_t n_gpu_layers, NezumiProgressCallback progress_cb, void *progress_user_data)
{
    llama_log_set(dummy_log_callback, nullptr);
    backend_init_once();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    if (progress_cb)
    {
        mparams.progress_callback = [](float progress, void *ctx) -> bool
        {
            auto *cb = reinterpret_cast<NezumiProgressCallback>(ctx);
            cb(progress, nullptr);
            return true;
        };
        mparams.progress_callback_user_data = reinterpret_cast<void *>(progress_cb);
    }

    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model)
        return nullptr;

    auto *state = new NezumiLlamaState();
    state->model = model;
    state->vocab = llama_model_get_vocab(model);
    state->n_ctx = n_ctx > 0 ? n_ctx : 2048;

    if (!reset_llama_context(state))
    {
        llama_model_free(model);
        delete state;
        return nullptr;
    }

    const char *tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl)
    {
        state->chat_template = tmpl;
    }

    state->sampler = create_sampler(state->temperature);
    return state;
}

static int generate_from_prompt(
    NezumiLlamaState *state,
    const std::string &prompt_str,
    int32_t max_tokens,
    float temperature,
    NezumiTokenCallback cb,
    void *user_data)
{
    if (!state)
        return -1;
    const llama_vocab *vocab = state->vocab;
    std::string input_to_decode;
    bool add_bos = false;
    if (!state->last_prompt.empty() && prompt_str.rfind(state->last_prompt, 0) == 0)
    {
        input_to_decode = prompt_str.substr(state->last_prompt.size());
    }
    else
    {
        if (!state->last_prompt.empty())
        {
            if (!reset_llama_context(state))
                return -4;
        }
        input_to_decode = prompt_str;
        add_bos = true;
    }

    if (state->temperature != temperature)
    {
        if (state->sampler)
            llama_sampler_free(state->sampler);
        state->sampler = create_sampler(temperature);
        state->temperature = temperature;
    }

    llama_sampler_reset(state->sampler);

    if (!input_to_decode.empty())
    {
        const int prompt_len = static_cast<int>(input_to_decode.size());
        std::vector<llama_token> tokens(prompt_len + 16);
        int n_tokens = llama_tokenize(vocab, input_to_decode.c_str(), prompt_len, tokens.data(), static_cast<int32_t>(tokens.size()), add_bos, false);
        if (n_tokens < 0)
        {
            tokens.resize(static_cast<size_t>(-n_tokens));
            n_tokens = llama_tokenize(vocab, input_to_decode.c_str(), prompt_len, tokens.data(), static_cast<int32_t>(tokens.size()), add_bos, false);
        }
        if (n_tokens < 0)
            return -2;
        tokens.resize(static_cast<size_t>(n_tokens));

        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        if (llama_decode(state->ctx, batch) != 0)
            return -3;
    }

    state->last_prompt = prompt_str;

    const int32_t limit = max_tokens > 0 ? max_tokens : 512;
    char piece_buf[256];
    std::string generated;
    generated.reserve(1024);
    size_t emitted_len = 0;
    bool stopped = false;

    for (int32_t i = 0; i < limit; ++i)
    {
        llama_token token_id = llama_sampler_sample(state->sampler, state->ctx, -1);
        llama_sampler_accept(state->sampler, token_id);

        if (llama_vocab_is_eog(vocab, token_id))
        {
            break;
        }

        int piece_len = llama_token_to_piece(vocab, token_id, piece_buf, static_cast<int32_t>(sizeof(piece_buf)) - 1, 0, false);
        if (piece_len < 0)
            continue;
        piece_buf[piece_len] = '\0';

        generated.append(piece_buf, static_cast<size_t>(piece_len));

        size_t search_start = emitted_len > STOP_LOOKBEHIND ? emitted_len - STOP_LOOKBEHIND : 0;
        size_t stop_pos = find_stop_pos(generated, search_start);
        if (stop_pos != std::string::npos)
        {
            if (stop_pos > emitted_len)
            {
                size_t safe_pos = utf8_truncate_to_boundary(generated, stop_pos);
                if (safe_pos > emitted_len)
                {
                    std::string out = generated.substr(emitted_len, safe_pos - emitted_len);
                    if (cb && cb(out.c_str(), user_data) != 0)
                        break;
                }
            }
            bool is_leading_start_marker = stop_pos == 0 && emitted_len == 0 &&
                                           (generated.rfind("<start_of_turn>", 0) == 0 || generated.rfind("<|im_start|>", 0) == 0);
            if (is_leading_start_marker)
            {
                // Allow a response to begin with a chat-start marker like
                // <start_of_turn>model or <|im_start|>. The caller will strip
                // these markers when rendering the response.
            }
            else
            {
                stopped = true;
                break;
            }
        }

        size_t safe_len = generated.size() > STOP_LOOKBEHIND ? generated.size() - STOP_LOOKBEHIND : 0;
        safe_len = utf8_truncate_to_boundary(generated, safe_len);
        if (safe_len > emitted_len)
        {
            std::string out = generated.substr(emitted_len, safe_len - emitted_len);
            if (cb && cb(out.c_str(), user_data) != 0)
                break;
            emitted_len = safe_len;
        }

        llama_batch next = llama_batch_get_one(&token_id, 1);
        if (llama_decode(state->ctx, next) != 0)
            break;
        state->last_prompt.append(piece_buf, static_cast<size_t>(piece_len));
    }

    if (!stopped && generated.size() > emitted_len)
    {
        size_t final_len = utf8_truncate_to_boundary(generated, generated.size());
        if (final_len > emitted_len)
        {
            std::string out = generated.substr(emitted_len, final_len - emitted_len);
            if (cb)
                cb(out.c_str(), user_data);
        }
    }
    return 0;
}

extern "C" int nezumi_llama_generate(NezumiLlamaState *state, const char *prompt, int32_t max_tokens, float temperature, NezumiTokenCallback cb, void *user_data)
{
    if (!state)
        return -1;

    std::string prompt_str(prompt ? prompt : "");
    return generate_from_prompt(state, prompt_str, max_tokens, temperature, cb, user_data);
}

extern "C" int nezumi_llama_chat(
    NezumiLlamaState *state,
    const NezumiChatMessage *messages,
    size_t n_messages,
    int32_t max_tokens,
    float temperature,
    NezumiTokenCallback cb,
    void *user_data)
{
    if (!state)
        return -1;
    if (!messages && n_messages > 0)
        return -5;

    update_chat_history(state, messages, n_messages);

    std::string prompt;
    if (!apply_chat_template(state, state->chat_messages, true, prompt))
    {
        // fprintf(stderr, "[nezumi_llama_chat] apply_chat_template failed (template=%s)\n",
        //         state->chat_template.empty() ? "(empty/nullptr)" : state->chat_template.substr(0, 40).c_str());
        return -5;
    }

    // add_ass=trueでもQwen3.5等でassistantが付かない場合があるので強制補完
    // "<|im_start|>\n" で終わっていたら "<|im_start|>assistant\n" に修正
    {
        const std::string tail_bare = "<|im_start|>\n";
        const std::string tail_ass = "<|im_start|>assistant\n";
        if (prompt.size() >= tail_bare.size() &&
            prompt.compare(prompt.size() - tail_bare.size(), tail_bare.size(), tail_bare) == 0)
        {
            prompt.replace(prompt.size() - tail_bare.size(), tail_bare.size(), tail_ass);
        }
    }

    return generate_from_prompt(state, prompt, max_tokens, temperature, cb, user_data);
}

extern "C" void nezumi_llama_free(NezumiLlamaState *state)
{
    if (!state)
        return;
    if (state->sampler)
        llama_sampler_free(state->sampler);
    if (state->ctx)
        llama_free(state->ctx);
    if (state->model)
        llama_model_free(state->model);
    backend_free_once();
    delete state;
}
