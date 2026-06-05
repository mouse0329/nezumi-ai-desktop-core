#include "litert_wrapper.h"
#include "engine.h"

#include <cstring>
#include <mutex>
#include <string>

static LiteRtLmEngine *g_engine = nullptr;
static LiteRtLmConversation *g_conversation = nullptr;
static std::string g_last_response;
static std::mutex g_mutex;

void litert_load_model(const char *path)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!path)
        return;

    LiteRtLmEngineSettings *settings =
        litert_lm_engine_settings_create(path, "cpu", nullptr, nullptr);
    if (!settings)
        return;

    LiteRtLmEngine *engine = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);
    if (!engine)
        return;

    if (g_conversation)
    {
        litert_lm_conversation_delete(g_conversation);
        g_conversation = nullptr;
    }
    if (g_engine)
    {
        litert_lm_engine_delete(g_engine);
        g_engine = nullptr;
    }

    g_engine = engine;
    g_conversation = litert_lm_conversation_create(g_engine, nullptr);
    if (!g_conversation)
    {
        litert_lm_engine_delete(g_engine);
        g_engine = nullptr;
    }
}

const char *litert_generate(const char *prompt)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_conversation || !prompt)
        return "";

    g_last_response.clear();

    // Build JSON message: {"role":"user","content":"<prompt>"}
    std::string msg = "{\"role\":\"user\",\"content\":\"";
    for (const char *p = prompt; *p; ++p)
    {
        if (*p == '"')
            msg += "\\\"";
        else if (*p == '\\')
            msg += "\\\\";
        else if (*p == '\n')
            msg += "\\n";
        else
            msg += *p;
    }
    msg += "\"}";

    LiteRtLmJsonResponse *resp =
        litert_lm_conversation_send_message(g_conversation, msg.c_str(), nullptr, nullptr);
    if (!resp)
        return "";

    const char *text = litert_lm_json_response_get_string(resp);
    if (text)
        g_last_response = text;
    litert_lm_json_response_delete(resp);

    return g_last_response.c_str();
}

void litert_free()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_conversation)
    {
        litert_lm_conversation_delete(g_conversation);
        g_conversation = nullptr;
    }
    if (g_engine)
    {
        litert_lm_engine_delete(g_engine);
        g_engine = nullptr;
    }
    g_last_response.clear();
}
