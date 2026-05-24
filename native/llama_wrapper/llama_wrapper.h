#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*NezumiProgressCallback)(float progress, void *user_data);
typedef int (*NezumiTokenCallback)(const char *token, void *user_data);

struct NezumiLlamaState;

NezumiLlamaState *nezumi_llama_load(
    const char *model_path,
    int32_t n_ctx,
    int32_t n_gpu_layers,
    NezumiProgressCallback progress_cb,
    void *progress_user_data
);

int nezumi_llama_generate(
    NezumiLlamaState *state,
    const char *prompt,
    int32_t max_tokens,
    float temperature,
    NezumiTokenCallback cb,
    void *user_data
);

void nezumi_llama_free(NezumiLlamaState *state);

#ifdef __cplusplus
}
#endif
