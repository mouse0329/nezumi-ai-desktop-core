#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NezumiLlamaState NezumiLlamaState;

/// コールバック: token文字列を受け取り 0=継続 / 非0=中断
typedef int (*NezumiTokenCallback)(const char* token, void* user_data);

/// モデルをロードしてStateを返す。失敗時はNULL
NezumiLlamaState* nezumi_llama_load(const char* model_path, int32_t n_ctx, int32_t n_gpu_layers);

/// ストリーミング生成。0=成功 / 負=エラー
int nezumi_llama_generate(
    NezumiLlamaState* state,
    const char*       prompt,
    int32_t           max_tokens,
    float             temperature,
    NezumiTokenCallback cb,
    void*             user_data
);

/// Stateを解放
void nezumi_llama_free(NezumiLlamaState* state);

#ifdef __cplusplus
}
#endif
