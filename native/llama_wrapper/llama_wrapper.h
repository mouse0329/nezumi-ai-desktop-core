#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void llama_load_model(const char* path);
const char* llama_generate(const char* prompt);

#ifdef __cplusplus
}
#endif
