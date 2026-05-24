#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void litert_load_model(const char* path);
const char* litert_generate(const char* prompt);

#ifdef __cplusplus
}
#endif
