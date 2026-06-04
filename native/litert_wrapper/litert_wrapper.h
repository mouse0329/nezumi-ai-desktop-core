#pragma once
#ifdef __cplusplus
extern "C"
{
#endif

    void litert_load_model(const char *path);
    const char *litert_generate(const char *prompt);
    void litert_free();

#ifdef __cplusplus
}
#endif
