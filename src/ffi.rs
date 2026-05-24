use crate::{NezumiCore, error::{NezumiError, to_ffi_code}};
use futures::StreamExt;
use std::ffi::{c_char, c_void, CStr, CString};

/// コールバック型: 0で継続・非0で中断
pub type TokenCallback = extern "C" fn(token: *const c_char, user_data: *mut c_void) -> i32;

/// FFI向けストリーミング生成（panic-free）
///
/// # Safety
/// `core_ptr` は有効な `NezumiCore` ポインタ、`prompt` は有効なnull終端UTF-8文字列であること
#[no_mangle]
pub unsafe extern "C" fn nezumi_generate(
    core_ptr: *mut NezumiCore,
    prompt: *const c_char,
    cb: TokenCallback,
    user_data: *mut c_void,
) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let core = unsafe { &*core_ptr };
        let prompt_str = unsafe { CStr::from_ptr(prompt) }
            .to_str()
            .map_err(|_| NezumiError::FfiError("invalid utf8".into()))?;

        tokio::runtime::Handle::current().block_on(async {
            let mut stream = core.generate(prompt_str).await?;
            while let Some(token) = stream.next().await {
                let cstr = CString::new(token)
                    .map_err(|_| NezumiError::FfiError("null byte in token".into()))?;
                if cb(cstr.as_ptr(), user_data) != 0 {
                    break;
                }
            }
            Ok::<_, NezumiError>(())
        })
    }));

    match result {
        Ok(r) => to_ffi_code(&r),
        Err(_) => -99,
    }
}

/// コアを解放する（`nezumi_core_new` で確保した場合に使用）
///
/// # Safety
/// `core_ptr` は `Box::into_raw` で得たポインタであること
#[no_mangle]
pub unsafe extern "C" fn nezumi_core_free(core_ptr: *mut NezumiCore) {
    if !core_ptr.is_null() {
        unsafe { drop(Box::from_raw(core_ptr)) };
    }
}
