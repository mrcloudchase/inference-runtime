use std::ffi::CString;
use crate::types::IRStreamCallback;

/// Invoke a streaming callback with a token string.
///
/// Returns `true` if generation should continue, `false` to stop.
/// If there is no callback, returns `true` (continue).
pub fn invoke_callback(
    callback: IRStreamCallback,
    user_data: *mut std::os::raw::c_void,
    token_text: &str,
) -> bool {
    match callback {
        Some(cb) => {
            if let Ok(c_str) = CString::new(token_text) {
                cb(c_str.as_ptr(), user_data)
            } else {
                true // continue on encoding error
            }
        }
        None => true, // no callback, continue
    }
}
