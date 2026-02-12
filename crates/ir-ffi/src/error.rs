use std::cell::RefCell;
use std::ffi::CString;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store an error message for later retrieval via `ir_last_error`.
pub fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

/// Take the last error message, leaving `None` in its place.
pub fn take_last_error() -> Option<CString> {
    LAST_ERROR.with(|e| e.borrow_mut().take())
}
