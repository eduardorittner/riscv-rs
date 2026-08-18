#[cfg(target_arch = "wasm32")]
mod ffi {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_name = customSyscall)]
        pub fn custom_syscall(a0: i32, a1: i32, a2: i32, a3: i32, a7: i32) -> i32;

        #[wasm_bindgen(js_name = jsExternalInterrupt)]
        pub fn js_external_interrupt() -> i32;

        #[wasm_bindgen(js_name = jsInterruptEnabled)]
        pub fn js_interrupt_enabled() -> i32;

        #[wasm_bindgen(js_name = jsGetIntInstDelay)]
        pub fn js_get_int_inst_delay() -> i32;

        #[wasm_bindgen(js_name = jsReadMMIO)]
        pub fn js_read_mmio(addr: u32, size: u32) -> u32;

        #[wasm_bindgen(js_name = jsWriteMMIO)]
        pub fn js_write_mmio(addr: u32, size: u32, val: u32);

        #[wasm_bindgen(js_name = readFromStdin)]
        pub fn read_from_stdin(buf_ptr: *mut u8, count: u32) -> i32;

        #[wasm_bindgen(js_name = readInteractiveCommand)]
        pub fn read_interactive_command(pstr_ptr: *mut u8) -> i32;

        #[wasm_bindgen(js_name = jsPrint)]
        pub fn js_print(msg: &str);

        #[wasm_bindgen(js_name = jsPrintErr)]
        pub fn js_print_err(msg: &str);

        /// Guest `write()` to fd 1, as raw bytes.
        #[wasm_bindgen(js_name = jsWriteStdout)]
        pub fn js_write_stdout(bytes: &[u8]);

        /// Guest `write()` to fd 2, as raw bytes.
        #[wasm_bindgen(js_name = jsWriteStderr)]
        pub fn js_write_stderr(bytes: &[u8]);

        #[wasm_bindgen(js_name = notifyUnknownSyscall)]
        pub fn notify_unknown_syscall(sys_num: u32, a0: u32, a1: u32, a2: u32, a3: u32);
    }
}

#[cfg(any(test, not(target_arch = "wasm32")))]
use std::cell::RefCell;

#[cfg(any(test, not(target_arch = "wasm32")))]
type CustomSyscallFn = Box<dyn Fn(i32, i32, i32, i32, i32) -> i32>;
#[cfg(any(test, not(target_arch = "wasm32")))]
type MmioReadFn = Box<dyn Fn(u32, u32) -> u32>;
#[cfg(any(test, not(target_arch = "wasm32")))]
type MmioWriteFn = Box<dyn Fn(u32, u32, u32)>;

#[cfg(any(test, not(target_arch = "wasm32")))]
std::thread_local! {
    pub static MOCK_STDOUT: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    pub static MOCK_STDERR: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    pub static MOCK_STDIN: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    pub static MOCK_CUSTOM_SYSCALL: RefCell<Option<CustomSyscallFn>> = const { RefCell::new(None) };
    pub static MOCK_MMIO_READ: RefCell<Option<MmioReadFn>> = const { RefCell::new(None) };
    pub static MOCK_MMIO_WRITE: RefCell<Option<MmioWriteFn>> = const { RefCell::new(None) };
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn reset_mocks() {
    MOCK_STDOUT.with(|s| s.borrow_mut().clear());
    MOCK_STDERR.with(|s| s.borrow_mut().clear());
    MOCK_STDIN.with(|s| *s.borrow_mut() = Vec::new());
    MOCK_CUSTOM_SYSCALL.with(|s| *s.borrow_mut() = None);
    MOCK_MMIO_READ.with(|s| *s.borrow_mut() = None);
    MOCK_MMIO_WRITE.with(|s| *s.borrow_mut() = None);
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn get_mock_stdout() -> Vec<String> {
    MOCK_STDOUT.with(|s| s.borrow().clone())
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn get_mock_stderr() -> Vec<String> {
    MOCK_STDERR.with(|s| s.borrow().clone())
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn set_mock_stdin(bytes: &[u8]) {
    MOCK_STDIN.with(|s| *s.borrow_mut() = bytes.to_vec());
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn set_mock_custom_syscall<F>(f: F)
where
    F: Fn(i32, i32, i32, i32, i32) -> i32 + 'static,
{
    MOCK_CUSTOM_SYSCALL.with(|s| *s.borrow_mut() = Some(Box::new(f)));
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn set_mock_mmio_read<F>(f: F)
where
    F: Fn(u32, u32) -> u32 + 'static,
{
    MOCK_MMIO_READ.with(|s| *s.borrow_mut() = Some(Box::new(f)));
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn set_mock_mmio_write<F>(f: F)
where
    F: Fn(u32, u32, u32) + 'static,
{
    MOCK_MMIO_WRITE.with(|s| *s.borrow_mut() = Some(Box::new(f)));
}

pub fn custom_syscall(a0: i32, a1: i32, a2: i32, a3: i32, a7: i32) -> i32 {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        let handled =
            MOCK_CUSTOM_SYSCALL.with(|hook| hook.borrow().as_ref().map(|f| f(a0, a1, a2, a3, a7)));
        if let Some(res) = handled {
            return res;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        ffi::custom_syscall(a0, a1, a2, a3, a7)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

pub fn js_external_interrupt() -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_external_interrupt()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

pub fn js_interrupt_enabled() -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_interrupt_enabled()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

pub fn js_get_int_inst_delay() -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_get_int_inst_delay()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

pub fn js_read_mmio(addr: u32, size: u32) -> u32 {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        let handled = MOCK_MMIO_READ.with(|hook| hook.borrow().as_ref().map(|f| f(addr, size)));
        if let Some(val) = handled {
            return val;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_read_mmio(addr, size)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

pub fn js_write_mmio(addr: u32, size: u32, val: u32) {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        let handled = MOCK_MMIO_WRITE.with(|hook| {
            if let Some(ref f) = *hook.borrow() {
                f(addr, size, val);
                true
            } else {
                false
            }
        });
        if handled {
            return;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_write_mmio(addr, size, val);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {}
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn read_from_stdin(buf_ptr: *mut u8, count: u32) -> i32 {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        if buf_ptr.is_null() || count == 0 {
            return 0;
        }
        let read_bytes = MOCK_STDIN.with(|stdin| {
            let mut stdin = stdin.borrow_mut();
            let to_read = (stdin.len() as u32).min(count) as usize;
            if to_read > 0 {
                let bytes: Vec<u8> = stdin.drain(..to_read).collect();
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf_ptr, to_read);
                }
                to_read as i32
            } else {
                0
            }
        });
        read_bytes
    }
    #[cfg(target_arch = "wasm32")]
    {
        ffi::read_from_stdin(buf_ptr, count)
    }
}

pub fn read_interactive_command(pstr_ptr: *mut u8) -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        ffi::read_interactive_command(pstr_ptr)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = pstr_ptr;
        0
    }
}

pub fn js_print(msg: &str) {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        MOCK_STDOUT.with(|s| s.borrow_mut().push(msg.to_string()));
    }
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_print(msg);
    }
}

pub fn js_print_err(msg: &str) {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        MOCK_STDERR.with(|s| s.borrow_mut().push(msg.to_string()));
    }
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_print_err(msg);
    }
}

/// A guest write to stdout, byte for byte.
///
/// Separate from `js_print`, which carries the simulator's own diagnostic
/// messages. Guest output must cross as bytes: it is not always valid UTF-8,
/// and a multi-byte sequence can be split across two `write()` calls. The
/// host decodes with a streaming decoder that holds the partial sequence.
pub fn js_write_stdout(bytes: &[u8]) {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        MOCK_STDOUT.with(|s| {
            s.borrow_mut()
                .push(String::from_utf8_lossy(bytes).into_owned())
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_write_stdout(bytes);
    }
}

/// A guest write to stderr, byte for byte. See `js_write_stdout`.
pub fn js_write_stderr(bytes: &[u8]) {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        MOCK_STDERR.with(|s| {
            s.borrow_mut()
                .push(String::from_utf8_lossy(bytes).into_owned())
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_write_stderr(bytes);
    }
}

pub fn notify_unknown_syscall(sys_num: u32, a0: u32, a1: u32, a2: u32, a3: u32) {
    #[cfg(target_arch = "wasm32")]
    {
        ffi::notify_unknown_syscall(sys_num, a0, a1, a2, a3);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (sys_num, a0, a1, a2, a3);
    }
}
