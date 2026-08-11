use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
mod ffi {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_name = customSyscall, catch)]
        pub fn custom_syscall(a0: i32, a1: i32, a2: i32, a3: i32, a7: i32) -> Result<i32, JsValue>;

        #[wasm_bindgen(js_name = jsExternalInterrupt)]
        pub fn js_external_interrupt() -> i32;

        #[wasm_bindgen(js_name = jsInterruptEnabled)]
        pub fn js_interrupt_enabled() -> i32;

        #[wasm_bindgen(js_name = jsGetIntInstDelay)]
        pub fn js_get_int_inst_delay() -> i32;

        #[wasm_bindgen(js_name = jsGetSleepDuration)]
        pub fn js_get_sleep_duration(sleep_type: i32) -> i32;

        #[wasm_bindgen(js_name = jsReadMMIO)]
        pub fn js_read_mmio(addr: u32, size: u32) -> u32;

        #[wasm_bindgen(js_name = jsWriteMMIO)]
        pub fn js_write_mmio(addr: u32, size: u32, val: u32);

        #[wasm_bindgen(js_name = jsSimStop)]
        pub fn js_sim_stop(snapshot: JsValue);

        #[wasm_bindgen(js_name = readFromStdin)]
        pub fn read_from_stdin(buf_ptr: *mut u8, count: u32) -> i32;

        #[wasm_bindgen(js_name = readInteractiveCommand)]
        pub fn read_interactive_command(pstr_ptr: *mut u8) -> i32;

        #[wasm_bindgen(js_name = jsPrint)]
        pub fn js_print(msg: &str);

        #[wasm_bindgen(js_name = jsPrintErr)]
        pub fn js_print_err(msg: &str);
    }
}

#[cfg(any(test, not(target_arch = "wasm32")))]
use std::cell::RefCell;

#[cfg(any(test, not(target_arch = "wasm32")))]
type CustomSyscallFn = Box<dyn Fn(i32, i32, i32, i32, i32) -> Result<i32, JsValue>>;
#[cfg(any(test, not(target_arch = "wasm32")))]
type MmioReadFn = Box<dyn Fn(u32, u32) -> u32>;
#[cfg(any(test, not(target_arch = "wasm32")))]
type MmioWriteFn = Box<dyn Fn(u32, u32, u32)>;

#[cfg(any(test, not(target_arch = "wasm32")))]
std::thread_local! {
    pub static MOCK_STDOUT: RefCell<Vec<String>> = RefCell::new(Vec::new());
    pub static MOCK_STDERR: RefCell<Vec<String>> = RefCell::new(Vec::new());
    pub static MOCK_STDIN: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    pub static MOCK_CUSTOM_SYSCALL: RefCell<Option<CustomSyscallFn>> = RefCell::new(None);
    pub static MOCK_MMIO_READ: RefCell<Option<MmioReadFn>> = RefCell::new(None);
    pub static MOCK_MMIO_WRITE: RefCell<Option<MmioWriteFn>> = RefCell::new(None);
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub fn reset_mocks() {
    MOCK_STDOUT.with(|s| s.borrow_mut().clear());
    MOCK_STDERR.with(|s| s.borrow_mut().clear());
    MOCK_STDIN.with(|s| s.borrow_mut().clear());
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
    F: Fn(i32, i32, i32, i32, i32) -> Result<i32, JsValue> + 'static,
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

pub fn custom_syscall(a0: i32, a1: i32, a2: i32, a3: i32, a7: i32) -> Result<i32, JsValue> {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        let handled = MOCK_CUSTOM_SYSCALL.with(|hook| {
            if let Some(ref f) = *hook.borrow() {
                Some(f(a0, a1, a2, a3, a7))
            } else {
                None
            }
        });
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
        Err(JsValue::NULL)
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

pub fn js_get_sleep_duration(_sleep_type: i32) -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_get_sleep_duration(_sleep_type)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

pub fn js_read_mmio(addr: u32, size: u32) -> u32 {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    {
        let handled = MOCK_MMIO_READ.with(|hook| {
            if let Some(ref f) = *hook.borrow() {
                Some(f(addr, size))
            } else {
                None
            }
        });
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

pub fn js_sim_stop(snapshot: JsValue) {
    #[cfg(target_arch = "wasm32")]
    {
        ffi::js_sim_stop(snapshot);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = snapshot;
    }
}

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
        return read_bytes;
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

