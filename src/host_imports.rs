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
