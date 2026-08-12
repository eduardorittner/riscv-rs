pub mod cli;
pub mod cpu;
pub mod disasm;
pub mod host_imports;
pub mod inst;
pub mod memory;
pub mod syscall;
pub mod utils;

use cli::SimConfig;
pub use cpu::{Cpu, CpuError, StepResult};
use disasm::Disassembler;
use goblin::elf::Elf;
pub use inst::{DecodedInst16, DecodedInst32};
pub use memory::{Memory, MemoryOps};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
pub use utils::{shift_then_mask, ShiftThenMask};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DebuggerSnapshot {
    pub pc: u32,
    pub gpr: Vec<u32>,  // 32 GP registers
    pub fpr: Vec<f64>,  // 32 FP registers
    pub csrs: Vec<u32>, // mstatus, mcause, mepc, mtvec, fcsr
    pub step_count: u64,
    pub is_halted: bool,
    pub is_breakpoint: bool,
    pub hit_address: u32,
}

#[wasm_bindgen]
pub struct Simulator {
    cpu: Cpu,
    mem: Memory,
    symbols: HashMap<u32, String>,
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Simulator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cpu: Cpu::new(),
            mem: Memory::new(),
            symbols: HashMap::new(),
        }
    }

    pub fn load_binary(&mut self, binary_bytes: &[u8], args_js: js_sys::Array) -> u32 {
        let mut args = Vec::new();
        for i in 0..args_js.length() {
            if let Some(val) = args_js.get(i).as_string() {
                args.push(val);
            }
        }

        let config = SimConfig::parse_args(&args);
        self.cpu = Cpu::new();
        self.mem = Memory::new();
        self.symbols.clear();

        for (reg_idx, val) in config.register_inits {
            self.cpu.write_reg(reg_idx, val);
        }

        let mut entry_point = 0u32;
        if let Ok(elf) = Elf::parse(binary_bytes) {
            entry_point = elf.header.e_entry as u32;

            if !elf.program_headers.is_empty() {
                for phdr in &elf.program_headers {
                    if phdr.p_type == goblin::elf::program_header::PT_LOAD {
                        let offset = phdr.p_offset as usize;
                        let filesz = phdr.p_filesz as usize;
                        let vaddr = phdr.p_vaddr as u32;

                        if offset + filesz <= binary_bytes.len() {
                            let segment_bytes = &binary_bytes[offset..offset + filesz];
                            self.mem.write_bytes(vaddr, segment_bytes);
                        }
                    }
                }

                for sym in &elf.syms {
                    if let Some(name) = elf.strtab.get_at(sym.st_name) {
                        if !name.is_empty() && sym.st_value != 0 {
                            self.symbols.insert(sym.st_value as u32, name.to_string());
                        }
                    }
                }
            } else {
                let mut current_addr: u32 = 0x10000;
                let mut section_addrs = Vec::with_capacity(elf.section_headers.len());

                for shdr in &elf.section_headers {
                    let size = shdr.sh_size as usize;
                    let offset = shdr.sh_offset as usize;

                    if size > 0
                        && (shdr.sh_flags & 2 != 0
                            || shdr.sh_type == goblin::elf::section_header::SHT_PROGBITS)
                    {
                        let addr = if shdr.sh_addr != 0 {
                            shdr.sh_addr as u32
                        } else {
                            let align = if shdr.sh_addralign > 0 {
                                shdr.sh_addralign as u32
                            } else {
                                4
                            };
                            current_addr = (current_addr + align - 1) & !(align - 1);
                            let assigned = current_addr;
                            current_addr += size as u32;
                            assigned
                        };
                        section_addrs.push(addr);

                        if shdr.sh_type != goblin::elf::section_header::SHT_NOBITS
                            && offset + size <= binary_bytes.len()
                        {
                            let bytes = &binary_bytes[offset..offset + size];
                            self.mem.write_bytes(addr, bytes);
                        }
                    } else {
                        section_addrs.push(0);
                    }
                }

                for sym in &elf.syms {
                    if let Some(name) = elf.strtab.get_at(sym.st_name) {
                        if !name.is_empty() {
                            let sh_idx = sym.st_shndx;
                            let sec_base = if sh_idx < section_addrs.len() {
                                section_addrs[sh_idx]
                            } else {
                                0x10000
                            };
                            let addr = sec_base.wrapping_add(sym.st_value as u32);
                            self.symbols.insert(addr, name.to_string());

                            if (name == "_start" || name == "main") && entry_point == 0 {
                                entry_point = addr;
                            }
                        }
                    }
                }
                if entry_point == 0 {
                    entry_point = 0x10000;
                }
            }

            if elf.header.e_entry != 0 {
                for sym in &elf.syms {
                    if let Some(name) = elf.strtab.get_at(sym.st_name) {
                        if name == "_start" && sym.st_value != 0 {
                            entry_point = sym.st_value as u32;
                            break;
                        }
                    }
                }
            }

            self.cpu.pc = entry_point;
        } else {
            self.mem.write_bytes(0, binary_bytes);
            self.cpu.pc = 0;
        }

        entry_point
    }

    pub fn run_full(&mut self) -> i32 {
        self.cpu.run(&mut self.mem)
    }

    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.cpu.debug_enabled = enabled;
    }

    pub fn add_breakpoint(&mut self, addr: u32) {
        self.cpu.breakpoints.insert(addr);
    }

    pub fn remove_breakpoint(&mut self, addr: u32) {
        self.cpu.breakpoints.remove(&addr);
    }

    pub fn clear_breakpoints(&mut self) {
        self.cpu.breakpoints.clear();
    }

    pub fn get_snapshot_js(&self, is_breakpoint: bool, hit_address: u32) -> JsValue {
        self.cpu.get_snapshot_js(is_breakpoint, hit_address)
    }

    pub fn debug_step(&mut self) -> JsValue {
        let res = self.cpu.step_instruction(&mut self.mem);
        let (is_bp, hit_addr) = match res {
            StepResult::BreakpointHit(addr) => (true, addr),
            _ => (false, 0),
        };
        self.get_snapshot_js(is_bp, hit_addr)
    }

    pub fn debug_step_over(&mut self) -> JsValue {
        let current_pc = self.cpu.pc;
        let inst = self.mem.read_u32(current_pc);
        let is_call = is_call_instruction(inst);

        if is_call {
            let next_pc = current_pc + if (inst & 0x3) == 0x3 { 4 } else { 2 };
            let had_bp = self.cpu.breakpoints.contains(&next_pc);
            self.cpu.breakpoints.insert(next_pc);
            let snap = self.run_until_breakpoint();
            if !had_bp {
                self.cpu.breakpoints.remove(&next_pc);
            }
            snap
        } else {
            self.debug_step()
        }
    }

    pub fn debug_step_out(&mut self) -> JsValue {
        let ra = self.cpu.regs[1]; // x1 / ra
        let had_bp = self.cpu.breakpoints.contains(&ra);
        self.cpu.breakpoints.insert(ra);
        let snap = self.run_until_breakpoint();
        if !had_bp {
            self.cpu.breakpoints.remove(&ra);
        }
        snap
    }

    pub fn run_until_breakpoint(&mut self) -> JsValue {
        loop {
            let res = self.cpu.step_instruction(&mut self.mem);
            match res {
                StepResult::BreakpointHit(addr) => {
                    return self.get_snapshot_js(true, addr);
                }
                StepResult::Halted(_) | StepResult::Trap(_) => {
                    return self.get_snapshot_js(false, 0);
                }
                StepResult::Ok => {}
            }
        }
    }

    pub fn read_memory_range(&self, addr: u32, len: u32) -> Vec<u8> {
        let mut buf = vec![0u8; len as usize];
        for i in 0..len {
            buf[i as usize] = self.mem.read_u8(addr.wrapping_add(i));
        }
        buf
    }

    pub fn write_memory_byte(&mut self, addr: u32, val: u8) {
        self.mem.write_u8(addr, val);
    }

    pub fn write_register(&mut self, reg_idx: usize, val: u32) {
        if reg_idx < 32 {
            self.cpu.write_reg(reg_idx, val);
        }
    }

    pub fn disassemble_range(&self, start_addr: u32, len: u32) -> JsValue {
        let mut disasm_list = Vec::new();
        let mut curr = start_addr;
        while curr < start_addr.saturating_add(len) {
            let inst = self.mem.read_u32(curr);
            let item =
                Disassembler::decode_instruction_with_symbols(curr, inst, Some(&self.symbols));
            let step = if item.is_compressed { 2 } else { 4 };
            disasm_list.push(item);
            curr = curr.saturating_add(step);
        }
        serde_wasm_bindgen::to_value(&disasm_list).unwrap_or(JsValue::NULL)
    }

    pub fn get_symbol_at(&self, addr: u32) -> Option<String> {
        self.symbols.get(&addr).cloned()
    }
}

fn is_call_instruction(inst: u32) -> bool {
    if (inst & 0x3) != 0x3 {
        let op2 = inst & 0x3;
        let funct3 = inst.shift_then_mask(13, 0x7);
        if op2 == 0x1 && funct3 == 0x1 {
            return true; // c.jal
        }
        if op2 == 0x2 && funct3 == 0x4 && (inst.shift_then_mask(12, 1) == 1) {
            return true; // c.jalr
        }
        return false;
    }
    let decoded = DecodedInst32::decode(inst);
    if decoded.opcode == inst::OP_JAL || decoded.opcode == inst::OP_JALR {
        return decoded.rd == 1 || decoded.rd == 5;
    }
    false
}

#[wasm_bindgen]
pub fn run_riscv_binary(binary_bytes: &[u8], args_js: js_sys::Array) -> i32 {
    let mut sim = Simulator::new();
    sim.load_binary(binary_bytes, args_js);
    sim.run_full()
}
