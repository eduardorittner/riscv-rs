pub mod cli;
pub mod cpu;
mod host_imports;
pub mod memory;
mod syscall;

use cli::SimConfig;
pub use cpu::Cpu;
pub use memory::{Memory, MemoryOps};
use goblin::elf::Elf;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn run_whisper_binary(binary_bytes: &[u8], args_js: js_sys::Array) -> i32 {
    let mut args = Vec::new();
    for i in 0..args_js.length() {
        if let Some(val) = args_js.get(i).as_string() {
            args.push(val);
        }
    }

    let config = SimConfig::parse_args(&args);
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // Apply CLI register initializations
    for (reg_idx, val) in config.register_inits {
        cpu.write_reg(reg_idx, val);
    }

    // Try parsing ELF binary
    if let Ok(elf) = Elf::parse(binary_bytes) {
        let mut entry_point = elf.header.e_entry as u32;

        if !elf.program_headers.is_empty() {
            // Standard ELF Executable with Program Headers (PT_LOAD)
            for phdr in &elf.program_headers {
                if phdr.p_type == goblin::elf::program_header::PT_LOAD {
                    let offset = phdr.p_offset as usize;
                    let filesz = phdr.p_filesz as usize;
                    let vaddr = phdr.p_vaddr as u32;

                    if offset + filesz <= binary_bytes.len() {
                        let segment_bytes = &binary_bytes[offset..offset + filesz];
                        mem.write_bytes(vaddr, segment_bytes);
                    }
                }
            }
        } else {
            // Relocatable object file (ET_REL) without program headers
            let mut current_addr: u32 = 0x10000;
            let mut section_addrs = Vec::with_capacity(elf.section_headers.len());

            for shdr in &elf.section_headers {
                let size = shdr.sh_size as usize;
                let offset = shdr.sh_offset as usize;

                if size > 0 && (shdr.sh_flags & 2 != 0 || shdr.sh_type == goblin::elf::section_header::SHT_PROGBITS) {
                    let addr = if shdr.sh_addr != 0 {
                        shdr.sh_addr as u32
                    } else {
                        let align = if shdr.sh_addralign > 0 { shdr.sh_addralign as u32 } else { 4 };
                        current_addr = (current_addr + align - 1) & !(align - 1);
                        let assigned = current_addr;
                        current_addr += size as u32;
                        assigned
                    };
                    section_addrs.push(addr);

                    if shdr.sh_type != goblin::elf::section_header::SHT_NOBITS && offset + size <= binary_bytes.len() {
                        let bytes = &binary_bytes[offset..offset + size];
                        mem.write_bytes(addr, bytes);
                    }
                } else {
                    section_addrs.push(0);
                }
            }

            // Find entry point symbol (_start or main)
            for sym in &elf.syms {
                if let Some(name) = elf.strtab.get_at(sym.st_name) {
                    if name == "_start" || name == "main" {
                        let sh_idx = sym.st_shndx as usize;
                        let sec_base = if sh_idx < section_addrs.len() { section_addrs[sh_idx] } else { 0x10000 };
                        entry_point = sec_base.wrapping_add(sym.st_value as u32);
                        break;
                    }
                }
            }
            if entry_point == 0 {
                entry_point = 0x10000;
            }
        }

        // If symbol _start exists in executable ELF, override entry_point if needed
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

        cpu.pc = entry_point;
    } else {
        // Raw binary code fallback
        mem.write_bytes(0, binary_bytes);
        cpu.pc = 0;
    }

    // Execute simulation loop
    cpu.run(&mut mem)
}

#[wasm_bindgen]
pub fn main_entry() {}
