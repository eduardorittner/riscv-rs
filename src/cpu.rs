use wasm_bindgen::JsValue;

use crate::inst::*;
use crate::memory::MemoryOps;
use crate::syscall::handle_ecall;
use crate::utils::ShiftThenMask;
use crate::{host_imports, DebuggerSnapshot};
use std::collections::HashMap;

const NAN_F32: u32 = 0x7FC00000;
const NAN_F64: u64 = 0xFFFFFFFF7FC00000;

/// Address of the first instruction to be executed after handling the current trap.
const MEPC: u16 = 0x341;
/// Event type that caused the current trap.
const MCAUSE: u16 = 0x341;
/// Address of the trap-handler's first instruction.
const MTVEC: u16 = 0x305;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuError {
    IllegalInstruction { pc: u32, raw: u32 },
    UnknownOpcode { pc: u32, opcode: u8 },
    UnalignedAccess { pc: u32, addr: u32 },
    MemoryFault { pc: u32, addr: u32 },
    UnhandledSyscall { pc: u32, number: i32 },
    ExecutionError(String),
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllegalInstruction { pc, raw } => {
                write!(f, "Illegal instruction {:#010x} at PC {:#010x}", raw, pc)
            }
            Self::UnknownOpcode { pc, opcode } => {
                write!(f, "Unknown opcode {:#04x} at PC {:#010x}", opcode, pc)
            }
            Self::UnalignedAccess { pc, addr } => {
                write!(f, "Unaligned memory access at {:#010x} (PC {:#010x})", addr, pc)
            }
            Self::MemoryFault { pc, addr } => {
                write!(f, "Memory fault at address {:#010x} (PC {:#010x})", addr, pc)
            }
            Self::UnhandledSyscall { pc, number } => {
                write!(f, "Unhandled syscall {} at PC {:#010x}", number, pc)
            }
            Self::ExecutionError(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for CpuError {}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum StepResult {
    Ok,
    BreakpointHit(u32),
    Halted(i32),
    Trap(u32),
}

pub struct Cpu {
    pub regs: [u32; 32],
    pub fregs: [f64; 32],
    pub pc: u32,
    pub fcsr: u32,
    pub csrs: HashMap<u16, u32>,
    pub is_halted: bool,
    pub exit_code: i32,
    pub debug_enabled: bool,
    pub breakpoints: std::collections::HashSet<u32>,
    pub step_counter: u64,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        let mut cpu = Self {
            regs: [0; 32],
            fregs: [0.0; 32],
            pc: 0,
            fcsr: 0,
            csrs: HashMap::new(),
            is_halted: false,
            exit_code: 0,
            debug_enabled: false,
            breakpoints: std::collections::HashSet::new(),
            step_counter: 0,
        };
        // Default Stack Pointer sp (x2) if not set by CLI
        cpu.regs[2] = 0x7FFFFFC;
        cpu
    }

    #[inline(always)]
    pub fn step_instruction<M: MemoryOps>(&mut self, mem: &mut M) -> StepResult {
        if self.debug_enabled && self.breakpoints.contains(&self.pc) {
            return StepResult::BreakpointHit(self.pc);
        }

        if self.is_halted {
            return StepResult::Halted(self.exit_code);
        }

        let pc = self.pc;
        let inst16 = mem.read_u16(pc);

        let instruction_result = if (inst16 & 0x3) != 0x3 {
            self.execute_inst16(inst16, mem)
        } else {
            let inst32 = mem.read_u32(pc);
            self.execute_inst32(inst32, mem)
        };

        self.step_counter += 1;

        if let Err(_err) = instruction_result {
            return StepResult::Trap(pc);
        }

        if self.is_halted {
            StepResult::Halted(self.exit_code)
        } else {
            StepResult::Ok
        }
    }

    #[inline(always)]
    pub fn read_reg(&self, reg: usize) -> u32 {
        if reg == 0 {
            0
        } else {
            self.regs[reg]
        }
    }

    #[inline(always)]
    pub fn write_reg(&mut self, reg: usize, val: u32) {
        if reg != 0 {
            self.regs[reg] = val;
        }
    }

    #[inline(always)]
    pub fn read_f32(&self, reg: usize) -> f32 {
        let bits = self.fregs[reg].to_bits();
        if (bits >> 32) == 0xFFFFFFFF {
            f32::from_bits(bits as u32)
        } else {
            f32::from_bits(NAN_F32)
        }
    }

    #[inline(always)]
    pub fn write_f32(&mut self, reg: usize, val: f32) {
        let bits = if val.is_nan() {
            NAN_F64
        } else {
            0xFFFFFFFFu64 << 32 | (val.to_bits() as u64)
        };
        self.fregs[reg] = f64::from_bits(bits);
    }

    #[inline(always)]
    pub fn read_f64(&self, reg: usize) -> f64 {
        self.fregs[reg]
    }

    #[inline(always)]
    pub fn write_f64(&mut self, reg: usize, val: f64) {
        let res_val = if val.is_nan() {
            f64::from_bits(0x7FF8000000000000)
        } else {
            val
        };
        self.fregs[reg] = res_val;
    }

    pub fn run<M: MemoryOps>(&mut self, mem: &mut M) -> i32 {
        let mut inst_counter: u32 = 0;
        let mut interrupt_delay = host_imports::js_get_int_inst_delay() as u32;
        if interrupt_delay == 0 {
            interrupt_delay = 1000;
        }

        while !self.is_halted {
            // Check interrupts periodically
            if inst_counter >= interrupt_delay {
                inst_counter = 0;
                interrupt_delay = host_imports::js_get_int_inst_delay() as u32;
                if interrupt_delay == 0 {
                    interrupt_delay = 1000;
                }

                if host_imports::js_interrupt_enabled() != 0 {
                    let irq = host_imports::js_external_interrupt();
                    if irq != 0 {
                        self.handle_interrupt(irq);
                    }
                }
            }

            let pc = self.pc;
            let inst16 = mem.read_u16(pc);

            // Compressed instruction (16-bit) if lower 2 bits are not 0b11
            if (inst16 & 0x3) != 0x3 {
                if let Err(e) = self.execute_inst16(inst16, mem) {
                    host_imports::js_print_err(&format!(
                        "CPU Exec Error (Compressed) at PC {:#010x}: {}",
                        pc, e
                    ));
                    self.is_halted = true;
                    break;
                }
            } else {
                let inst32 = mem.read_u32(pc);
                if let Err(e) = self.execute_inst32(inst32, mem) {
                    host_imports::js_print_err(&format!(
                        "CPU Exec Error at PC {:#010x}: {}",
                        pc, e
                    ));
                    self.is_halted = true;
                    break;
                }
            }

            inst_counter += 1;
            self.step_counter += 1;
        }

        self.is_halted = true;
        host_imports::js_sim_stop(self.get_snapshot_js(false, self.pc));
        self.exit_code
    }

    pub fn get_snapshot_js(&self, is_breakpoint: bool, hit_address: u32) -> JsValue {
        let mstatus = *self.csrs.get(&0x300).unwrap_or(&0);
        let mcause = *self.csrs.get(&0x342).unwrap_or(&0);
        let mepc = *self.csrs.get(&0x341).unwrap_or(&0);
        let mtvec = *self.csrs.get(&0x305).unwrap_or(&0);
        let fcsr = self.fcsr;

        let snapshot = DebuggerSnapshot {
            pc: self.pc,
            gpr: self.regs.to_vec(),
            fpr: self.fregs.to_vec(),
            csrs: vec![mstatus, mcause, mepc, mtvec, fcsr],
            step_count: self.step_counter,
            is_halted: self.is_halted,
            is_breakpoint,
            hit_address,
        };
        serde_wasm_bindgen::to_value(&snapshot).unwrap()
    }

    fn handle_interrupt(&mut self, irq: i32) {
        self.csrs.insert(MEPC, self.pc);
        self.csrs.insert(MCAUSE, 0x80000000 | (irq as u32));
        if let Some(&mtvec) = self.csrs.get(&MTVEC) {
            self.pc = mtvec;
        }
    }

    #[inline(always)]
    pub fn execute_inst32<M: MemoryOps>(&mut self, raw: u32, mem: &mut M) -> Result<(), CpuError> {
        let inst = DecodedInst32::decode(raw);
        let mut next_pc = self.pc.wrapping_add(4);

        match inst.opcode {
            OP_LUI
            | OP_AUIPC
            | OP_JAL
            | OP_JALR
            | OP_BRANCH
            | OP_LOAD
            | OP_STORE
            | OP_IMM
            | OP_OP
            | OP_MISC_MEM => {
                if inst.opcode == OP_OP && inst.funct7 == 0x01 {
                    self.exec_rv32m(&inst, mem)?;
                } else {
                    self.exec_rv32i(&inst, mem, &mut next_pc)?;
                }
            }
            OP_AMO => self.exec_rv32a(&inst, mem)?,
            OP_LOAD_FP | OP_STORE_FP | OP_MADD | OP_MSUB | OP_NMSUB | OP_NMADD | OP_OP_FP => {
                self.exec_rv32fd(&inst, mem)?;
            }
            OP_SYSTEM => self.exec_csr(&inst, mem, &mut next_pc)?,
            _ => return Err(CpuError::UnknownOpcode { pc: self.pc, opcode: inst.opcode }),
        }

        self.pc = next_pc;
        Ok(())
    }

    #[inline(always)]
    fn exec_rv32i<M: MemoryOps>(
        &mut self,
        inst: &DecodedInst32,
        mem: &mut M,
        next_pc: &mut u32,
    ) -> Result<(), CpuError> {
        match inst.opcode {
            OP_LUI => {
                self.write_reg(inst.rd, inst.u_imm());
            }
            OP_AUIPC => {
                self.write_reg(inst.rd, self.pc.wrapping_add(inst.u_imm()));
            }
            OP_JAL => {
                self.write_reg(inst.rd, *next_pc);
                *next_pc = self.pc.wrapping_add(inst.j_imm() as u32);
            }
            OP_JALR => {
                let target = (self.read_reg(inst.rs1).wrapping_add(inst.i_imm() as u32)) & !1;
                self.write_reg(inst.rd, *next_pc);
                *next_pc = target;
            }
            OP_BRANCH => {
                let src1 = self.read_reg(inst.rs1);
                let src2 = self.read_reg(inst.rs2);
                let take = match inst.funct3 {
                    0 => src1 == src2,
                    1 => src1 != src2,
                    4 => (src1 as i32) < (src2 as i32),
                    5 => (src1 as i32) >= (src2 as i32),
                    6 => src1 < src2,
                    7 => src1 >= src2,
                    _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                };
                if take {
                    *next_pc = self.pc.wrapping_add(inst.b_imm() as u32);
                }
            }
            OP_LOAD => {
                let addr = self.read_reg(inst.rs1).wrapping_add(inst.i_imm() as u32);
                let val = match inst.funct3 {
                    0 => (mem.read_u8(addr) as i8 as i32) as u32,
                    1 => (mem.read_u16(addr) as i16 as i32) as u32,
                    2 => mem.read_u32(addr),
                    4 => mem.read_u8(addr) as u32,
                    5 => mem.read_u16(addr) as u32,
                    _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                };
                self.write_reg(inst.rd, val);
            }
            OP_STORE => {
                let addr = self.read_reg(inst.rs1).wrapping_add(inst.s_imm() as u32);
                let val = self.read_reg(inst.rs2);
                match inst.funct3 {
                    0 => mem.write_u8(addr, val as u8),
                    1 => mem.write_u16(addr, val as u16),
                    2 => mem.write_u32(addr, val),
                    _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                }
            }
            OP_IMM => {
                let src1 = self.read_reg(inst.rs1);
                let imm = inst.i_imm();
                let shamt = (inst.rs2 & 0x1F) as u32;
                let val = match inst.funct3 {
                    0 => src1.wrapping_add(imm as u32),
                    2 => if (src1 as i32) < imm { 1 } else { 0 },
                    3 => if src1 < (imm as u32) { 1 } else { 0 },
                    4 => src1 ^ (imm as u32),
                    6 => src1 | (imm as u32),
                    7 => src1 & (imm as u32),
                    1 => src1 << shamt,
                    5 => {
                        if inst.funct7 == 0x20 {
                            ((src1 as i32) >> shamt) as u32
                        } else {
                            src1 >> shamt
                        }
                    }
                    _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                };
                self.write_reg(inst.rd, val);
            }
            OP_OP => {
                let src1 = self.read_reg(inst.rs1);
                let src2 = self.read_reg(inst.rs2);
                let val = match (inst.funct7, inst.funct3) {
                    (0x00, 0) => src1.wrapping_add(src2),
                    (0x20, 0) => src1.wrapping_sub(src2),
                    (0x00, 1) => src1 << (src2 & 0x1F),
                    (0x00, 2) => if (src1 as i32) < (src2 as i32) { 1 } else { 0 },
                    (0x00, 3) => if src1 < src2 { 1 } else { 0 },
                    (0x00, 4) => src1 ^ src2,
                    (0x00, 5) => src1 >> (src2 & 0x1F),
                    (0x20, 5) => ((src1 as i32) >> (src2 & 0x1F)) as u32,
                    (0x00, 6) => src1 | src2,
                    (0x00, 7) => src1 & src2,
                    _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                };
                self.write_reg(inst.rd, val);
            }
            OP_MISC_MEM => {}
            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
        }
        Ok(())
    }

    #[inline(always)]
    fn exec_rv32m<M: MemoryOps>(
        &mut self,
        inst: &DecodedInst32,
        _mem: &mut M,
    ) -> Result<(), CpuError> {
        let src1 = self.read_reg(inst.rs1);
        let src2 = self.read_reg(inst.rs2);
        let val = match inst.funct3 {
            0 => src1.wrapping_mul(src2),
            1 => (((src1 as i32 as i64 * src2 as i32 as i64) >> 32) & 0xFFFFFFFF) as u32,
            2 => (((src1 as i32 as i64 * src2 as u64 as i64) >> 32) & 0xFFFFFFFF) as u32,
            3 => (((src1 as u64 * src2 as u64) >> 32) & 0xFFFFFFFF) as u32,
            4 => {
                if src2 == 0 {
                    0xFFFFFFFF
                } else {
                    ((src1 as i32).wrapping_div(src2 as i32)) as u32
                }
            }
            5 => src1.checked_div(src2).unwrap_or(0xFFFFFFFF),
            6 => {
                if src2 == 0 {
                    src1
                } else {
                    ((src1 as i32).wrapping_rem(src2 as i32)) as u32
                }
            }
            7 => {
                if src2 == 0 {
                    src1
                } else {
                    src1 % src2
                }
            }
            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
        };
        self.write_reg(inst.rd, val);
        Ok(())
    }

    #[inline(always)]
    fn exec_rv32a<M: MemoryOps>(
        &mut self,
        inst: &DecodedInst32,
        mem: &mut M,
    ) -> Result<(), CpuError> {
        let funct5 = inst.funct7 >> 2;
        let addr = self.read_reg(inst.rs1);
        let src2 = self.read_reg(inst.rs2);
        let old_val = mem.read_u32(addr);
        let mut rd_val = old_val;
        let new_val = match funct5 {
            0x02 => old_val, // LR.W
            0x03 => {
                rd_val = 0;
                src2
            } // SC.W
            0x01 => src2,                       // AMOSWAP.W
            0x00 => old_val.wrapping_add(src2), // AMOADD.W
            0x04 => old_val ^ src2,             // AMOXOR.W
            0x0C => old_val & src2,             // AMOAND.W
            0x08 => old_val | src2,             // AMOOR.W
            0x10 => (old_val as i32).min(src2 as i32) as u32, // AMOMIN.W
            0x14 => (old_val as i32).max(src2 as i32) as u32, // AMOMAX.W
            0x18 => old_val.min(src2),          // AMOMINU.W
            0x1C => old_val.max(src2),          // AMOMAXU.W
            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
        };
        if funct5 != 0x02 {
            mem.write_u32(addr, new_val);
        }
        self.write_reg(inst.rd, rd_val);
        Ok(())
    }

    #[inline(always)]
    fn exec_rv32fd<M: MemoryOps>(
        &mut self,
        inst: &DecodedInst32,
        mem: &mut M,
    ) -> Result<(), CpuError> {
        let rd = inst.rd;
        let rs1 = inst.rs1;
        let rs2 = inst.rs2;
        let funct3 = inst.funct3;

        match inst.opcode {
            OP_LOAD_FP => {
                let addr = self.read_reg(rs1).wrapping_add(inst.i_imm() as u32);
                if funct3 == 2 {
                    self.write_f32(rd, f32::from_bits(mem.read_u32(addr)));
                } else if funct3 == 3 {
                    let low = mem.read_u32(addr) as u64;
                    let high = mem.read_u32(addr + 4) as u64;
                    self.write_f64(rd, f64::from_bits((high << 32) | low));
                } else {
                    return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw });
                }
            }
            OP_STORE_FP => {
                let addr = self.read_reg(rs1).wrapping_add(inst.s_imm() as u32);
                if funct3 == 2 {
                    let bits = self.read_f32(rs2).to_bits();
                    mem.write_u32(addr, bits);
                } else if funct3 == 3 {
                    let bits = self.read_f64(rs2).to_bits();
                    mem.write_u32(addr, bits as u32);
                    mem.write_u32(addr + 4, (bits >> 32) as u32);
                } else {
                    return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw });
                }
            }
            OP_OP_FP => {
                let fmt = inst.funct7 & 0x3;
                let funct5 = inst.funct7 >> 2;
                match (fmt, funct5) {
                    (0, 0x00) => self.write_f32(rd, self.read_f32(rs1) + self.read_f32(rs2)),
                    (0, 0x01) => self.write_f32(rd, self.read_f32(rs1) - self.read_f32(rs2)),
                    (0, 0x02) => self.write_f32(rd, self.read_f32(rs1) * self.read_f32(rs2)),
                    (0, 0x03) => self.write_f32(rd, self.read_f32(rs1) / self.read_f32(rs2)),
                    (0, 0x0B) => self.write_f32(rd, self.read_f32(rs1).sqrt()),
                    (0, 0x04) => {
                        let s1 = self.read_f32(rs1);
                        let s2 = self.read_f32(rs2);
                        let b1 = s1.to_bits();
                        let b2 = s2.to_bits();
                        let res_bits = match funct3 {
                            0 => (b1 & 0x7FFFFFFF) | (b2 & 0x80000000),
                            1 => (b1 & 0x7FFFFFFF) | ((!b2) & 0x80000000),
                            2 => (b1 & 0x7FFFFFFF) | ((b1 ^ b2) & 0x80000000),
                            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                        };
                        self.fregs[rd] = f64::from_bits(0xFFFFFFFF00000000u64 | (res_bits as u64));
                    }
                    (0, 0x05) => {
                        let s1 = self.read_f32(rs1);
                        let s2 = self.read_f32(rs2);
                        let res = match funct3 {
                            0 => if s1.is_nan() { s2 } else if s2.is_nan() { s1 } else { s1.min(s2) },
                            1 => if s1.is_nan() { s2 } else if s2.is_nan() { s1 } else { s1.max(s2) },
                            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                        };
                        self.write_f32(rd, res);
                    }
                    (0, 0x18) => {
                        let s = self.read_f32(rs1);
                        let val = if rs2 == 0 { s as i32 as u32 } else { s as u32 };
                        self.write_reg(rd, val);
                    }
                    (0, 0x1A) => {
                        let val = self.read_reg(rs1);
                        let s = if rs2 == 0 { (val as i32) as f32 } else { val as f32 };
                        self.write_f32(rd, s);
                    }
                    (0, 0x1C) => {
                        if funct3 == 0 {
                            let bits = self.read_f32(rs1).to_bits();
                            self.write_reg(rd, bits);
                        } else if funct3 == 1 {
                            let s = self.read_f32(rs1);
                            let bits = s.to_bits();
                            let is_neg = (bits & 0x80000000) != 0;
                            let mask = if s.is_infinite() {
                                if is_neg { 1 << 0 } else { 1 << 7 }
                            } else if s.is_nan() {
                                if (bits & 0x00400000) != 0 { 1 << 9 } else { 1 << 8 }
                            } else if s == 0.0 {
                                if is_neg { 1 << 3 } else { 1 << 4 }
                            } else if s.is_subnormal() {
                                if is_neg { 1 << 2 } else { 1 << 5 }
                            } else if is_neg {
                                1 << 1
                            } else {
                                1 << 6
                            };
                            self.write_reg(rd, mask);
                        }
                    }
                    (0, 0x1E) => {
                        let val = self.read_reg(rs1);
                        self.write_f32(rd, f32::from_bits(val));
                    }
                    (0, 0x14) => {
                        let s1 = self.read_f32(rs1);
                        let s2 = self.read_f32(rs2);
                        let res = match funct3 {
                            0 => if s1 <= s2 { 1 } else { 0 },
                            1 => if s1 < s2 { 1 } else { 0 },
                            2 => if s1 == s2 { 1 } else { 0 },
                            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                        };
                        self.write_reg(rd, res);
                    }
                    (0, 0x08) => {
                        self.write_f32(rd, self.read_f64(rs1) as f32);
                    }

                    // Double precision (.d) fmt == 1
                    (1, 0x00) => self.write_f64(rd, self.read_f64(rs1) + self.read_f64(rs2)),
                    (1, 0x01) => self.write_f64(rd, self.read_f64(rs1) - self.read_f64(rs2)),
                    (1, 0x02) => self.write_f64(rd, self.read_f64(rs1) * self.read_f64(rs2)),
                    (1, 0x03) => self.write_f64(rd, self.read_f64(rs1) / self.read_f64(rs2)),
                    (1, 0x0B) => self.write_f64(rd, self.read_f64(rs1).sqrt()),
                    (1, 0x04) => {
                        let b1 = self.fregs[rs1].to_bits();
                        let b2 = self.fregs[rs2].to_bits();
                        let res_bits = match funct3 {
                            0 => (b1 & 0x7FFFFFFFFFFFFFFF) | (b2 & 0x8000000000000000),
                            1 => (b1 & 0x7FFFFFFFFFFFFFFF) | ((!b2) & 0x8000000000000000),
                            2 => (b1 & 0x7FFFFFFFFFFFFFFF) | ((b1 ^ b2) & 0x8000000000000000),
                            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                        };
                        self.fregs[rd] = f64::from_bits(res_bits);
                    }
                    (1, 0x05) => {
                        let d1 = self.read_f64(rs1);
                        let d2 = self.read_f64(rs2);
                        let res = match funct3 {
                            0 => if d1.is_nan() { d2 } else if d2.is_nan() { d1 } else { d1.min(d2) },
                            1 => if d1.is_nan() { d2 } else if d2.is_nan() { d1 } else { d1.max(d2) },
                            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                        };
                        self.write_f64(rd, res);
                    }
                    (1, 0x08) => {
                        self.write_f64(rd, self.read_f32(rs1) as f64);
                    }
                    (1, 0x18) => {
                        let d = self.read_f64(rs1);
                        let val = if rs2 == 0 { d as i32 as u32 } else { d as u32 };
                        self.write_reg(rd, val);
                    }
                    (1, 0x1A) => {
                        let val = self.read_reg(rs1);
                        let d = if rs2 == 0 { (val as i32) as f64 } else { val as f64 };
                        self.write_f64(rd, d);
                    }
                    (1, 0x1C) => {
                        if funct3 == 1 {
                            let d = self.read_f64(rs1);
                            let bits = d.to_bits();
                            let is_neg = (bits & 0x8000000000000000) != 0;
                            let mask = if d.is_infinite() {
                                if is_neg { 1 << 0 } else { 1 << 7 }
                            } else if d.is_nan() {
                                if (bits & 0x0008000000000000) != 0 { 1 << 9 } else { 1 << 8 }
                            } else if d == 0.0 {
                                if is_neg { 1 << 3 } else { 1 << 4 }
                            } else if d.is_subnormal() {
                                if is_neg { 1 << 2 } else { 1 << 5 }
                            } else if is_neg {
                                1 << 1
                            } else {
                                1 << 6
                            };
                            self.write_reg(rd, mask);
                        }
                    }
                    (1, 0x14) => {
                        let d1 = self.read_f64(rs1);
                        let d2 = self.read_f64(rs2);
                        let res = match funct3 {
                            0 => if d1 <= d2 { 1 } else { 0 },
                            1 => if d1 < d2 { 1 } else { 0 },
                            2 => if d1 == d2 { 1 } else { 0 },
                            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                        };
                        self.write_reg(rd, res);
                    }
                    _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
                }
            }
            OP_MADD | OP_MSUB | OP_NMSUB | OP_NMADD => {
                let rs3 = inst.rs3;
                let fmt = inst.raw.shift_then_mask(25, 3) as u8;
                if fmt == 0 {
                    let s1 = self.read_f32(rs1);
                    let s2 = self.read_f32(rs2);
                    let s3 = self.read_f32(rs3);
                    let res = match inst.opcode {
                        OP_MADD => (s1 * s2) + s3,
                        OP_MSUB => (s1 * s2) - s3,
                        OP_NMSUB => -((s1 * s2) - s3),
                        OP_NMADD => -((s1 * s2) + s3),
                        _ => unreachable!(),
                    };
                    self.write_f32(rd, res);
                } else if fmt == 1 {
                    let d1 = self.read_f64(rs1);
                    let d2 = self.read_f64(rs2);
                    let d3 = self.read_f64(rs3);
                    let res = match inst.opcode {
                        OP_MADD => (d1 * d2) + d3,
                        OP_MSUB => (d1 * d2) - d3,
                        OP_NMSUB => -((d1 * d2) - d3),
                        OP_NMADD => -((d1 * d2) + d3),
                        _ => unreachable!(),
                    };
                    self.write_f64(rd, res);
                } else {
                    return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw });
                }
            }
            _ => return Err(CpuError::IllegalInstruction { pc: self.pc, raw: inst.raw }),
        }
        Ok(())
    }

    #[inline(always)]
    fn exec_csr<M: MemoryOps>(
        &mut self,
        inst: &DecodedInst32,
        mem: &mut M,
        next_pc: &mut u32,
    ) -> Result<(), CpuError> {
        let funct3 = inst.funct3;
        let rd = inst.rd;
        let rs1 = inst.rs1;

        if funct3 == 0 {
            let imm12 = inst.raw >> 20;
            if imm12 == 0 {
                // ECALL
                handle_ecall(self, mem);
            } else if imm12 == 1 {
                // EBREAK
                self.is_halted = true;
            } else if imm12 == 0x302 {
                // MRET: set PC to mepc
                if let Some(&mepc) = self.csrs.get(&0x341) {
                    *next_pc = mepc;
                }
            }
        } else {
            // CSR instructions
            let csr_num = ((inst.raw >> 20) & 0xFFF) as u16;
            let old_val = *self.csrs.get(&csr_num).unwrap_or(&0);
            let src1_val = self.read_reg(rs1);
            let new_val = match funct3 {
                1 => src1_val,                // CSRRW
                2 => old_val | src1_val,      // CSRRS
                3 => old_val & !src1_val,     // CSRRC
                5 => rs1 as u32,              // CSRRWI
                6 => old_val | (rs1 as u32),  // CSRRSI
                7 => old_val & !(rs1 as u32), // CSRRCI
                _ => old_val,
            };
            self.csrs.insert(csr_num, new_val);
            self.write_reg(rd, old_val);
        }
        Ok(())
    }

    #[inline(always)]
    pub fn execute_inst16<M: MemoryOps>(
        &mut self,
        inst: u16,
        mem: &mut M,
    ) -> Result<(), CpuError> {
        let decoded = DecodedInst16::decode(inst);
        let mut next_pc = self.pc.wrapping_add(2);

        match (decoded.op, decoded.funct3) {
            // Quadrant 0
            (0, 0) => {
                // C.ADDI4SPN
                let rdc = ((inst.shift_then_mask(2, 0x7)) + 8) as usize;
                let imm = (inst.shift_then_mask(7, 0x30)
                    | inst.shift_then_mask(1, 0x3C0)
                    | inst.shift_then_mask(4, 0x4)
                    | inst.shift_then_mask(2, 0x8)) as u32;
                if imm != 0 {
                    self.write_reg(rdc, self.read_reg(2).wrapping_add(imm));
                }
            }
            (0, 2) => {
                // C.LW
                let rdc = ((inst.shift_then_mask(2, 0x7)) + 8) as usize;
                let rs1c = ((inst.shift_then_mask(7, 0x7)) + 8) as usize;
                let offset = (inst.shift_then_mask(6, 0x4)
                    | inst.shift_then_mask(10, 0x38)
                    | inst.shift_then_mask(3, 0x40)
                    | inst.shift_then_mask(2, 0x8)) as u32;
                let addr = self.read_reg(rs1c).wrapping_add(offset);
                self.write_reg(rdc, mem.read_u32(addr));
            }
            (0, 6) => {
                // C.SW
                let rs2c = ((inst.shift_then_mask(2, 0x7)) + 8) as usize;
                let rs1c = ((inst.shift_then_mask(7, 0x7)) + 8) as usize;
                let offset = (inst.shift_then_mask(6, 0x4)
                    | inst.shift_then_mask(10, 0x38)
                    | inst.shift_then_mask(3, 0x40)
                    | inst.shift_then_mask(2, 0x8)) as u32;
                let addr = self.read_reg(rs1c).wrapping_add(offset);
                mem.write_u32(addr, self.read_reg(rs2c));
            }

            // Quadrant 1
            (1, 0) => {
                // C.NOP / C.ADDI
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                let imm6 = (inst.shift_then_mask(12, 1) << 5) | inst.shift_then_mask(2, 0x1F);
                let sign_ext = ((imm6 as i16) << 10) >> 10;
                if rd != 0 && sign_ext != 0 {
                    self.write_reg(rd, self.read_reg(rd).wrapping_add(sign_ext as i32 as u32));
                }
            }
            (1, 1) => {
                // C.JAL
                let imm11 = (inst.shift_then_mask(12, 1) << 11)
                    | (inst.shift_then_mask(8, 1) << 10)
                    | (inst.shift_then_mask(9, 3) << 8)
                    | (inst.shift_then_mask(6, 1) << 7)
                    | (inst.shift_then_mask(7, 1) << 6)
                    | (inst.shift_then_mask(2, 1) << 5)
                    | (inst.shift_then_mask(11, 1) << 4)
                    | (inst.shift_then_mask(3, 7) << 1);
                let offset = ((imm11 as i16) << 4) >> 4;
                self.write_reg(1, next_pc);
                next_pc = self.pc.wrapping_add(offset as i32 as u32);
            }
            (1, 2) => {
                // C.LI
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                let imm6 = (inst.shift_then_mask(12, 1) << 5) | inst.shift_then_mask(2, 0x1F);
                let sign_ext = ((imm6 as i16) << 10) >> 10;
                self.write_reg(rd, sign_ext as i32 as u32);
            }
            (1, 3) => {
                // C.ADDI16SP / C.LUI
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                if rd == 2 {
                    // C.ADDI16SP
                    let imm = (inst.shift_then_mask(12, 1) << 9)
                        | (inst.shift_then_mask(3, 3) << 7)
                        | (inst.shift_then_mask(5, 1) << 6)
                        | (inst.shift_then_mask(2, 1) << 5)
                        | (inst.shift_then_mask(6, 1) << 4);
                    let offset = ((imm as i16) << 6) >> 6;
                    self.write_reg(
                        2,
                        self.read_reg(2).wrapping_add((offset as i32 as u32) << 4),
                    );
                } else if rd != 0 {
                    // C.LUI
                    let imm6 = (inst.shift_then_mask(12, 1) << 5) | inst.shift_then_mask(2, 0x1F);
                    let sign_ext = ((imm6 as i16) << 10) >> 10;
                    self.write_reg(rd, (sign_ext as i32 as u32) << 12);
                }
            }
            (1, 5) => {
                // C.J
                let imm11 = (inst.shift_then_mask(12, 1) << 11)
                    | (inst.shift_then_mask(8, 1) << 10)
                    | (inst.shift_then_mask(9, 3) << 8)
                    | (inst.shift_then_mask(6, 1) << 7)
                    | (inst.shift_then_mask(7, 1) << 6)
                    | (inst.shift_then_mask(2, 1) << 5)
                    | (inst.shift_then_mask(11, 1) << 4)
                    | (inst.shift_then_mask(3, 7) << 1);
                let offset = ((imm11 as i16) << 4) >> 4;
                next_pc = self.pc.wrapping_add(offset as i32 as u32);
            }
            (1, 6) => {
                // C.BEQZ
                let rs1c = ((inst.shift_then_mask(7, 0x7)) + 8) as usize;
                let imm = (inst.shift_then_mask(12, 1) << 8)
                    | (inst.shift_then_mask(5, 3) << 6)
                    | (inst.shift_then_mask(2, 1) << 5)
                    | (inst.shift_then_mask(10, 3) << 3)
                    | (inst.shift_then_mask(3, 3) << 1);
                let offset = ((imm as i16) << 7) >> 7;
                if self.read_reg(rs1c) == 0 {
                    next_pc = self.pc.wrapping_add(offset as i32 as u32);
                }
            }
            (1, 7) => {
                // C.BNEZ
                let rs1c = ((inst.shift_then_mask(7, 0x7)) + 8) as usize;
                let imm = (inst.shift_then_mask(12, 1) << 8)
                    | (inst.shift_then_mask(5, 3) << 6)
                    | (inst.shift_then_mask(2, 1) << 5)
                    | (inst.shift_then_mask(10, 3) << 3)
                    | (inst.shift_then_mask(3, 3) << 1);
                let offset = ((imm as i16) << 7) >> 7;
                if self.read_reg(rs1c) != 0 {
                    next_pc = self.pc.wrapping_add(offset as i32 as u32);
                }
            }

            // Quadrant 2
            (2, 0) => {
                // C.SLLI
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                let shamt = (inst.shift_then_mask(12, 1) << 5) | inst.shift_then_mask(2, 0x1F);
                if rd != 0 {
                    self.write_reg(rd, self.read_reg(rd) << shamt);
                }
            }
            (2, 2) => {
                // C.LWSP
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                let offset = (inst.shift_then_mask(2, 0x1C)
                    | inst.shift_then_mask(12, 1) << 5
                    | inst.shift_then_mask(7, 0x3) << 6) as u32;
                let addr = self.read_reg(2).wrapping_add(offset);
                if rd != 0 {
                    self.write_reg(rd, mem.read_u32(addr));
                }
            }
            (2, 4) => {
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                let rs2 = inst.shift_then_mask(2, 0x1F) as usize;
                let bit12 = inst.shift_then_mask(12, 1);
                if bit12 == 0 && rs2 == 0 {
                    // C.JR
                    if rd != 0 {
                        next_pc = self.read_reg(rd) & !1;
                    }
                } else if bit12 == 0 && rs2 != 0 {
                    // C.MV
                    if rd != 0 {
                        self.write_reg(rd, self.read_reg(rs2));
                    }
                } else if bit12 == 1 && rd != 0 && rs2 == 0 {
                    // C.JALR
                    let target = self.read_reg(rd) & !1;
                    self.write_reg(1, next_pc);
                    next_pc = target;
                } else if bit12 == 1 && rd != 0 && rs2 != 0 {
                    // C.ADD
                    self.write_reg(rd, self.read_reg(rd).wrapping_add(self.read_reg(rs2)));
                }
            }
            (2, 6) => {
                // C.SWSP
                let rs2 = inst.shift_then_mask(2, 0x1F) as usize;
                let offset = (inst.shift_then_mask(9, 0x3C) | inst.shift_then_mask(7, 0x3) << 6) as u32;
                let addr = self.read_reg(2).wrapping_add(offset);
                mem.write_u32(addr, self.read_reg(rs2));
            }

            _ => {
                return Err(CpuError::IllegalInstruction {
                    pc: self.pc,
                    raw: inst as u32,
                });
            }
        }

        self.pc = next_pc;
        Ok(())
    }
}
