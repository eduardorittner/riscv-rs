use crate::inst::*;
use crate::memory::MemoryOps;
use crate::syscall::handle_ecall;
use crate::utils::ShiftThenMask;
use crate::{host_imports, DebuggerSnapshot, SliceOutcome, SliceStatus};
use std::collections::HashMap;

const NAN_F32: u32 = 0x7FC00000;
const NAN_F64: u64 = 0xFFFFFFFF7FC00000;

/// Machine status: holds the global interrupt-enable state (MIE, bit 3) and its
/// saved copy (MPIE, bit 7).
const MSTATUS: u16 = 0x300;
/// Address of the first instruction to be executed after handling the current trap.
const MEPC: u16 = 0x341;
/// Event type that caused the current trap.
const MCAUSE: u16 = 0x342;
/// Address of the trap-handler's first instruction.
const MTVEC: u16 = 0x305;

/// `mstatus.MIE` — the global machine-mode interrupt enable.
const MSTATUS_MIE: u32 = 1 << 3;
/// `mstatus.MPIE` — the value `MIE` held before the current trap was taken.
const MSTATUS_MPIE: u32 = 1 << 7;

pub const A0: usize = 10;
pub const A1: usize = 11;
pub const A2: usize = 12;
pub const A3: usize = 13;
pub const A7: usize = 17;

/// Register number of a 3-bit compressed register field starting at `shift`.
/// The field selects one of x8..x15.
#[inline(always)]
pub(crate) fn creg(inst: u16, shift: u32) -> usize {
    (inst.shift_then_mask(shift, 0x7) + 8) as usize
}

/// Word offset of the CL/CS formats (`c.lw`, `c.sw`, `c.flw`, `c.fsw`):
/// uimm[5:3] = inst[12:10], uimm[2] = inst[6], uimm[6] = inst[5].
#[inline(always)]
pub(crate) fn cl_word_offset(inst: u16) -> u32 {
    ((inst.shift_then_mask(10, 0x7) as u32) << 3)
        | ((inst.shift_then_mask(6, 0x1) as u32) << 2)
        | ((inst.shift_then_mask(5, 0x1) as u32) << 6)
}

/// Doubleword offset of the CL/CS formats (`c.fld`, `c.fsd`):
/// uimm[5:3] = inst[12:10], uimm[7:6] = inst[6:5].
#[inline(always)]
pub(crate) fn cl_double_offset(inst: u16) -> u32 {
    ((inst.shift_then_mask(10, 0x7) as u32) << 3) | ((inst.shift_then_mask(5, 0x3) as u32) << 6)
}

/// Stack-relative word offset of the CI format (`c.lwsp`, `c.flwsp`):
/// uimm[5] = inst[12], uimm[4:2] = inst[6:4], uimm[7:6] = inst[3:2].
#[inline(always)]
pub(crate) fn ci_word_sp_offset(inst: u16) -> u32 {
    ((inst.shift_then_mask(12, 0x1) as u32) << 5)
        | ((inst.shift_then_mask(4, 0x7) as u32) << 2)
        | ((inst.shift_then_mask(2, 0x3) as u32) << 6)
}

/// Stack-relative doubleword offset of the CI format (`c.fldsp`):
/// uimm[5] = inst[12], uimm[4:3] = inst[6:5], uimm[8:6] = inst[4:2].
#[inline(always)]
pub(crate) fn ci_double_sp_offset(inst: u16) -> u32 {
    ((inst.shift_then_mask(12, 0x1) as u32) << 5)
        | ((inst.shift_then_mask(5, 0x3) as u32) << 3)
        | ((inst.shift_then_mask(2, 0x7) as u32) << 6)
}

/// Stack-relative word offset of the CSS format (`c.swsp`, `c.fswsp`):
/// uimm[5:2] = inst[12:9], uimm[7:6] = inst[8:7].
#[inline(always)]
pub(crate) fn css_word_sp_offset(inst: u16) -> u32 {
    ((inst.shift_then_mask(9, 0xF) as u32) << 2) | ((inst.shift_then_mask(7, 0x3) as u32) << 6)
}

/// Stack-relative doubleword offset of the CSS format (`c.fsdsp`):
/// uimm[5:3] = inst[12:10], uimm[8:6] = inst[9:7].
#[inline(always)]
pub(crate) fn css_double_sp_offset(inst: u16) -> u32 {
    ((inst.shift_then_mask(10, 0x7) as u32) << 3) | ((inst.shift_then_mask(7, 0x7) as u32) << 6)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuError {
    IllegalInstruction { pc: u32, raw: u32 },
    UnknownOpcode { pc: u32, opcode: u8 },
    UnalignedAccess { pc: u32, addr: u32 },
    MemoryFault { pc: u32, addr: u32 },
    UnhandledSyscall { pc: u32, number: i32 },
    UnknownSyscall(crate::syscall::UnknownSyscall),
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
                write!(
                    f,
                    "Unaligned memory access at {:#010x} (PC {:#010x})",
                    addr, pc
                )
            }
            Self::MemoryFault { pc, addr } => {
                write!(
                    f,
                    "Memory fault at address {:#010x} (PC {:#010x})",
                    addr, pc
                )
            }
            Self::UnhandledSyscall { pc, number } => {
                write!(f, "Unhandled syscall {} at PC {:#010x}", number, pc)
            }
            Self::UnknownSyscall(sys) => {
                write!(
                    f,
                    "Unknown syscall {} (a0: {:#x}, a1: {:#x}, a2: {:#x}, a3: {:#x})",
                    sys.sys_num, sys.a0, sys.a1, sys.a2, sys.a3
                )
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
    pub has_custom_syscalls: bool,
    /// Set when execution stopped on a trap rather than on a clean exit. The
    /// debugger snapshot carries it so the UI never reports a crash as success.
    pub trapped: bool,
    /// Breakpoint address already reported to the host. Execution resumes
    /// through it once, so a `continue` from a breakpoint makes progress.
    resume_past_breakpoint: Option<u32>,
}

/// Exit code reported when the guest stops on a trap. It matches the shell
/// convention for "terminated by a fault" and is simply any nonzero value that
/// a normal `exit()` is unlikely to produce.
pub const TRAP_EXIT_CODE: i32 = 134;

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
            has_custom_syscalls: false,
            trapped: false,
            resume_past_breakpoint: None,
        };
        // Default Stack Pointer sp (x2) if not set by CLI
        cpu.regs[2] = 0x7FFFFFC;
        cpu
    }

    #[inline(always)]
    pub fn step_instruction<M: MemoryOps>(&mut self, mem: &mut M) -> StepResult {
        // A breakpoint is reported once. Without the one-shot skip, a step or a
        // continue from the breakpoint address would report the same hit for
        // ever and the session could never leave it.
        if self.debug_enabled
            && self.breakpoints.contains(&self.pc)
            && self.resume_past_breakpoint != Some(self.pc)
        {
            self.resume_past_breakpoint = Some(self.pc);
            return StepResult::BreakpointHit(self.pc);
        }
        self.resume_past_breakpoint = None;

        if self.is_halted {
            return StepResult::Halted(self.exit_code);
        }

        let pc = self.pc;
        let (window, wide) = mem.fetch_window(pc);
        let inst16 = window as u16;

        let instruction_result = if (inst16 & 0x3) != 0x3 {
            self.execute_inst16(inst16, mem)
        } else {
            // `wide` is false only on a page edge or beside MMIO, where the
            // fetch could not safely widen itself.
            let inst32 = if wide { window } else { mem.read_u32(pc) };
            self.execute_inst32(inst32, mem)
        };

        self.step_counter += 1;

        if let Err(err) = instruction_result {
            if let CpuError::UnknownSyscall(ref sys_err) = err {
                host_imports::js_print_err(&format!(
                    "Unknown Syscall: {} (a0: {:#x}, a1: {:#x}, a2: {:#x}, a3: {:#x})\n",
                    sys_err.sys_num, sys_err.a0, sys_err.a1, sys_err.a2, sys_err.a3
                ));
                host_imports::notify_unknown_syscall(
                    sys_err.sys_num,
                    sys_err.a0,
                    sys_err.a1,
                    sys_err.a2,
                    sys_err.a3,
                );
            } else {
                host_imports::js_print_err(&format!(
                    "CPU Exec Error at PC {:#010x}: {}\n",
                    pc, err
                ));
            }
            // Every trap stops the machine. Without this the debugger would
            // hand back the same snapshot for ever on a faulting instruction.
            self.trapped = true;
            self.exit_code = TRAP_EXIT_CODE;
            self.is_halted = true;
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

    /// Execute at most `budget` instructions and then hand control back.
    ///
    /// The caller drives the run one slice at a time, so a host that shares its
    /// thread with a message queue (the browser worker) can answer messages
    /// between slices. The slice ends early on a halt, on a trap, or on a
    /// breakpoint hit.
    pub fn run_slice<M: MemoryOps>(&mut self, mem: &mut M, budget: u32) -> SliceOutcome {
        let budget = budget as u64;
        let mut executed: u64 = 0;
        // A slice runs to completion without yielding, so the debug settings
        // cannot change while it runs. Hoisting the test keeps the inner loop as
        // cheap as the unsliced loop it replaced.
        let watch_breakpoints = self.debug_enabled && !self.breakpoints.is_empty();
        let mut skip_breakpoint = self.resume_past_breakpoint.take();

        let status = loop {
            if self.is_halted {
                break SliceStatus::Halted;
            }
            if executed >= budget {
                break SliceStatus::Running;
            }

            // Poll for an external interrupt, then run until the next poll is
            // due. Making the poll interval the bound of the inner loop keeps
            // the per-instruction work down to one counter.
            let mut interrupt_delay = host_imports::js_get_int_inst_delay() as u64;
            if interrupt_delay == 0 {
                interrupt_delay = 1000;
            }
            if host_imports::js_interrupt_enabled() != 0 && self.interrupts_enabled() {
                let irq = host_imports::js_external_interrupt();
                if irq != 0 {
                    self.handle_interrupt(irq);
                }
            }

            let chunk = interrupt_delay.min(budget - executed);
            let mut done: u64 = 0;
            let mut stopped = None;

            while done < chunk {
                if self.is_halted {
                    stopped = Some(SliceStatus::Halted);
                    break;
                }

                if watch_breakpoints {
                    if self.breakpoints.contains(&self.pc) && skip_breakpoint != Some(self.pc) {
                        self.resume_past_breakpoint = Some(self.pc);
                        stopped = Some(SliceStatus::Breakpoint);
                        break;
                    }
                    // Only the first instruction of a slice may resume through a
                    // breakpoint that was already reported.
                    skip_breakpoint = None;
                }

                let pc = self.pc;
                // One page lookup for the whole instruction. The low halfword
                // classifies it; the high halfword is already in hand when the
                // instruction turns out to be 32 bits wide.
                let (window, wide) = mem.fetch_window(pc);
                let inst16 = window as u16;

                // Compressed instruction (16-bit) if lower 2 bits are not 0b11
                let compressed = (inst16 & 0x3) != 0x3;
                let result = if compressed {
                    self.execute_inst16(inst16, mem)
                } else {
                    let inst32 = if wide { window } else { mem.read_u32(pc) };
                    self.execute_inst32(inst32, mem)
                };

                if let Err(e) = result {
                    if let CpuError::UnknownSyscall(ref sys_err) = e {
                        host_imports::js_print_err(&format!(
                            "Unknown Syscall: {} (a0: {:#x}, a1: {:#x}, a2: {:#x}, a3: {:#x})\n",
                            sys_err.sys_num, sys_err.a0, sys_err.a1, sys_err.a2, sys_err.a3
                        ));
                        host_imports::notify_unknown_syscall(
                            sys_err.sys_num,
                            sys_err.a0,
                            sys_err.a1,
                            sys_err.a2,
                            sys_err.a3,
                        );
                    } else if compressed {
                        host_imports::js_print_err(&format!(
                            "CPU Exec Error (Compressed) at PC {:#010x}: {}\n",
                            pc, e
                        ));
                    } else {
                        host_imports::js_print_err(&format!(
                            "CPU Exec Error at PC {:#010x}: {}\n",
                            pc, e
                        ));
                    }
                    // The faulting instruction is not counted, which matches the
                    // instruction total the run loop reported before slicing.
                    self.trapped = true;
                    self.exit_code = TRAP_EXIT_CODE;
                    self.is_halted = true;
                    stopped = Some(SliceStatus::Trapped);
                    break;
                }

                self.step_counter += 1;
                done += 1;
            }

            executed += done;
            if let Some(status) = stopped {
                break status;
            }
        };

        SliceOutcome {
            status,
            steps: executed,
            pc: self.pc,
            exit_code: self.exit_code,
        }
    }

    /// Run to completion. The CLI and the batch entry points use it; the browser
    /// worker drives `run_slice` instead so it can answer messages during a run.
    pub fn run<M: MemoryOps>(&mut self, mem: &mut M) -> i32 {
        loop {
            let outcome = self.run_slice(mem, u32::MAX);
            match outcome.status {
                // A breakpoint cannot stall this loop: `run_slice` steps past an
                // already-reported breakpoint on the next call.
                SliceStatus::Running | SliceStatus::Breakpoint => continue,
                SliceStatus::Halted | SliceStatus::Trapped => break,
            }
        }

        self.is_halted = true;
        self.exit_code
    }

    /// The machine state the debugger UI reads. `DebuggerSnapshot` crosses the
    /// WASM boundary as a typed object, so a field rename here becomes a
    /// TypeScript error in the JavaScript that reads it.
    pub fn snapshot(&self, is_breakpoint: bool, hit_address: u32) -> DebuggerSnapshot {
        let mstatus = *self.csrs.get(&MSTATUS).unwrap_or(&0);
        let mcause = *self.csrs.get(&MCAUSE).unwrap_or(&0);
        let mepc = *self.csrs.get(&MEPC).unwrap_or(&0);
        let mtvec = *self.csrs.get(&MTVEC).unwrap_or(&0);
        let fcsr = self.fcsr;

        DebuggerSnapshot {
            pc: self.pc,
            gpr: self.regs.to_vec(),
            fpr: self.fregs.to_vec(),
            csrs: vec![mstatus, mcause, mepc, mtvec, fcsr],
            step_count: self.step_counter,
            is_halted: self.is_halted,
            is_breakpoint,
            hit_address,
            exit_code: self.exit_code,
            trapped: self.trapped,
        }
    }

    /// True when `mstatus.MIE` allows machine-mode interrupts to be delivered.
    #[inline(always)]
    pub fn interrupts_enabled(&self) -> bool {
        (*self.csrs.get(&MSTATUS).unwrap_or(&0) & MSTATUS_MIE) != 0
    }

    /// Take an external machine interrupt: save the interrupted PC and the
    /// interrupt-enable state, then vector to `mtvec`.
    pub fn handle_interrupt(&mut self, irq: i32) {
        self.csrs.insert(MEPC, self.pc);
        self.csrs.insert(MCAUSE, 0x80000000 | (irq as u32));
        // Save MIE into MPIE and disable interrupts for the duration of the
        // handler, so a second interrupt cannot overwrite mepc.
        let mstatus = *self.csrs.get(&MSTATUS).unwrap_or(&0);
        let mut new_mstatus = mstatus & !(MSTATUS_MIE | MSTATUS_MPIE);
        if (mstatus & MSTATUS_MIE) != 0 {
            new_mstatus |= MSTATUS_MPIE;
        }
        self.csrs.insert(MSTATUS, new_mstatus);
        if let Some(&mtvec) = self.csrs.get(&MTVEC) {
            self.pc = mtvec;
        }
    }

    #[inline(always)]
    pub fn execute_inst32<M: MemoryOps>(&mut self, raw: u32, mem: &mut M) -> Result<(), CpuError> {
        let inst = DecodedInst32::decode(raw);
        let mut next_pc = self.pc.wrapping_add(4);

        match inst.opcode {
            // The M extension shares the OP major opcode with the base integer
            // register-register instructions; `funct7 == 0x01` is what tells
            // them apart. Giving OP_OP its own arm keeps that test off the path
            // of every other opcode, which is where it used to sit.
            OP_OP => {
                if inst.funct7 == 0x01 {
                    self.exec_rv32m(&inst, mem)?;
                } else {
                    self.exec_rv32i(&inst, mem, &mut next_pc)?;
                }
            }
            OP_LUI | OP_AUIPC | OP_JAL | OP_JALR | OP_BRANCH | OP_LOAD | OP_STORE | OP_IMM
            | OP_MISC_MEM => self.exec_rv32i(&inst, mem, &mut next_pc)?,
            OP_AMO => self.exec_rv32a(&inst, mem)?,
            OP_LOAD_FP | OP_STORE_FP | OP_MADD | OP_MSUB | OP_NMSUB | OP_NMADD | OP_OP_FP => {
                self.exec_rv32fd(&inst, mem)?;
            }
            OP_SYSTEM => self.exec_csr(&inst, mem, &mut next_pc)?,
            _ => {
                return Err(CpuError::UnknownOpcode {
                    pc: self.pc,
                    opcode: inst.opcode,
                })
            }
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
                    _ => {
                        return Err(CpuError::IllegalInstruction {
                            pc: self.pc,
                            raw: inst.raw,
                        })
                    }
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
                    _ => {
                        return Err(CpuError::IllegalInstruction {
                            pc: self.pc,
                            raw: inst.raw,
                        })
                    }
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
                    _ => {
                        return Err(CpuError::IllegalInstruction {
                            pc: self.pc,
                            raw: inst.raw,
                        })
                    }
                }
            }
            OP_IMM => {
                let src1 = self.read_reg(inst.rs1);
                let imm = inst.i_imm();
                let shamt = (inst.rs2 & 0x1F) as u32;
                let val = match inst.funct3 {
                    0 => src1.wrapping_add(imm as u32),
                    2 => {
                        if (src1 as i32) < imm {
                            1
                        } else {
                            0
                        }
                    }
                    3 => {
                        if src1 < (imm as u32) {
                            1
                        } else {
                            0
                        }
                    }
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
                    _ => {
                        return Err(CpuError::IllegalInstruction {
                            pc: self.pc,
                            raw: inst.raw,
                        })
                    }
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
                    (0x00, 2) => {
                        if (src1 as i32) < (src2 as i32) {
                            1
                        } else {
                            0
                        }
                    }
                    (0x00, 3) => {
                        if src1 < src2 {
                            1
                        } else {
                            0
                        }
                    }
                    (0x00, 4) => src1 ^ src2,
                    (0x00, 5) => src1 >> (src2 & 0x1F),
                    (0x20, 5) => ((src1 as i32) >> (src2 & 0x1F)) as u32,
                    (0x00, 6) => src1 | src2,
                    (0x00, 7) => src1 & src2,
                    _ => {
                        return Err(CpuError::IllegalInstruction {
                            pc: self.pc,
                            raw: inst.raw,
                        })
                    }
                };
                self.write_reg(inst.rd, val);
            }
            OP_MISC_MEM => {}
            _ => {
                return Err(CpuError::IllegalInstruction {
                    pc: self.pc,
                    raw: inst.raw,
                })
            }
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
            _ => {
                return Err(CpuError::IllegalInstruction {
                    pc: self.pc,
                    raw: inst.raw,
                })
            }
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
            0x01 => src2,    // AMOSWAP.W
            0x00 => old_val.wrapping_add(src2), // AMOADD.W
            0x04 => old_val ^ src2, // AMOXOR.W
            0x0C => old_val & src2, // AMOAND.W
            0x08 => old_val | src2, // AMOOR.W
            0x10 => (old_val as i32).min(src2 as i32) as u32, // AMOMIN.W
            0x14 => (old_val as i32).max(src2 as i32) as u32, // AMOMAX.W
            0x18 => old_val.min(src2), // AMOMINU.W
            0x1C => old_val.max(src2), // AMOMAXU.W
            _ => {
                return Err(CpuError::IllegalInstruction {
                    pc: self.pc,
                    raw: inst.raw,
                })
            }
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
                    let raw = mem.read_u32(addr);
                    self.fregs[rd] = f64::from_bits(0xFFFFFFFF00000000u64 | (raw as u64));
                } else if funct3 == 3 {
                    let low = mem.read_u32(addr) as u64;
                    let high = mem.read_u32(addr + 4) as u64;
                    self.write_f64(rd, f64::from_bits((high << 32) | low));
                } else {
                    return Err(CpuError::IllegalInstruction {
                        pc: self.pc,
                        raw: inst.raw,
                    });
                }
            }
            OP_STORE_FP => {
                let addr = self.read_reg(rs1).wrapping_add(inst.s_imm() as u32);
                if funct3 == 2 {
                    let bits = (self.fregs[rs2].to_bits() & 0xFFFFFFFF) as u32;
                    mem.write_u32(addr, bits);
                } else if funct3 == 3 {
                    let bits = self.read_f64(rs2).to_bits();
                    mem.write_u32(addr, bits as u32);
                    mem.write_u32(addr + 4, (bits >> 32) as u32);
                } else {
                    return Err(CpuError::IllegalInstruction {
                        pc: self.pc,
                        raw: inst.raw,
                    });
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
                            _ => {
                                return Err(CpuError::IllegalInstruction {
                                    pc: self.pc,
                                    raw: inst.raw,
                                })
                            }
                        };
                        self.fregs[rd] = f64::from_bits(0xFFFFFFFF00000000u64 | (res_bits as u64));
                    }
                    (0, 0x05) => {
                        let s1 = self.read_f32(rs1);
                        let s2 = self.read_f32(rs2);
                        let res = match funct3 {
                            0 => {
                                if s1.is_nan() {
                                    s2
                                } else if s2.is_nan() {
                                    s1
                                } else {
                                    s1.min(s2)
                                }
                            }
                            1 => {
                                if s1.is_nan() {
                                    s2
                                } else if s2.is_nan() {
                                    s1
                                } else {
                                    s1.max(s2)
                                }
                            }
                            _ => {
                                return Err(CpuError::IllegalInstruction {
                                    pc: self.pc,
                                    raw: inst.raw,
                                })
                            }
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
                        let s = if rs2 == 0 {
                            (val as i32) as f32
                        } else {
                            val as f32
                        };
                        self.write_f32(rd, s);
                    }
                    (0, 0x1C) => {
                        if funct3 == 0 {
                            let bits = (self.fregs[rs1].to_bits() & 0xFFFFFFFF) as u32;
                            self.write_reg(rd, bits);
                        } else if funct3 == 1 {
                            let s = self.read_f32(rs1);
                            let bits = s.to_bits();
                            let is_neg = (bits & 0x80000000) != 0;
                            let mask = if s.is_infinite() {
                                if is_neg {
                                    1 << 0
                                } else {
                                    1 << 7
                                }
                            } else if s.is_nan() {
                                if (bits & 0x00400000) != 0 {
                                    1 << 9
                                } else {
                                    1 << 8
                                }
                            } else if s == 0.0 {
                                if is_neg {
                                    1 << 3
                                } else {
                                    1 << 4
                                }
                            } else if s.is_subnormal() {
                                if is_neg {
                                    1 << 2
                                } else {
                                    1 << 5
                                }
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
                        self.fregs[rd] = f64::from_bits(0xFFFFFFFF00000000u64 | (val as u64));
                    }
                    (0, 0x14) => {
                        let s1 = self.read_f32(rs1);
                        let s2 = self.read_f32(rs2);
                        let res = match funct3 {
                            0 => {
                                if s1 <= s2 {
                                    1
                                } else {
                                    0
                                }
                            }
                            1 => {
                                if s1 < s2 {
                                    1
                                } else {
                                    0
                                }
                            }
                            2 => {
                                if s1 == s2 {
                                    1
                                } else {
                                    0
                                }
                            }
                            _ => {
                                return Err(CpuError::IllegalInstruction {
                                    pc: self.pc,
                                    raw: inst.raw,
                                })
                            }
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
                            _ => {
                                return Err(CpuError::IllegalInstruction {
                                    pc: self.pc,
                                    raw: inst.raw,
                                })
                            }
                        };
                        self.fregs[rd] = f64::from_bits(res_bits);
                    }
                    (1, 0x05) => {
                        let d1 = self.read_f64(rs1);
                        let d2 = self.read_f64(rs2);
                        let res = match funct3 {
                            0 => {
                                if d1.is_nan() {
                                    d2
                                } else if d2.is_nan() {
                                    d1
                                } else {
                                    d1.min(d2)
                                }
                            }
                            1 => {
                                if d1.is_nan() {
                                    d2
                                } else if d2.is_nan() {
                                    d1
                                } else {
                                    d1.max(d2)
                                }
                            }
                            _ => {
                                return Err(CpuError::IllegalInstruction {
                                    pc: self.pc,
                                    raw: inst.raw,
                                })
                            }
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
                        let d = if rs2 == 0 {
                            (val as i32) as f64
                        } else {
                            val as f64
                        };
                        self.write_f64(rd, d);
                    }
                    (1, 0x1C) => {
                        if funct3 == 1 {
                            let d = self.read_f64(rs1);
                            let bits = d.to_bits();
                            let is_neg = (bits & 0x8000000000000000) != 0;
                            let mask = if d.is_infinite() {
                                if is_neg {
                                    1 << 0
                                } else {
                                    1 << 7
                                }
                            } else if d.is_nan() {
                                if (bits & 0x0008000000000000) != 0 {
                                    1 << 9
                                } else {
                                    1 << 8
                                }
                            } else if d == 0.0 {
                                if is_neg {
                                    1 << 3
                                } else {
                                    1 << 4
                                }
                            } else if d.is_subnormal() {
                                if is_neg {
                                    1 << 2
                                } else {
                                    1 << 5
                                }
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
                            0 => {
                                if d1 <= d2 {
                                    1
                                } else {
                                    0
                                }
                            }
                            1 => {
                                if d1 < d2 {
                                    1
                                } else {
                                    0
                                }
                            }
                            2 => {
                                if d1 == d2 {
                                    1
                                } else {
                                    0
                                }
                            }
                            _ => {
                                return Err(CpuError::IllegalInstruction {
                                    pc: self.pc,
                                    raw: inst.raw,
                                })
                            }
                        };
                        self.write_reg(rd, res);
                    }
                    _ => {
                        return Err(CpuError::IllegalInstruction {
                            pc: self.pc,
                            raw: inst.raw,
                        })
                    }
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
                        _ => {
                            return Err(CpuError::IllegalInstruction {
                                pc: self.pc,
                                raw: inst.raw,
                            })
                        }
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
                        _ => {
                            return Err(CpuError::IllegalInstruction {
                                pc: self.pc,
                                raw: inst.raw,
                            })
                        }
                    };
                    self.write_f64(rd, res);
                } else {
                    return Err(CpuError::IllegalInstruction {
                        pc: self.pc,
                        raw: inst.raw,
                    });
                }
            }
            _ => {
                return Err(CpuError::IllegalInstruction {
                    pc: self.pc,
                    raw: inst.raw,
                })
            }
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
                handle_ecall(self, mem).map_err(CpuError::UnknownSyscall)?;
            } else if imm12 == 1 {
                // EBREAK
                self.is_halted = true;
            } else if imm12 == 0x302 {
                // MRET: return to mepc and restore MIE from MPIE, which is then
                // set back to 1 as the privileged specification requires.
                if let Some(&mepc) = self.csrs.get(&MEPC) {
                    *next_pc = mepc;
                }
                let mstatus = *self.csrs.get(&MSTATUS).unwrap_or(&0);
                let mut new_mstatus = (mstatus & !MSTATUS_MIE) | MSTATUS_MPIE;
                if (mstatus & MSTATUS_MPIE) != 0 {
                    new_mstatus |= MSTATUS_MIE;
                }
                self.csrs.insert(MSTATUS, new_mstatus);
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
    pub fn execute_inst16<M: MemoryOps>(&mut self, inst: u16, mem: &mut M) -> Result<(), CpuError> {
        let decoded = DecodedInst16::decode(inst);
        let mut next_pc = self.pc.wrapping_add(2);

        match (decoded.op, decoded.funct3) {
            // Quadrant 0
            (0, 0) => {
                // C.ADDI4SPN. The all-zero immediate is a reserved encoding, and
                // the all-zero halfword is the canonical illegal instruction.
                let rdc = creg(inst, 2);
                let imm = (inst.shift_then_mask(7, 0x30)
                    | inst.shift_then_mask(1, 0x3C0)
                    | inst.shift_then_mask(4, 0x4)
                    | inst.shift_then_mask(2, 0x8)) as u32;
                if imm == 0 {
                    return Err(CpuError::IllegalInstruction {
                        pc: self.pc,
                        raw: inst as u32,
                    });
                }
                self.write_reg(rdc, self.read_reg(2).wrapping_add(imm));
            }
            (0, 1) => {
                // C.FLD
                let rdc = creg(inst, 2);
                let rs1c = creg(inst, 7);
                let addr = self.read_reg(rs1c).wrapping_add(cl_double_offset(inst));
                let low = mem.read_u32(addr) as u64;
                let high = mem.read_u32(addr.wrapping_add(4)) as u64;
                self.write_f64(rdc, f64::from_bits((high << 32) | low));
            }
            (0, 2) => {
                // C.LW
                let rdc = creg(inst, 2);
                let rs1c = creg(inst, 7);
                let addr = self.read_reg(rs1c).wrapping_add(cl_word_offset(inst));
                self.write_reg(rdc, mem.read_u32(addr));
            }
            (0, 3) => {
                // C.FLW
                let rdc = creg(inst, 2);
                let rs1c = creg(inst, 7);
                let addr = self.read_reg(rs1c).wrapping_add(cl_word_offset(inst));
                let raw = mem.read_u32(addr);
                self.fregs[rdc] = f64::from_bits(0xFFFFFFFF00000000u64 | (raw as u64));
            }
            (0, 5) => {
                // C.FSD
                let rs2c = creg(inst, 2);
                let rs1c = creg(inst, 7);
                let addr = self.read_reg(rs1c).wrapping_add(cl_double_offset(inst));
                let bits = self.read_f64(rs2c).to_bits();
                mem.write_u32(addr, bits as u32);
                mem.write_u32(addr.wrapping_add(4), (bits >> 32) as u32);
            }
            (0, 6) => {
                // C.SW
                let rs2c = creg(inst, 2);
                let rs1c = creg(inst, 7);
                let addr = self.read_reg(rs1c).wrapping_add(cl_word_offset(inst));
                mem.write_u32(addr, self.read_reg(rs2c));
            }
            (0, 7) => {
                // C.FSW
                let rs2c = creg(inst, 2);
                let rs1c = creg(inst, 7);
                let addr = self.read_reg(rs1c).wrapping_add(cl_word_offset(inst));
                let bits = (self.fregs[rs2c].to_bits() & 0xFFFFFFFF) as u32;
                mem.write_u32(addr, bits);
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
                    // C.ADDI16SP. The immediate already carries its scale: the
                    // encoded bits are nzimm[9:4], so it is only sign extended
                    // from bit 9 and never shifted again.
                    let imm = (inst.shift_then_mask(12, 1) << 9)
                        | (inst.shift_then_mask(3, 3) << 7)
                        | (inst.shift_then_mask(5, 1) << 6)
                        | (inst.shift_then_mask(2, 1) << 5)
                        | (inst.shift_then_mask(6, 1) << 4);
                    if imm == 0 {
                        return Err(CpuError::IllegalInstruction {
                            pc: self.pc,
                            raw: inst as u32,
                        });
                    }
                    let offset = ((imm as i16) << 6) >> 6;
                    self.write_reg(2, self.read_reg(2).wrapping_add(offset as i32 as u32));
                } else if rd != 0 {
                    // C.LUI. A zero immediate is a reserved encoding.
                    let imm6 = (inst.shift_then_mask(12, 1) << 5) | inst.shift_then_mask(2, 0x1F);
                    if imm6 == 0 {
                        return Err(CpuError::IllegalInstruction {
                            pc: self.pc,
                            raw: inst as u32,
                        });
                    }
                    let sign_ext = ((imm6 as i16) << 10) >> 10;
                    self.write_reg(rd, (sign_ext as i32 as u32) << 12);
                } else {
                    // rd = x0 is reserved in this group.
                    return Err(CpuError::IllegalInstruction {
                        pc: self.pc,
                        raw: inst as u32,
                    });
                }
            }
            (1, 4) => {
                // C.SRLI / C.SRAI / C.ANDI / C.SUB / C.XOR / C.OR / C.AND
                let rdc = creg(inst, 7);
                let bit12 = inst.shift_then_mask(12, 1);
                match inst.shift_then_mask(10, 0x3) {
                    0 | 1 => {
                        // C.SRLI / C.SRAI. RV32C requires shamt[5] to be zero;
                        // the shamt[5] = 1 code points are reserved.
                        if bit12 != 0 {
                            return Err(CpuError::IllegalInstruction {
                                pc: self.pc,
                                raw: inst as u32,
                            });
                        }
                        // shamt = 0 encodes the RV128 c.srli64/c.srai64 hints,
                        // which act as a no-op here.
                        let shamt = inst.shift_then_mask(2, 0x1F) as u32;
                        if shamt != 0 {
                            let src = self.read_reg(rdc);
                            let val = if inst.shift_then_mask(10, 0x3) == 0 {
                                src >> shamt
                            } else {
                                ((src as i32) >> shamt) as u32
                            };
                            self.write_reg(rdc, val);
                        }
                    }
                    2 => {
                        // C.ANDI
                        let imm6 =
                            (inst.shift_then_mask(12, 1) << 5) | inst.shift_then_mask(2, 0x1F);
                        let sign_ext = ((imm6 as i16) << 10) >> 10;
                        self.write_reg(rdc, self.read_reg(rdc) & (sign_ext as i32 as u32));
                    }
                    _ => {
                        // bit 12 selects the RV64-only C.SUBW / C.ADDW group,
                        // whose four code points are all reserved on RV32.
                        if bit12 != 0 {
                            return Err(CpuError::IllegalInstruction {
                                pc: self.pc,
                                raw: inst as u32,
                            });
                        }
                        let rs2c = creg(inst, 2);
                        let src1 = self.read_reg(rdc);
                        let src2 = self.read_reg(rs2c);
                        let val = match inst.shift_then_mask(5, 0x3) {
                            0 => src1.wrapping_sub(src2), // C.SUB
                            1 => src1 ^ src2,             // C.XOR
                            2 => src1 | src2,             // C.OR
                            _ => src1 & src2,             // C.AND
                        };
                        self.write_reg(rdc, val);
                    }
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
                // C.SLLI. RV32C requires shamt[5] to be zero.
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                if inst.shift_then_mask(12, 1) != 0 {
                    return Err(CpuError::IllegalInstruction {
                        pc: self.pc,
                        raw: inst as u32,
                    });
                }
                let shamt = inst.shift_then_mask(2, 0x1F) as u32;
                if rd != 0 && shamt != 0 {
                    self.write_reg(rd, self.read_reg(rd) << shamt);
                }
            }
            (2, 1) => {
                // C.FLDSP
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                let addr = self.read_reg(2).wrapping_add(ci_double_sp_offset(inst));
                let low = mem.read_u32(addr) as u64;
                let high = mem.read_u32(addr.wrapping_add(4)) as u64;
                self.write_f64(rd, f64::from_bits((high << 32) | low));
            }
            (2, 2) => {
                // C.LWSP. rd = x0 is a reserved encoding.
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                if rd == 0 {
                    return Err(CpuError::IllegalInstruction {
                        pc: self.pc,
                        raw: inst as u32,
                    });
                }
                let addr = self.read_reg(2).wrapping_add(ci_word_sp_offset(inst));
                self.write_reg(rd, mem.read_u32(addr));
            }
            (2, 3) => {
                // C.FLWSP
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                let addr = self.read_reg(2).wrapping_add(ci_word_sp_offset(inst));
                let raw = mem.read_u32(addr);
                self.fregs[rd] = f64::from_bits(0xFFFFFFFF00000000u64 | (raw as u64));
            }
            (2, 4) => {
                let rd = inst.shift_then_mask(7, 0x1F) as usize;
                let rs2 = inst.shift_then_mask(2, 0x1F) as usize;
                let bit12 = inst.shift_then_mask(12, 1);
                if bit12 == 0 && rs2 == 0 {
                    // C.JR. rs1 = x0 is a reserved encoding.
                    if rd == 0 {
                        return Err(CpuError::IllegalInstruction {
                            pc: self.pc,
                            raw: inst as u32,
                        });
                    }
                    next_pc = self.read_reg(rd) & !1;
                } else if bit12 == 0 && rs2 != 0 {
                    // C.MV (rd = x0 is a hint)
                    if rd != 0 {
                        self.write_reg(rd, self.read_reg(rs2));
                    }
                } else if rd == 0 && rs2 == 0 {
                    // C.EBREAK
                    self.is_halted = true;
                } else if rs2 == 0 {
                    // C.JALR
                    let target = self.read_reg(rd) & !1;
                    self.write_reg(1, next_pc);
                    next_pc = target;
                } else {
                    // C.ADD (rd = x0 is a hint)
                    if rd != 0 {
                        self.write_reg(rd, self.read_reg(rd).wrapping_add(self.read_reg(rs2)));
                    }
                }
            }
            (2, 5) => {
                // C.FSDSP
                let rs2 = inst.shift_then_mask(2, 0x1F) as usize;
                let addr = self.read_reg(2).wrapping_add(css_double_sp_offset(inst));
                let bits = self.read_f64(rs2).to_bits();
                mem.write_u32(addr, bits as u32);
                mem.write_u32(addr.wrapping_add(4), (bits >> 32) as u32);
            }
            (2, 6) => {
                // C.SWSP
                let rs2 = inst.shift_then_mask(2, 0x1F) as usize;
                let addr = self.read_reg(2).wrapping_add(css_word_sp_offset(inst));
                mem.write_u32(addr, self.read_reg(rs2));
            }
            (2, 7) => {
                // C.FSWSP
                let rs2 = inst.shift_then_mask(2, 0x1F) as usize;
                let addr = self.read_reg(2).wrapping_add(css_word_sp_offset(inst));
                let bits = (self.fregs[rs2].to_bits() & 0xFFFFFFFF) as u32;
                mem.write_u32(addr, bits);
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
