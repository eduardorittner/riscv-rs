use crate::host_imports;
use crate::memory::MemoryOps;
use crate::syscall::handle_ecall;
use std::collections::HashMap;

pub struct Cpu {
    pub regs: [u32; 32],
    pub fregs: [f64; 32],
    pub pc: u32,
    #[allow(dead_code)]
    pub fcsr: u32,
    pub csrs: HashMap<u16, u32>,
    pub is_halted: bool,
    pub exit_code: i32,
    #[allow(dead_code)]
    pub isa_imac: bool,
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
            isa_imac: true,
        };
        // Default Stack Pointer sp (x2) if not set by CLI
        cpu.regs[2] = 0x7FFFFFC;
        cpu
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
            f32::from_bits(0x7FC00000) // Canonical NaN
        }
    }

    #[inline(always)]
    pub fn write_f32(&mut self, reg: usize, val: f32) {
        let bits = if val.is_nan() {
            0xFFFFFFFF7FC00000u64 | ((val.to_bits() as u64) & 0x80000000)
        } else {
            0xFFFFFFFF00000000u64 | (val.to_bits() as u64)
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
            f64::from_bits(0x7FF8000000000000 | (val.to_bits() & 0x8000000000000000))
        } else {
            val
        };
        self.fregs[reg] = res_val;
    }

    pub fn run<M: MemoryOps>(&mut self, mem: &mut M) -> i32 {
        let mut inst_counter: u32 = 0;
        let mut int_delay = host_imports::js_get_int_inst_delay() as u32;
        if int_delay == 0 {
            int_delay = 1000;
        }

        while !self.is_halted {
            // Check interrupts periodically
            if inst_counter >= int_delay {
                inst_counter = 0;
                int_delay = host_imports::js_get_int_inst_delay() as u32;
                if int_delay == 0 {
                    int_delay = 1000;
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
                if let Err(e) = self.execute_c_inst(inst16, mem) {
                    host_imports::js_print_err(&format!("CPU Exec Error (Compressed) at PC {:#010x}: {}", pc, e));
                    break;
                }
            } else {
                let inst32 = mem.read_u32(pc);
                if let Err(e) = self.execute_inst(inst32, mem) {
                    host_imports::js_print_err(&format!("CPU Exec Error at PC {:#010x}: {}", pc, e));
                    break;
                }
            }

            inst_counter += 1;
        }

        host_imports::js_sim_stop();
        self.exit_code
    }

    fn handle_interrupt(&mut self, irq: i32) {
        // Save PC to mepc and set mcause
        self.csrs.insert(0x341, self.pc); // mepc
        self.csrs.insert(0x342, 0x80000000 | (irq as u32)); // mcause
        if let Some(&mtvec) = self.csrs.get(&0x305) {
            self.pc = mtvec;
        }
    }

    pub fn execute_inst<M: MemoryOps>(&mut self, inst: u32, mem: &mut M) -> Result<(), String> {
        let opcode = inst & 0x7F;
        let rd = ((inst >> 7) & 0x1F) as usize;
        let funct3 = (inst >> 12) & 0x7;
        let rs1 = ((inst >> 15) & 0x1F) as usize;
        let rs2 = ((inst >> 20) & 0x1F) as usize;
        let funct7 = (inst >> 25) & 0x7F;

        let mut next_pc = self.pc.wrapping_add(4);

        match opcode {
            // LUI
            0x37 => {
                let imm = inst & 0xFFFFF000;
                self.write_reg(rd, imm);
            }
            // AUIPC
            0x17 => {
                let imm = inst & 0xFFFFF000;
                self.write_reg(rd, self.pc.wrapping_add(imm));
            }
            // JAL
            0x6F => {
                let imm20 = (inst >> 31) & 1;
                let imm10_1 = (inst >> 21) & 0x3FF;
                let imm11 = (inst >> 20) & 1;
                let imm19_12 = (inst >> 12) & 0xFF;
                let offset = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                let sign_ext = ((offset as i32) << 11) >> 11;
                self.write_reg(rd, next_pc);
                next_pc = self.pc.wrapping_add(sign_ext as u32);
            }
            // JALR
            0x67 => {
                let imm = (inst as i32) >> 20;
                let target = (self.read_reg(rs1).wrapping_add(imm as u32)) & !1;
                self.write_reg(rd, next_pc);
                next_pc = target;
            }
            // Branch B-type
            0x63 => {
                let imm12 = (inst >> 31) & 1;
                let imm10_5 = (inst >> 25) & 0x3F;
                let imm4_1 = (inst >> 8) & 0xF;
                let imm11 = (inst >> 7) & 1;
                let offset = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
                let sign_ext = ((offset as i32) << 19) >> 19;

                let src1 = self.read_reg(rs1);
                let src2 = self.read_reg(rs2);
                let take_branch = match funct3 {
                    0 => src1 == src2,                  // BEQ
                    1 => src1 != src2,                  // BNE
                    4 => (src1 as i32) < (src2 as i32), // BLT
                    5 => (src1 as i32) >= (src2 as i32),// BGE
                    6 => src1 < src2,                   // BLTU
                    7 => src1 >= src2,                  // BGEU
                    _ => return Err(format!("Unknown branch funct3: {}", funct3)),
                };
                if take_branch {
                    next_pc = self.pc.wrapping_add(sign_ext as u32);
                }
            }
            // Load I-type
            0x03 => {
                let imm = (inst as i32) >> 20;
                let addr = self.read_reg(rs1).wrapping_add(imm as u32);
                let val = match funct3 {
                    0 => (mem.read_u8(addr) as i8) as i32 as u32,  // LB
                    1 => (mem.read_u16(addr) as i16) as i32 as u32,// LH
                    2 => mem.read_u32(addr),                       // LW
                    4 => mem.read_u8(addr) as u32,                 // LBU
                    5 => mem.read_u16(addr) as u32,                // LHU
                    _ => return Err(format!("Unknown load funct3: {}", funct3)),
                };
                self.write_reg(rd, val);
            }
            // Store S-type
            0x23 => {
                let imm11_5 = (inst >> 25) & 0x7F;
                let imm4_0 = (inst >> 7) & 0x1F;
                let offset = (imm11_5 << 5) | imm4_0;
                let sign_ext = ((offset as i32) << 20) >> 20;
                let addr = self.read_reg(rs1).wrapping_add(sign_ext as u32);
                let val = self.read_reg(rs2);
                match funct3 {
                    0 => mem.write_u8(addr, val as u8),
                    1 => mem.write_u16(addr, val as u16),
                    2 => mem.write_u32(addr, val),
                    _ => return Err(format!("Unknown store funct3: {}", funct3)),
                }
            }
            // OP-IMM (I-type)
            0x13 => {
                let imm = (inst as i32) >> 20;
                let src1 = self.read_reg(rs1);
                let shamt = (inst >> 20) & 0x1F;
                let val = match funct3 {
                    0 => src1.wrapping_add(imm as u32),                     // ADDI
                    2 => if (src1 as i32) < imm { 1 } else { 0 },           // SLTI
                    3 => if src1 < (imm as u32) { 1 } else { 0 },           // SLTIU
                    4 => src1 ^ (imm as u32),                               // XORI
                    6 => src1 | (imm as u32),                               // ORI
                    7 => src1 & (imm as u32),                               // ANDI
                    1 => src1 << shamt,                                     // SLLI
                    5 => if (inst >> 30) & 1 == 1 {                         // SRAI / SRLI
                        ((src1 as i32) >> shamt) as u32
                    } else {
                        src1 >> shamt
                    },
                    _ => return Err(format!("Unknown OP-IMM funct3: {}", funct3)),
                };
                self.write_reg(rd, val);
            }
            // OP (R-type)
            0x33 => {
                let src1 = self.read_reg(rs1);
                let src2 = self.read_reg(rs2);
                let val = match (funct7, funct3) {
                    (0x00, 0) => src1.wrapping_add(src2),                       // ADD
                    (0x20, 0) => src1.wrapping_sub(src2),                       // SUB
                    (0x00, 1) => src1 << (src2 & 0x1F),                         // SLL
                    (0x00, 2) => if (src1 as i32) < (src2 as i32) { 1 } else { 0 }, // SLT
                    (0x00, 3) => if src1 < src2 { 1 } else { 0 },               // SLTU
                    (0x00, 4) => src1 ^ src2,                                   // XOR
                    (0x00, 5) => src1 >> (src2 & 0x1F),                         // SRL
                    (0x20, 5) => ((src1 as i32) >> (src2 & 0x1F)) as u32,       // SRA
                    (0x00, 6) => src1 | src2,                                   // OR
                    (0x00, 7) => src1 & src2,                                   // AND

                    // M Extension
                    (0x01, 0) => src1.wrapping_mul(src2),                       // MUL
                    (0x01, 1) => (((src1 as i32 as i64 * src2 as i32 as i64) >> 32) & 0xFFFFFFFF) as u32, // MULH
                    (0x01, 2) => (((src1 as i32 as i64 * src2 as u64 as i64) >> 32) & 0xFFFFFFFF) as u32, // MULHSU
                    (0x01, 3) => (((src1 as u64 * src2 as u64) >> 32) & 0xFFFFFFFF) as u32, // MULHU
                    (0x01, 4) => if src2 == 0 { 0xFFFFFFFF } else { ((src1 as i32).wrapping_div(src2 as i32)) as u32 }, // DIV
                    (0x01, 5) => if src2 == 0 { 0xFFFFFFFF } else { src1 / src2 }, // DIVU
                    (0x01, 6) => if src2 == 0 { src1 } else { ((src1 as i32).wrapping_rem(src2 as i32)) as u32 }, // REM
                    (0x01, 7) => if src2 == 0 { src1 } else { src1 % src2 },     // REMU

                    _ => return Err(format!("Unknown OP funct7={:#x} funct3={}", funct7, funct3)),
                };
                self.write_reg(rd, val);
            }
            // Atomic A Extension (opcode=0x2F)
            0x2F => {
                let funct5 = funct7 >> 2;
                let addr = self.read_reg(rs1);
                let src2 = self.read_reg(rs2);
                let old_val = mem.read_u32(addr);
                let mut rd_val = old_val;
                let new_val = match funct5 {
                    0x02 => old_val, // LR.W
                    0x03 => { rd_val = 0; src2 } // SC.W
                    0x01 => src2, // AMOSWAP.W
                    0x00 => old_val.wrapping_add(src2), // AMOADD.W
                    0x04 => old_val ^ src2, // AMOXOR.W
                    0x0C => old_val & src2, // AMOAND.W
                    0x08 => old_val | src2, // AMOOR.W
                    0x10 => std::cmp::min(old_val as i32, src2 as i32) as u32, // AMOMIN.W
                    0x14 => std::cmp::max(old_val as i32, src2 as i32) as u32, // AMOMAX.W
                    0x18 => std::cmp::min(old_val, src2), // AMOMINU.W
                    0x1C => std::cmp::max(old_val, src2), // AMOMAXU.W
                    _ => return Err(format!("Unknown Atomic funct5: {:#x}", funct5)),
                };
                mem.write_u32(addr, new_val);
                self.write_reg(rd, rd_val);
            }
            // F & D Floating Point Loads (opcode=0x07)
            0x07 => {
                let imm = (inst as i32) >> 20;
                let addr = self.read_reg(rs1).wrapping_add(imm as u32);
                if funct3 == 2 { // FLW
                    let bits = mem.read_u32(addr);
                    self.write_f32(rd, f32::from_bits(bits));
                } else if funct3 == 3 { // FLD
                    let b0 = mem.read_u32(addr) as u64;
                    let b1 = mem.read_u32(addr + 4) as u64;
                    self.write_f64(rd, f64::from_bits(b0 | (b1 << 32)));
                }
            }
            // F & D Floating Point Stores (opcode=0x27)
            0x27 => {
                let imm11_5 = (inst >> 25) & 0x7F;
                let imm4_0 = (inst >> 7) & 0x1F;
                let offset = (imm11_5 << 5) | imm4_0;
                let sign_ext = ((offset as i32) << 20) >> 20;
                let addr = self.read_reg(rs1).wrapping_add(sign_ext as u32);
                if funct3 == 2 { // FSW
                    let bits = self.read_f32(rs2).to_bits();
                    mem.write_u32(addr, bits);
                } else if funct3 == 3 { // FSD
                    let bits = self.read_f64(rs2).to_bits();
                    mem.write_u32(addr, bits as u32);
                    mem.write_u32(addr + 4, (bits >> 32) as u32);
                }
            }
            // OP-FP Floating Point Compute (opcode=0x53)
            0x53 => {
                let fmt = funct7 & 0x3;
                let funct5 = funct7 >> 2;
                match (fmt, funct5) {
                    // Single precision (.s) fmt == 0
                    (0, 0x00) => { // FADD.S
                        self.write_f32(rd, self.read_f32(rs1) + self.read_f32(rs2));
                    }
                    (0, 0x01) => { // FSUB.S
                        self.write_f32(rd, self.read_f32(rs1) - self.read_f32(rs2));
                    }
                    (0, 0x02) => { // FMUL.S
                        self.write_f32(rd, self.read_f32(rs1) * self.read_f32(rs2));
                    }
                    (0, 0x03) => { // FDIV.S
                        self.write_f32(rd, self.read_f32(rs1) / self.read_f32(rs2));
                    }
                    (0, 0x0B) => { // FSQRT.S
                        self.write_f32(rd, self.read_f32(rs1).sqrt());
                    }
                    (0, 0x04) => { // FSGNJ / FSGNJN / FSGNJX .S
                        let s1 = self.read_f32(rs1);
                        let s2 = self.read_f32(rs2);
                        let b1 = s1.to_bits();
                        let b2 = s2.to_bits();
                        let res_bits = match funct3 {
                            0 => (b1 & 0x7FFFFFFF) | (b2 & 0x80000000),         // FSGNJ
                            1 => (b1 & 0x7FFFFFFF) | ((!b2) & 0x80000000),      // FSGNJN
                            2 => (b1 & 0x7FFFFFFF) | ((b1 ^ b2) & 0x80000000),  // FSGNJX
                            _ => return Err(format!("Unknown FSGNJ funct3: {}", funct3)),
                        };
                        self.write_f32(rd, f32::from_bits(res_bits));
                    }
                    (0, 0x05) => { // FMIN / FMAX .S
                        let s1 = self.read_f32(rs1);
                        let s2 = self.read_f32(rs2);
                        let res = match funct3 {
                            0 => if s1.is_nan() { s2 } else if s2.is_nan() { s1 } else { s1.min(s2) }, // FMIN.S
                            1 => if s1.is_nan() { s2 } else if s2.is_nan() { s1 } else { s1.max(s2) }, // FMAX.S
                            _ => return Err(format!("Unknown FMIN/FMAX funct3: {}", funct3)),
                        };
                        self.write_f32(rd, res);
                    }
                    (0, 0x18) => { // FCVT.W.S / FCVT.WU.S
                        let s = self.read_f32(rs1);
                        let val = if rs2 == 0 { // FCVT.W.S
                            s as i32 as u32
                        } else { // FCVT.WU.S
                            s as u32
                        };
                        self.write_reg(rd, val);
                    }
                    (0, 0x1A) => { // FCVT.S.W / FCVT.S.WU
                        let val = self.read_reg(rs1);
                        let s = if rs2 == 0 { // FCVT.S.W
                            (val as i32) as f32
                        } else { // FCVT.S.WU
                            val as f32
                        };
                        self.write_f32(rd, s);
                    }
                    (0, 0x1C) => {
                        if funct3 == 0 { // FMV.X.W
                            let bits = self.read_f32(rs1).to_bits();
                            self.write_reg(rd, bits);
                        } else if funct3 == 1 { // FCLASS.S
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
                            } else {
                                if is_neg { 1 << 1 } else { 1 << 6 }
                            };
                            self.write_reg(rd, mask);
                        }
                    }
                    (0, 0x1E) => { // FMV.W.X
                        let val = self.read_reg(rs1);
                        self.write_f32(rd, f32::from_bits(val));
                    }
                    (0, 0x14) => { // FEQ.S / FLT.S / FLE.S
                        let s1 = self.read_f32(rs1);
                        let s2 = self.read_f32(rs2);
                        let res = match funct3 {
                            0 => if s1 <= s2 { 1 } else { 0 }, // FLE.S
                            1 => if s1 < s2 { 1 } else { 0 },  // FLT.S
                            2 => if s1 == s2 { 1 } else { 0 }, // FEQ.S
                            _ => return Err(format!("Unknown FCOMP funct3: {}", funct3)),
                        };
                        self.write_reg(rd, res);
                    }
                    (0, 0x08) => { // FCVT.S.D (rs2=1)
                        self.write_f32(rd, self.read_f64(rs1) as f32);
                    }

                    // Double precision (.d) fmt == 1
                    (1, 0x00) => { // FADD.D
                        self.write_f64(rd, self.read_f64(rs1) + self.read_f64(rs2));
                    }
                    (1, 0x01) => { // FSUB.D
                        self.write_f64(rd, self.read_f64(rs1) - self.read_f64(rs2));
                    }
                    (1, 0x02) => { // FMUL.D
                        self.write_f64(rd, self.read_f64(rs1) * self.read_f64(rs2));
                    }
                    (1, 0x03) => { // FDIV.D
                        self.write_f64(rd, self.read_f64(rs1) / self.read_f64(rs2));
                    }
                    (1, 0x0B) => { // FSQRT.D
                        self.write_f64(rd, self.read_f64(rs1).sqrt());
                    }
                    (1, 0x04) => { // FSGNJ / FSGNJN / FSGNJX .D
                        let b1 = self.fregs[rs1].to_bits();
                        let b2 = self.fregs[rs2].to_bits();
                        let res_bits = match funct3 {
                            0 => (b1 & 0x7FFFFFFFFFFFFFFF) | (b2 & 0x8000000000000000),        // FSGNJ.D
                            1 => (b1 & 0x7FFFFFFFFFFFFFFF) | ((!b2) & 0x8000000000000000),     // FSGNJN.D
                            2 => (b1 & 0x7FFFFFFFFFFFFFFF) | ((b1 ^ b2) & 0x8000000000000000), // FSGNJX.D
                            _ => return Err(format!("Unknown FSGNJ.D funct3: {}", funct3)),
                        };
                        self.fregs[rd] = f64::from_bits(res_bits);
                    }
                    (1, 0x05) => { // FMIN / FMAX .D
                        let d1 = self.read_f64(rs1);
                        let d2 = self.read_f64(rs2);
                        let res = match funct3 {
                            0 => if d1.is_nan() { d2 } else if d2.is_nan() { d1 } else { d1.min(d2) }, // FMIN.D
                            1 => if d1.is_nan() { d2 } else if d2.is_nan() { d1 } else { d1.max(d2) }, // FMAX.D
                            _ => return Err(format!("Unknown FMIN/FMAX.D funct3: {}", funct3)),
                        };
                        self.write_f64(rd, res);
                    }
                    (1, 0x08) => { // FCVT.D.S (rs2=0)
                        self.write_f64(rd, self.read_f32(rs1) as f64);
                    }
                    (1, 0x18) => { // FCVT.W.D / FCVT.WU.D
                        let d = self.read_f64(rs1);
                        let val = if rs2 == 0 { // FCVT.W.D
                            d as i32 as u32
                        } else { // FCVT.WU.D
                            d as u32
                        };
                        self.write_reg(rd, val);
                    }
                    (1, 0x1A) => { // FCVT.D.W / FCVT.D.WU
                        let val = self.read_reg(rs1);
                        let d = if rs2 == 0 { // FCVT.D.W
                            (val as i32) as f64
                        } else { // FCVT.D.WU
                            val as f64
                        };
                        self.write_f64(rd, d);
                    }
                    (1, 0x1C) => {
                        if funct3 == 1 { // FCLASS.D
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
                            } else {
                                if is_neg { 1 << 1 } else { 1 << 6 }
                            };
                            self.write_reg(rd, mask);
                        }
                    }
                    (1, 0x14) => { // FEQ.D / FLT.D / FLE.D
                        let d1 = self.read_f64(rs1);
                        let d2 = self.read_f64(rs2);
                        let res = match funct3 {
                            0 => if d1 <= d2 { 1 } else { 0 }, // FLE.D
                            1 => if d1 < d2 { 1 } else { 0 },  // FLT.D
                            2 => if d1 == d2 { 1 } else { 0 }, // FEQ.D
                            _ => return Err(format!("Unknown FCOMP.D funct3: {}", funct3)),
                        };
                        self.write_reg(rd, res);
                    }
                    _ => return Err(format!("Unknown OP-FP fmt={} funct5={:#x}", fmt, funct5)),
                }
            }
            // FMADD, FMSUB, FNMSUB, FNMADD (opcodes 0x43, 0x47, 0x4B, 0x4F)
            0x43 | 0x47 | 0x4B | 0x4F => {
                let rs3 = ((inst >> 27) & 0x1F) as usize;
                let fmt = (inst >> 25) & 3;
                if fmt == 0 { // Single precision .s
                    let s1 = self.read_f32(rs1);
                    let s2 = self.read_f32(rs2);
                    let s3 = self.read_f32(rs3);
                    let res = match opcode {
                        0x43 => (s1 * s2) + s3,    // FMADD.S
                        0x47 => (s1 * s2) - s3,    // FMSUB.S
                        0x4B => -((s1 * s2) - s3), // FNMSUB.S
                        0x4F => -((s1 * s2) + s3), // FNMADD.S
                        _ => unreachable!(),
                    };
                    self.write_f32(rd, res);
                } else if fmt == 1 { // Double precision .d
                    let d1 = self.read_f64(rs1);
                    let d2 = self.read_f64(rs2);
                    let d3 = self.read_f64(rs3);
                    let res = match opcode {
                        0x43 => (d1 * d2) + d3,    // FMADD.D
                        0x47 => (d1 * d2) - d3,    // FMSUB.D
                        0x4B => -((d1 * d2) - d3), // FNMSUB.D
                        0x4F => -((d1 * d2) + d3), // FNMADD.D
                        _ => unreachable!(),
                    };
                    self.write_f64(rd, res);
                } else {
                    return Err(format!("Unsupported FMA fmt: {}", fmt));
                }
            }
            // SYSTEM / ECALL / CSR (opcode=0x73)
            0x73 => {
                if funct3 == 0 {
                    let imm12 = inst >> 20;
                    if imm12 == 0 {
                        // ECALL
                        handle_ecall(self, mem);
                    } else if imm12 == 1 {
                        // EBREAK
                        self.is_halted = true;
                    } else if imm12 == 0x302 {
                        // MRET: set PC to mepc
                        if let Some(&mepc) = self.csrs.get(&0x341) {
                            next_pc = mepc;
                        }
                    }
                } else {
                    // CSR instructions
                    let csr_num = ((inst >> 20) & 0xFFF) as u16;
                    let old_val = *self.csrs.get(&csr_num).unwrap_or(&0);
                    let src1_val = self.read_reg(rs1);
                    let new_val = match funct3 {
                        1 => src1_val,                  // CSRRW
                        2 => old_val | src1_val,        // CSRRS
                        3 => old_val & !src1_val,       // CSRRC
                        5 => rs1 as u32,                // CSRRWI
                        6 => old_val | (rs1 as u32),    // CSRRSI
                        7 => old_val & !(rs1 as u32),   // CSRRCI
                        _ => old_val,
                    };
                    self.csrs.insert(csr_num, new_val);
                    self.write_reg(rd, old_val);
                }
            }
            // FENCE / FENCE.I (opcode=0x0F)
            0x0F => {}
            _ => return Err(format!("Unrecognized 32-bit opcode: {:#x}", opcode)),
        }

        self.pc = next_pc;
        Ok(())
    }

    pub fn execute_c_inst<M: MemoryOps>(&mut self, inst: u16, mem: &mut M) -> Result<(), String> {
        let op = inst & 0x3;
        let funct3 = (inst >> 13) & 0x7;
        let mut next_pc = self.pc.wrapping_add(2);

        match (op, funct3) {
            // Quadrant 0
            (0, 0) => { // C.ADDI4SPN
                let rdc = (((inst >> 2) & 0x7) + 8) as usize;
                let imm = (((inst >> 7) & 0x30) | ((inst >> 1) & 0x3C0) | ((inst >> 4) & 0x4) | ((inst >> 2) & 0x8)) as u32;
                if imm != 0 {
                    self.write_reg(rdc, self.read_reg(2).wrapping_add(imm));
                }
            }
            (0, 2) => { // C.LW
                let rdc = (((inst >> 2) & 0x7) + 8) as usize;
                let rs1c = (((inst >> 7) & 0x7) + 8) as usize;
                let offset = (((inst >> 6) & 0x4) | ((inst >> 10) & 0x38) | ((inst >> 3) & 0x40) | ((inst >> 2) & 0x8)) as u32;
                let addr = self.read_reg(rs1c).wrapping_add(offset);
                self.write_reg(rdc, mem.read_u32(addr));
            }
            (0, 6) => { // C.SW
                let rs2c = (((inst >> 2) & 0x7) + 8) as usize;
                let rs1c = (((inst >> 7) & 0x7) + 8) as usize;
                let offset = (((inst >> 6) & 0x4) | ((inst >> 10) & 0x38) | ((inst >> 3) & 0x40) | ((inst >> 2) & 0x8)) as u32;
                let addr = self.read_reg(rs1c).wrapping_add(offset);
                mem.write_u32(addr, self.read_reg(rs2c));
            }

            // Quadrant 1
            (1, 0) => { // C.NOP / C.ADDI
                let rd = ((inst >> 7) & 0x1F) as usize;
                let imm6 = (((inst >> 12) & 1) << 5) | ((inst >> 2) & 0x1F);
                let sign_ext = ((imm6 as i16) << 10) >> 10;
                if rd != 0 && sign_ext != 0 {
                    self.write_reg(rd, self.read_reg(rd).wrapping_add(sign_ext as i32 as u32));
                }
            }
            (1, 1) => { // C.JAL
                let imm11 = (((inst >> 12) & 1) << 11) | (((inst >> 8) & 1) << 10) | (((inst >> 9) & 3) << 8) | (((inst >> 6) & 1) << 7) | (((inst >> 7) & 1) << 6) | (((inst >> 2) & 1) << 5) | (((inst >> 11) & 1) << 4) | (((inst >> 3) & 7) << 1);
                let offset = ((imm11 as i16) << 4) >> 4;
                self.write_reg(1, next_pc);
                next_pc = self.pc.wrapping_add(offset as i32 as u32);
            }
            (1, 2) => { // C.LI
                let rd = ((inst >> 7) & 0x1F) as usize;
                let imm6 = (((inst >> 12) & 1) << 5) | ((inst >> 2) & 0x1F);
                let sign_ext = ((imm6 as i16) << 10) >> 10;
                self.write_reg(rd, sign_ext as i32 as u32);
            }
            (1, 3) => { // C.ADDI16SP / C.LUI
                let rd = ((inst >> 7) & 0x1F) as usize;
                if rd == 2 { // C.ADDI16SP
                    let imm = (((inst >> 12) & 1) << 9) | (((inst >> 3) & 3) << 7) | (((inst >> 5) & 1) << 6) | (((inst >> 2) & 1) << 5) | (((inst >> 6) & 1) << 4);
                    let offset = ((imm as i16) << 6) >> 6;
                    self.write_reg(2, self.read_reg(2).wrapping_add((offset as i32 as u32) << 4));
                } else if rd != 0 { // C.LUI
                    let imm6 = (((inst >> 12) & 1) << 5) | ((inst >> 2) & 0x1F);
                    let sign_ext = ((imm6 as i16) << 10) >> 10;
                    self.write_reg(rd, (sign_ext as i32 as u32) << 12);
                }
            }
            (1, 5) => { // C.J
                let imm11 = (((inst >> 12) & 1) << 11) | (((inst >> 8) & 1) << 10) | (((inst >> 9) & 3) << 8) | (((inst >> 6) & 1) << 7) | (((inst >> 7) & 1) << 6) | (((inst >> 2) & 1) << 5) | (((inst >> 11) & 1) << 4) | (((inst >> 3) & 7) << 1);
                let offset = ((imm11 as i16) << 4) >> 4;
                next_pc = self.pc.wrapping_add(offset as i32 as u32);
            }
            (1, 6) => { // C.BEQZ
                let rs1c = (((inst >> 7) & 0x7) + 8) as usize;
                let imm = (((inst >> 12) & 1) << 8) | (((inst >> 5) & 3) << 6) | (((inst >> 2) & 1) << 5) | (((inst >> 10) & 3) << 3) | (((inst >> 3) & 3) << 1);
                let offset = ((imm as i16) << 7) >> 7;
                if self.read_reg(rs1c) == 0 {
                    next_pc = self.pc.wrapping_add(offset as i32 as u32);
                }
            }
            (1, 7) => { // C.BNEZ
                let rs1c = (((inst >> 7) & 0x7) + 8) as usize;
                let imm = (((inst >> 12) & 1) << 8) | (((inst >> 5) & 3) << 6) | (((inst >> 2) & 1) << 5) | (((inst >> 10) & 3) << 3) | (((inst >> 3) & 3) << 1);
                let offset = ((imm as i16) << 7) >> 7;
                if self.read_reg(rs1c) != 0 {
                    next_pc = self.pc.wrapping_add(offset as i32 as u32);
                }
            }

            // Quadrant 2
            (2, 0) => { // C.SLLI
                let rd = ((inst >> 7) & 0x1F) as usize;
                let shamt = (((inst >> 12) & 1) << 5) | ((inst >> 2) & 0x1F);
                if rd != 0 {
                    self.write_reg(rd, self.read_reg(rd) << shamt);
                }
            }
            (2, 2) => { // C.LWSP
                let rd = ((inst >> 7) & 0x1F) as usize;
                let offset = (((inst >> 2) & 0x1C) | ((inst >> 12) & 1) << 5 | ((inst >> 7) & 0x3) << 6) as u32;
                let addr = self.read_reg(2).wrapping_add(offset);
                if rd != 0 {
                    self.write_reg(rd, mem.read_u32(addr));
                }
            }
            (2, 4) => {
                let rd = ((inst >> 7) & 0x1F) as usize;
                let rs2 = ((inst >> 2) & 0x1F) as usize;
                let bit12 = (inst >> 12) & 1;
                if bit12 == 0 && rs2 == 0 { // C.JR
                    if rd != 0 {
                        next_pc = self.read_reg(rd) & !1;
                    }
                } else if bit12 == 0 && rs2 != 0 { // C.MV
                    if rd != 0 {
                        self.write_reg(rd, self.read_reg(rs2));
                    }
                } else if bit12 == 1 && rd != 0 && rs2 == 0 { // C.JALR
                    let target = self.read_reg(rd) & !1;
                    self.write_reg(1, next_pc);
                    next_pc = target;
                } else if bit12 == 1 && rd != 0 && rs2 != 0 { // C.ADD
                    self.write_reg(rd, self.read_reg(rd).wrapping_add(self.read_reg(rs2)));
                }
            }
            (2, 6) => { // C.SWSP
                let rs2 = ((inst >> 2) & 0x1F) as usize;
                let offset = (((inst >> 9) & 0x3C) | ((inst >> 7) & 0x3) << 6) as u32;
                let addr = self.read_reg(2).wrapping_add(offset);
                mem.write_u32(addr, self.read_reg(rs2));
            }

            _ => return Err(format!("Unrecognized 16-bit compressed instruction: {:#06x}", inst)),
        }

        self.pc = next_pc;
        Ok(())
    }
}
