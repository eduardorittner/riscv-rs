use crate::utils::ShiftThenMask;

/// Load Upper Immediate: Places 20-bit upper immediate into rd, zeroing lower 12 bits (rd = imm << 12).
pub const OP_LUI: u8 = 0x37;

/// Add Upper Immediate to PC: Adds 20-bit upper immediate to PC and stores in rd (rd = PC + (imm << 12)).
pub const OP_AUIPC: u8 = 0x17;

/// Jump and Link: Unconditional PC-relative jump, stores return address PC+4 in rd (rd = PC+4; PC += offset).
pub const OP_JAL: u8 = 0x6F;

/// Jump and Link Register: Indirect jump to (rs1 + imm) & !1, stores return address PC+4 in rd.
pub const OP_JALR: u8 = 0x67;

/// Conditional Branch: Compares rs1 & rs2 and jumps to PC + offset if condition met (BEQ, BNE, BLT, BGE, etc.).
pub const OP_BRANCH: u8 = 0x63;

/// Memory Load: Reads byte/half/word from memory address (rs1 + offset) into rd (LB, LH, LW, LBU, LHU).
pub const OP_LOAD: u8 = 0x03;

/// Memory Store: Writes byte/half/word from register rs2 to memory address (rs1 + offset) (SB, SH, SW).
pub const OP_STORE: u8 = 0x23;

/// Integer Immediate Operations: Register-immediate arithmetic & logic (ADDI, SLTI, XORI, ORI, ANDI, SLLI, SRLI, SRAI).
pub const OP_IMM: u8 = 0x13;

/// Integer Register Operations: Register-register arithmetic & logic (ADD, SUB, SLL, SLT, XOR, SRL, SRA, OR, AND, and RV32M MUL/DIV).
pub const OP_OP: u8 = 0x33;

/// Miscellaneous Memory: Memory and instruction ordering synchronization barriers (FENCE, FENCE.I).
pub const OP_MISC_MEM: u8 = 0x0F;

/// System Operations: CSR read/write (CSRRW, CSRRS, CSRRC), environment call (ECALL), breakpoint (EBREAK), and mret.
pub const OP_SYSTEM: u8 = 0x73;

/// Atomic Memory Operations: Read-modify-write atomic memory primitives (LR.W, SC.W, AMOSWAP, AMOADD, etc.).
pub const OP_AMO: u8 = 0x2F;

/// Floating-Point Load: Reads single/double precision float from memory (rs1 + offset) into FP register rd (FLW, FLD).
pub const OP_LOAD_FP: u8 = 0x07;

/// Floating-Point Store: Writes single/double precision float from FP register rs2 to memory (rs1 + offset) (FSW, FSD).
pub const OP_STORE_FP: u8 = 0x27;

/// FP Fused Multiply-Add: Computes (s1 * s2) + s3 for single/double precision floats without intermediate rounding.
pub const OP_MADD: u8 = 0x43;

/// FP Fused Multiply-Subtract: Computes (s1 * s2) - s3 for single/double precision floats.
pub const OP_MSUB: u8 = 0x47;

/// FP Fused Negated Multiply-Subtract: Computes -((s1 * s2) - s3) for single/double precision floats.
pub const OP_NMSUB: u8 = 0x4B;

/// FP Fused Negated Multiply-Add: Computes -((s1 * s2) + s3) for single/double precision floats.
pub const OP_NMADD: u8 = 0x4F;

/// Floating-Point Operations: FP arithmetic (FADD, FSUB, FMUL, FDIV, FSQRT), sign-injection, comparisons, and conversions.
pub const OP_OP_FP: u8 = 0x53;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedInst32 {
    pub raw: u32,
    pub opcode: u8,
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub rs3: usize,
    pub funct3: u8,
    pub funct7: u8,
}

impl DecodedInst32 {
    #[inline(always)]
    pub fn decode(inst: u32) -> Self {
        Self {
            raw: inst,
            opcode: (inst & 0x7F) as u8,
            rd: inst.shift_then_mask(7, 0x1F) as usize,
            funct3: inst.shift_then_mask(12, 0x7) as u8,
            rs1: inst.shift_then_mask(15, 0x1F) as usize,
            rs2: inst.shift_then_mask(20, 0x1F) as usize,
            rs3: inst.shift_then_mask(27, 0x1F) as usize,
            funct7: inst.shift_then_mask(25, 0x7F) as u8,
        }
    }

    #[inline(always)]
    pub fn i_imm(&self) -> i32 {
        (self.raw as i32) >> 20
    }

    #[inline(always)]
    pub fn s_imm(&self) -> i32 {
        let imm11_5 = self.raw.shift_then_mask(25, 0x7F);
        let imm4_0 = self.raw.shift_then_mask(7, 0x1F);
        let raw_s = (imm11_5 << 5) | imm4_0;
        ((raw_s as i32) << 20) >> 20
    }

    #[inline(always)]
    pub fn b_imm(&self) -> i32 {
        let imm12 = self.raw.shift_then_mask(31, 1);
        let imm10_5 = self.raw.shift_then_mask(25, 0x3F);
        let imm4_1 = self.raw.shift_then_mask(8, 0xF);
        let imm11 = self.raw.shift_then_mask(7, 1);
        let offset = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
        ((offset as i32) << 19) >> 19
    }

    #[inline(always)]
    pub fn u_imm(&self) -> u32 {
        self.raw & 0xFFFF_F000
    }

    #[inline(always)]
    pub fn j_imm(&self) -> i32 {
        let imm20 = self.raw.shift_then_mask(31, 1);
        let imm10_1 = self.raw.shift_then_mask(21, 0x3FF);
        let imm11 = self.raw.shift_then_mask(20, 1);
        let imm19_12 = self.raw.shift_then_mask(12, 0xFF);
        let offset = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
        ((offset as i32) << 11) >> 11
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedInst16 {
    pub raw: u16,
    pub op: u8,
    pub funct3: u8,
}

impl DecodedInst16 {
    #[inline(always)]
    pub fn decode(inst: u16) -> Self {
        Self {
            raw: inst,
            op: (inst & 0x3) as u8,
            funct3: inst.shift_then_mask(13, 0x7) as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoded_inst32() {
        // ADDI x1, x2, -5 => 0xFE510093
        let inst = DecodedInst32::decode(0xFE510093);
        assert_eq!(inst.opcode, OP_IMM);
        assert_eq!(inst.rd, 1);
        assert_eq!(inst.rs1, 2);
        assert_eq!(inst.funct3, 0);
        assert_eq!(inst.i_imm(), -27);
    }
}
