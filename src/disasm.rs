use crate::cpu::{
    ci_double_sp_offset, ci_word_sp_offset, cl_double_offset, cl_word_offset, creg,
    css_double_sp_offset, css_word_sp_offset,
};
use crate::inst::*;
use crate::utils::ShiftThenMask;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tsify_next::Tsify;

#[derive(Clone, Debug, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct DisassembledInst {
    pub address: u32,
    pub opcode_hex: String,
    pub asm_text: String,
    pub is_compressed: bool,
    pub label: Option<String>,
}

pub fn reg_name(reg: u32) -> &'static str {
    match reg & 0x1f {
        0 => "zero",
        1 => "ra",
        2 => "sp",
        3 => "gp",
        4 => "tp",
        5 => "t0",
        6 => "t1",
        7 => "t2",
        8 => "s0",
        9 => "s1",
        10 => "a0",
        11 => "a1",
        12 => "a2",
        13 => "a3",
        14 => "a4",
        15 => "a5",
        16 => "a6",
        17 => "a7",
        18 => "s2",
        19 => "s3",
        20 => "s4",
        21 => "s5",
        22 => "s6",
        23 => "s7",
        24 => "s8",
        25 => "s9",
        26 => "s10",
        27 => "s11",
        28 => "t3",
        29 => "t4",
        30 => "t5",
        31 => "t6",
        _ => "unknown",
    }
}

pub fn freg_name(reg: u32) -> &'static str {
    match reg & 0x1f {
        0 => "ft0",
        1 => "ft1",
        2 => "ft2",
        3 => "ft3",
        4 => "ft4",
        5 => "ft5",
        6 => "ft6",
        7 => "ft7",
        8 => "fs0",
        9 => "fs1",
        10 => "fa0",
        11 => "fa1",
        12 => "fa2",
        13 => "fa3",
        14 => "fa4",
        15 => "fa5",
        16 => "fa6",
        17 => "fa7",
        18 => "fs2",
        19 => "fs3",
        20 => "fs4",
        21 => "fs5",
        22 => "fs6",
        23 => "fs7",
        24 => "fs8",
        25 => "fs9",
        26 => "fs10",
        27 => "fs11",
        28 => "ft8",
        29 => "ft9",
        30 => "ft10",
        31 => "ft11",
        _ => "unknown",
    }
}

fn format_target(addr: u32, symbols: Option<&HashMap<u32, String>>) -> String {
    if let Some(syms) = symbols {
        if let Some(name) = syms.get(&addr) {
            return format!("0x{:x} <{}>", addr, name);
        }
    }
    format!("0x{:x}", addr)
}

pub struct Disassembler;

impl Disassembler {
    pub fn decode_instruction(address: u32, opcode: u32) -> DisassembledInst {
        Self::decode_instruction_with_symbols(address, opcode, None)
    }

    pub fn decode_instruction_with_symbols(
        address: u32,
        opcode: u32,
        symbols: Option<&HashMap<u32, String>>,
    ) -> DisassembledInst {
        let is_compressed = (opcode & 0x3) != 0x3;
        let (hex_str, asm) = if is_compressed {
            let half = (opcode & 0xFFFF) as u16;
            (format!("{:04x}", half), Self::decode_rvc(address, half))
        } else {
            (
                format!("{:08x}", opcode),
                Self::decode_rv32(address, opcode, symbols),
            )
        };

        let label = symbols.and_then(|syms| syms.get(&address).cloned());

        DisassembledInst {
            address,
            opcode_hex: hex_str,
            asm_text: asm,
            is_compressed,
            label,
        }
    }

    fn decode_rv32(address: u32, raw_inst: u32, symbols: Option<&HashMap<u32, String>>) -> String {
        let decoded = DecodedInst32::decode(raw_inst);
        let rd = decoded.rd as u32;
        let rs1 = decoded.rs1 as u32;
        let rs2 = decoded.rs2 as u32;

        match decoded.opcode {
            // LUI
            OP_LUI => {
                let imm = (decoded.u_imm() as i32) >> 12;
                format!("lui {}, {}", reg_name(rd), imm)
            }
            // AUIPC
            OP_AUIPC => {
                let imm = (decoded.u_imm() as i32) >> 12;
                format!("auipc {}, {}", reg_name(rd), imm)
            }
            // JAL
            OP_JAL => {
                let sign_ext = decoded.j_imm();
                let target = address.wrapping_add(sign_ext as u32);
                let target_str = format_target(target, symbols);
                if rd == 0 {
                    format!("j {}", target_str)
                } else {
                    format!("jal {}, {}", reg_name(rd), target_str)
                }
            }
            // JALR
            OP_JALR => {
                let imm = decoded.i_imm();
                if rd == 0 && rs1 == 1 && imm == 0 {
                    "ret".to_string()
                } else if rd == 0 && imm == 0 {
                    format!("jr {}", reg_name(rs1))
                } else {
                    format!("jalr {}, {}({})", reg_name(rd), imm, reg_name(rs1))
                }
            }
            // Branch
            OP_BRANCH => {
                let sign_ext = decoded.b_imm();
                let target = address.wrapping_add(sign_ext as u32);
                let target_str = format_target(target, symbols);
                let op = match decoded.funct3 {
                    0x0 => {
                        if rs2 == 0 {
                            "beqz"
                        } else {
                            "beq"
                        }
                    }
                    0x1 => {
                        if rs2 == 0 {
                            "bnez"
                        } else {
                            "bne"
                        }
                    }
                    0x4 => {
                        if rs2 == 0 {
                            "bltz"
                        } else {
                            "blt"
                        }
                    }
                    0x5 => {
                        if rs2 == 0 {
                            "bgez"
                        } else {
                            "bge"
                        }
                    }
                    0x6 => "bltu",
                    0x7 => "bgeu",
                    _ => "unknown_branch",
                };
                if op == "beqz" || op == "bnez" || op == "bltz" || op == "bgez" {
                    format!("{} {}, {}", op, reg_name(rs1), target_str)
                } else {
                    format!(
                        "{} {}, {}, {}",
                        op,
                        reg_name(rs1),
                        reg_name(rs2),
                        target_str
                    )
                }
            }
            // Load
            OP_LOAD => {
                let imm = decoded.i_imm();
                let op = match decoded.funct3 {
                    0x0 => "lb",
                    0x1 => "lh",
                    0x2 => "lw",
                    0x4 => "lbu",
                    0x5 => "lhu",
                    _ => "load_unknown",
                };
                format!("{} {}, {}({})", op, reg_name(rd), imm, reg_name(rs1))
            }
            // Store
            OP_STORE => {
                let imm = decoded.s_imm();
                let op = match decoded.funct3 {
                    0x0 => "sb",
                    0x1 => "sh",
                    0x2 => "sw",
                    _ => "store_unknown",
                };
                format!("{} {}, {}({})", op, reg_name(rs2), imm, reg_name(rs1))
            }
            // OP-IMM
            OP_IMM => {
                let imm = decoded.i_imm();
                let shamt = rs2;
                match decoded.funct3 {
                    0x0 => {
                        if raw_inst == 0x00000013 {
                            "nop".to_string()
                        } else if rs1 == 0 {
                            format!("li {}, {}", reg_name(rd), imm)
                        } else if imm == 0 {
                            format!("mv {}, {}", reg_name(rd), reg_name(rs1))
                        } else {
                            format!("addi {}, {}, {}", reg_name(rd), reg_name(rs1), imm)
                        }
                    }
                    0x2 => format!("slti {}, {}, {}", reg_name(rd), reg_name(rs1), imm),
                    0x3 => format!("sltiu {}, {}, {}", reg_name(rd), reg_name(rs1), imm as u32),
                    0x4 => format!("xori {}, {}, {}", reg_name(rd), reg_name(rs1), imm),
                    0x6 => format!("ori {}, {}, {}", reg_name(rd), reg_name(rs1), imm),
                    0x7 => format!("andi {}, {}, {}", reg_name(rd), reg_name(rs1), imm),
                    0x1 => format!("slli {}, {}, {}", reg_name(rd), reg_name(rs1), shamt),
                    0x5 => {
                        if decoded.funct7 == 0x20 {
                            format!("srai {}, {}, {}", reg_name(rd), reg_name(rs1), shamt)
                        } else {
                            format!("srli {}, {}, {}", reg_name(rd), reg_name(rs1), shamt)
                        }
                    }
                    _ => format!("op_imm_unknown 0x{:08x}", raw_inst),
                }
            }
            // OP
            OP_OP => {
                let op = match (decoded.funct7, decoded.funct3) {
                    (0x00, 0x0) => "add",
                    (0x20, 0x0) => "sub",
                    (0x00, 0x1) => "sll",
                    (0x00, 0x2) => "slt",
                    (0x00, 0x3) => "sltu",
                    (0x00, 0x4) => "xor",
                    (0x00, 0x5) => "srl",
                    (0x20, 0x5) => "sra",
                    (0x00, 0x6) => "or",
                    (0x00, 0x7) => "and",
                    (0x01, 0x0) => "mul",
                    (0x01, 0x1) => "mulh",
                    (0x01, 0x2) => "mulhsu",
                    (0x01, 0x3) => "mulhu",
                    (0x01, 0x4) => "div",
                    (0x01, 0x5) => "divu",
                    (0x01, 0x6) => "rem",
                    (0x01, 0x7) => "remu",
                    _ => "op_unknown",
                };
                format!(
                    "{} {}, {}, {}",
                    op,
                    reg_name(rd),
                    reg_name(rs1),
                    reg_name(rs2)
                )
            }
            // FLW / FSW
            OP_LOAD_FP => {
                let imm = decoded.i_imm();
                format!("flw {}, {}({})", freg_name(rd), imm, reg_name(rs1))
            }
            OP_STORE_FP => {
                let imm = decoded.s_imm();
                format!("fsw {}, {}({})", freg_name(rs2), imm, reg_name(rs1))
            }
            // SYSTEM
            OP_SYSTEM => {
                if raw_inst == 0x00000073 {
                    "ecall".to_string()
                } else if raw_inst == 0x00100073 {
                    "ebreak".to_string()
                } else if raw_inst == 0x30200073 {
                    "mret".to_string()
                } else {
                    let csr = raw_inst >> 20;
                    let op = match decoded.funct3 {
                        0x1 => "csrrw",
                        0x2 => "csrrs",
                        0x3 => "csrrc",
                        0x5 => "csrrwi",
                        0x6 => "csrrsi",
                        0x7 => "csrrci",
                        _ => "csr_op",
                    };
                    format!("{} {}, 0x{:x}, {}", op, reg_name(rd), csr, reg_name(rs1))
                }
            }
            _ => format!("unimpl 0x{:08x}", raw_inst),
        }
    }

    fn decode_rvc(address: u32, inst: u16) -> String {
        let op = inst & 0x3;
        let funct3 = inst.shift_then_mask(13, 0x7);
        let rd = inst.shift_then_mask(7, 0x1F) as u32;
        let rs2 = inst.shift_then_mask(2, 0x1F) as u32;
        let rdc = creg(inst, 2) as u32;
        let rs1c = creg(inst, 7) as u32;
        let imm6 = ((inst.shift_then_mask(12, 1) << 5) | inst.shift_then_mask(2, 0x1F)) as i16;
        let imm6_sext = ((imm6 << 10) >> 10) as i32;

        match (op, funct3) {
            // Quadrant 0
            (0x0, 0x0) => {
                let imm = inst.shift_then_mask(7, 0x30)
                    | inst.shift_then_mask(1, 0x3C0)
                    | inst.shift_then_mask(4, 0x4)
                    | inst.shift_then_mask(2, 0x8);
                if imm == 0 {
                    format!("c.reserved 0x{:04x}", inst)
                } else {
                    format!("c.addi4spn {}, sp, {}", reg_name(rdc), imm)
                }
            }
            (0x0, 0x1) => format!(
                "c.fld {}, {}({})",
                freg_name(rdc),
                cl_double_offset(inst),
                reg_name(rs1c)
            ),
            (0x0, 0x2) => format!(
                "c.lw {}, {}({})",
                reg_name(rdc),
                cl_word_offset(inst),
                reg_name(rs1c)
            ),
            (0x0, 0x3) => format!(
                "c.flw {}, {}({})",
                freg_name(rdc),
                cl_word_offset(inst),
                reg_name(rs1c)
            ),
            (0x0, 0x5) => format!(
                "c.fsd {}, {}({})",
                freg_name(rdc),
                cl_double_offset(inst),
                reg_name(rs1c)
            ),
            (0x0, 0x6) => format!(
                "c.sw {}, {}({})",
                reg_name(rdc),
                cl_word_offset(inst),
                reg_name(rs1c)
            ),
            (0x0, 0x7) => format!(
                "c.fsw {}, {}({})",
                freg_name(rdc),
                cl_word_offset(inst),
                reg_name(rs1c)
            ),

            // Quadrant 1
            (0x1, 0x0) => {
                if rd == 0 {
                    "c.nop".to_string()
                } else {
                    format!("c.addi {}, {}", reg_name(rd), imm6_sext)
                }
            }
            (0x1, 0x1) => format!("c.jal {}", format_target_rvc(address, inst)),
            (0x1, 0x2) => format!("c.li {}, {}", reg_name(rd), imm6_sext),
            (0x1, 0x3) => {
                if rd == 2 {
                    let imm = (inst.shift_then_mask(12, 1) << 9)
                        | (inst.shift_then_mask(3, 3) << 7)
                        | (inst.shift_then_mask(5, 1) << 6)
                        | (inst.shift_then_mask(2, 1) << 5)
                        | (inst.shift_then_mask(6, 1) << 4);
                    if imm == 0 {
                        format!("c.reserved 0x{:04x}", inst)
                    } else {
                        format!("c.addi16sp sp, {}", (((imm as i16) << 6) >> 6) as i32)
                    }
                } else if rd == 0 || imm6 == 0 {
                    format!("c.reserved 0x{:04x}", inst)
                } else {
                    format!("c.lui {}, {}", reg_name(rd), imm6_sext)
                }
            }
            (0x1, 0x4) => {
                let bit12 = inst.shift_then_mask(12, 1);
                match inst.shift_then_mask(10, 0x3) {
                    0 | 1 => {
                        let op_name = if inst.shift_then_mask(10, 0x3) == 0 {
                            "c.srli"
                        } else {
                            "c.srai"
                        };
                        if bit12 != 0 {
                            format!("c.reserved 0x{:04x}", inst)
                        } else {
                            format!(
                                "{} {}, {}",
                                op_name,
                                reg_name(rs1c),
                                inst.shift_then_mask(2, 0x1F)
                            )
                        }
                    }
                    2 => format!("c.andi {}, {}", reg_name(rs1c), imm6_sext),
                    _ => {
                        if bit12 != 0 {
                            format!("c.reserved 0x{:04x}", inst)
                        } else {
                            let op_name = match inst.shift_then_mask(5, 0x3) {
                                0 => "c.sub",
                                1 => "c.xor",
                                2 => "c.or",
                                _ => "c.and",
                            };
                            format!("{} {}, {}", op_name, reg_name(rs1c), reg_name(rdc))
                        }
                    }
                }
            }
            (0x1, 0x5) => format!("c.j {}", format_target_rvc(address, inst)),
            (0x1, 0x6) => format!(
                "c.beqz {}, {}",
                reg_name(rs1c),
                format_branch_target_rvc(address, inst)
            ),
            (0x1, 0x7) => format!(
                "c.bnez {}, {}",
                reg_name(rs1c),
                format_branch_target_rvc(address, inst)
            ),

            // Quadrant 2
            (0x2, 0x0) => {
                if inst.shift_then_mask(12, 1) != 0 {
                    format!("c.reserved 0x{:04x}", inst)
                } else {
                    format!("c.slli {}, {}", reg_name(rd), inst.shift_then_mask(2, 0x1F))
                }
            }
            (0x2, 0x1) => format!(
                "c.fldsp {}, {}(sp)",
                freg_name(rd),
                ci_double_sp_offset(inst)
            ),
            (0x2, 0x2) => {
                if rd == 0 {
                    format!("c.reserved 0x{:04x}", inst)
                } else {
                    format!("c.lwsp {}, {}(sp)", reg_name(rd), ci_word_sp_offset(inst))
                }
            }
            (0x2, 0x3) => format!("c.flwsp {}, {}(sp)", freg_name(rd), ci_word_sp_offset(inst)),
            (0x2, 0x4) => {
                let bit12 = inst.shift_then_mask(12, 1);
                if bit12 == 0 && rs2 == 0 {
                    if rd == 0 {
                        format!("c.reserved 0x{:04x}", inst)
                    } else {
                        format!("c.jr {}", reg_name(rd))
                    }
                } else if bit12 == 0 {
                    format!("c.mv {}, {}", reg_name(rd), reg_name(rs2))
                } else if rd == 0 && rs2 == 0 {
                    "c.ebreak".to_string()
                } else if rs2 == 0 {
                    format!("c.jalr {}", reg_name(rd))
                } else {
                    format!("c.add {}, {}", reg_name(rd), reg_name(rs2))
                }
            }
            (0x2, 0x5) => format!(
                "c.fsdsp {}, {}(sp)",
                freg_name(rs2),
                css_double_sp_offset(inst)
            ),
            (0x2, 0x6) => format!("c.swsp {}, {}(sp)", reg_name(rs2), css_word_sp_offset(inst)),
            (0x2, 0x7) => format!(
                "c.fswsp {}, {}(sp)",
                freg_name(rs2),
                css_word_sp_offset(inst)
            ),

            _ => format!("c.reserved 0x{:04x}", inst),
        }
    }
}

/// Sign-extended jump offset of the CJ format (`c.j`, `c.jal`), rendered as an
/// absolute target address.
fn format_target_rvc(address: u32, inst: u16) -> String {
    let imm11 = (inst.shift_then_mask(12, 1) << 11)
        | (inst.shift_then_mask(8, 1) << 10)
        | (inst.shift_then_mask(9, 3) << 8)
        | (inst.shift_then_mask(6, 1) << 7)
        | (inst.shift_then_mask(7, 1) << 6)
        | (inst.shift_then_mask(2, 1) << 5)
        | (inst.shift_then_mask(11, 1) << 4)
        | (inst.shift_then_mask(3, 7) << 1);
    let offset = (((imm11 as i16) << 4) >> 4) as i32;
    format!("0x{:x}", address.wrapping_add(offset as u32))
}

/// Sign-extended branch offset of the CB format (`c.beqz`, `c.bnez`), rendered
/// as an absolute target address.
fn format_branch_target_rvc(address: u32, inst: u16) -> String {
    let imm = (inst.shift_then_mask(12, 1) << 8)
        | (inst.shift_then_mask(5, 3) << 6)
        | (inst.shift_then_mask(2, 1) << 5)
        | (inst.shift_then_mask(10, 3) << 3)
        | (inst.shift_then_mask(3, 3) << 1);
    let offset = (((imm as i16) << 7) >> 7) as i32;
    format!("0x{:x}", address.wrapping_add(offset as u32))
}
