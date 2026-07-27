use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
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

    fn decode_rv32(address: u32, inst: u32, symbols: Option<&HashMap<u32, String>>) -> String {
        let opcode = inst & 0x7f;
        let rd = (inst >> 7) & 0x1f;
        let funct3 = (inst >> 12) & 0x7;
        let rs1 = (inst >> 15) & 0x1f;
        let rs2 = (inst >> 20) & 0x1f;
        let funct7 = (inst >> 25) & 0x7f;

        match opcode {
            // LUI
            0x37 => {
                let imm = ((inst & 0xfffff000) as i32) >> 12;
                format!("lui {}, {}", reg_name(rd), imm)
            }
            // AUIPC
            0x17 => {
                let imm = ((inst & 0xfffff000) as i32) >> 12;
                format!("auipc {}, {}", reg_name(rd), imm)
            }
            // JAL
            0x6f => {
                let imm20 = (inst >> 31) & 1;
                let imm10_1 = (inst >> 21) & 0x3ff;
                let imm11 = (inst >> 20) & 1;
                let imm19_12 = (inst >> 12) & 0xff;
                let offset = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                let sign_ext = ((offset as i32) << 11) >> 11;
                let target = address.wrapping_add(sign_ext as u32);
                let target_str = format_target(target, symbols);
                if rd == 0 {
                    format!("j {}", target_str)
                } else {
                    format!("jal {}, {}", reg_name(rd), target_str)
                }
            }
            // JALR
            0x67 => {
                let imm = (inst as i32) >> 20;
                if rd == 0 && rs1 == 1 && imm == 0 {
                    "ret".to_string()
                } else if rd == 0 && imm == 0 {
                    format!("jr {}", reg_name(rs1))
                } else {
                    format!("jalr {}, {}({})", reg_name(rd), imm, reg_name(rs1))
                }
            }
            // Branch
            0x63 => {
                let imm12 = (inst >> 31) & 1;
                let imm10_5 = (inst >> 25) & 0x3f;
                let imm4_1 = (inst >> 8) & 0xf;
                let imm11 = (inst >> 7) & 1;
                let offset = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
                let sign_ext = ((offset as i32) << 19) >> 19;
                let target = address.wrapping_add(sign_ext as u32);
                let target_str = format_target(target, symbols);
                let op = match funct3 {
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
            0x03 => {
                let imm = (inst as i32) >> 20;
                let op = match funct3 {
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
            0x23 => {
                let imm5 = (inst >> 7) & 0x1f;
                let imm7 = (inst >> 25) & 0x7f;
                let imm = ((((imm7 << 5) | imm5) as i32) << 20) >> 20;
                let op = match funct3 {
                    0x0 => "sb",
                    0x1 => "sh",
                    0x2 => "sw",
                    _ => "store_unknown",
                };
                format!("{} {}, {}({})", op, reg_name(rs2), imm, reg_name(rs1))
            }
            // OP-IMM
            0x13 => {
                let imm = (inst as i32) >> 20;
                let shamt = rs2;
                match funct3 {
                    0x0 => {
                        if inst == 0x00000013 {
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
                        if funct7 == 0x20 {
                            format!("srai {}, {}, {}", reg_name(rd), reg_name(rs1), shamt)
                        } else {
                            format!("srli {}, {}, {}", reg_name(rd), reg_name(rs1), shamt)
                        }
                    }
                    _ => format!("op_imm_unknown 0x{:08x}", inst),
                }
            }
            // OP
            0x33 => {
                let op = match (funct7, funct3) {
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
            0x07 => {
                let imm = (inst as i32) >> 20;
                format!("flw {}, {}({})", freg_name(rd), imm, reg_name(rs1))
            }
            0x27 => {
                let imm5 = (inst >> 7) & 0x1f;
                let imm7 = (inst >> 25) & 0x7f;
                let imm = ((((imm7 << 5) | imm5) as i32) << 20) >> 20;
                format!("fsw {}, {}({})", freg_name(rs2), imm, reg_name(rs1))
            }
            // SYSTEM
            0x73 => {
                if inst == 0x00000073 {
                    "ecall".to_string()
                } else if inst == 0x00100073 {
                    "ebreak".to_string()
                } else if inst == 0x30200073 {
                    "mret".to_string()
                } else {
                    let csr = inst >> 20;
                    let op = match funct3 {
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
            _ => format!("unimpl 0x{:08x}", inst),
        }
    }

    fn decode_rvc(_address: u32, inst: u16) -> String {
        let op = inst & 0x3;
        let funct3 = (inst >> 13) & 0x7;
        match (op, funct3) {
            (0x0, 0x0) => "c.addi4spn".to_string(),
            (0x0, 0x2) => "c.lw".to_string(),
            (0x0, 0x6) => "c.sw".to_string(),
            (0x1, 0x0) => {
                if inst == 0x0001 {
                    "c.nop".to_string()
                } else {
                    "c.addi".to_string()
                }
            }
            (0x1, 0x1) => "c.jal".to_string(),
            (0x1, 0x2) => "c.li".to_string(),
            (0x1, 0x3) => "c.lui/c.addi16sp".to_string(),
            (0x1, 0x5) => "c.j".to_string(),
            (0x1, 0x6) => "c.beqz".to_string(),
            (0x1, 0x7) => "c.bnez".to_string(),
            (0x2, 0x0) => "c.slli".to_string(),
            (0x2, 0x2) => "c.lwsp".to_string(),
            (0x2, 0x4) => "c.mv/c.add/c.jr/c.jalr".to_string(),
            (0x2, 0x6) => "c.swsp".to_string(),
            _ => format!("c.unimpl 0x{:04x}", inst),
        }
    }
}
