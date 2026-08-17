use riscv_rs::{Cpu, Memory, MemoryOps};

fn encode_r(opcode: u32, rd: usize, funct3: u32, rs1: usize, rs2: usize, funct7: u32) -> u32 {
    (funct7 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

fn encode_i(opcode: u32, rd: usize, funct3: u32, rs1: usize, imm: i32) -> u32 {
    (((imm as u32) & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

fn encode_s(opcode: u32, funct3: u32, rs1: usize, rs2: usize, imm: i32) -> u32 {
    let imm_u = imm as u32;
    (((imm_u >> 5) & 0x7F) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((imm_u & 0x1F) << 7)
        | opcode
}

fn encode_b(opcode: u32, funct3: u32, rs1: usize, rs2: usize, imm: i32) -> u32 {
    let imm_u = imm as u32;
    (((imm_u >> 12) & 1) << 31)
        | (((imm_u >> 5) & 0x3F) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | (((imm_u >> 1) & 0xF) << 8)
        | (((imm_u >> 11) & 1) << 7)
        | opcode
}

fn encode_u(opcode: u32, rd: usize, imm: u32) -> u32 {
    (imm & 0xFFFFF000) | ((rd as u32) << 7) | opcode
}

fn encode_j(opcode: u32, rd: usize, imm: i32) -> u32 {
    let imm_u = imm as u32;
    (((imm_u >> 20) & 1) << 31)
        | (((imm_u >> 1) & 0x3FF) << 21)
        | (((imm_u >> 11) & 1) << 20)
        | (((imm_u >> 12) & 0xFF) << 12)
        | ((rd as u32) << 7)
        | opcode
}

fn encode_r4(
    opcode: u32,
    rd: usize,
    funct3: u32,
    rs1: usize,
    rs2: usize,
    rs3: usize,
    fmt: u32,
) -> u32 {
    ((rs3 as u32) << 27)
        | (fmt << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

// ---------------------------------------------------------------------------
// 1. RV32I Base Integer Tests
// ---------------------------------------------------------------------------

#[test]
fn test_rv32i_arithmetic_logic_r_type() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    cpu.write_reg(1, 40);
    cpu.write_reg(2, 2);
    cpu.write_reg(3, 0xFFFFFFFF); // -1

    // ADD x4 = x1 + x2 (42)
    let inst = encode_r(0x33, 4, 0, 1, 2, 0x00);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(4), 42);

    // SUB x5 = x1 - x2 (38)
    let inst = encode_r(0x33, 5, 0, 1, 2, 0x20);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(5), 38);

    // SLL x6 = x1 << x2 (160)
    let inst = encode_r(0x33, 6, 1, 1, 2, 0x00);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(6), 160);

    // SLT x7 = (x3 < x1) -> -1 < 40 = 1
    let inst = encode_r(0x33, 7, 2, 3, 1, 0x00);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(7), 1);

    // SLTU x8 = (x3 <u x1) -> 0xFFFFFFFF < 40 = 0
    let inst = encode_r(0x33, 8, 3, 3, 1, 0x00);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 0);

    // XOR x9 = x1 ^ x2
    let inst = encode_r(0x33, 9, 4, 1, 2, 0x00);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(9), 40 ^ 2);

    // SRL x10 = x3 >> 1 (logical)
    let inst = encode_r(0x33, 10, 5, 3, 2, 0x00);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(10), 0x3FFFFFFF);

    // SRA x11 = x3 >> 1 (arithmetic sign extension)
    let inst = encode_r(0x33, 11, 5, 3, 2, 0x20);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(11), 0xFFFFFFFF);

    // OR x12 = x1 | x2
    let inst = encode_r(0x33, 12, 6, 1, 2, 0x00);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(12), 40 | 2);

    // AND x13 = x1 & x2
    let inst = encode_r(0x33, 13, 7, 1, 2, 0x00);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(13), 40 & 2);

    // Test writing to x0 remains 0
    let inst = encode_r(0x33, 0, 0, 1, 2, 0x00);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(0), 0);
}

#[test]
fn test_rv32i_immediate_instructions() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    cpu.write_reg(1, 100);

    // ADDI x2 = x1 + (-50) -> 50
    let inst = encode_i(0x13, 2, 0, 1, -50);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(2), 50);

    // SLTI x3 = (x1 < -50) -> 0
    let inst = encode_i(0x13, 3, 2, 1, -50);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(3), 0);

    // SLTIU x4 = (x1 <u 200) -> 1
    let inst = encode_i(0x13, 4, 3, 1, 200);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(4), 1);

    // XORI, ORI, ANDI
    let inst = encode_i(0x13, 5, 4, 1, 0x0F);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(5), 100 ^ 0x0F);

    let inst = encode_i(0x13, 6, 6, 1, 0x0F);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(6), 100 | 0x0F);

    let inst = encode_i(0x13, 7, 7, 1, 0x0F);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(7), 100 & 0x0F);

    // SLLI, SRLI, SRAI
    let inst = encode_i(0x13, 8, 1, 1, 4); // SLLI by 4
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 1600);

    cpu.write_reg(9, 0x80000000);
    let inst = encode_i(0x13, 10, 5, 9, 4); // SRLI
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(10), 0x08000000);

    let inst = encode_i(0x13, 11, 5, 9, 4 | 0x400); // SRAI (funct7 bit 30 set)
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(11), 0xF8000000);
}

#[test]
fn test_rv32i_upper_immediate() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    cpu.pc = 0x1000;

    // LUI x1, 0x12345000
    let inst = encode_u(0x37, 1, 0x12345000);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(1), 0x12345000);

    // AUIPC x2, 0x10000000 (with PC = 0x1004)
    let inst = encode_u(0x17, 2, 0x10000000);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(2), 0x10001004);
}

#[test]
fn test_rv32i_jumps_and_branches() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    cpu.pc = 0x1000;

    // JAL x1, offset 16
    let inst = encode_j(0x6F, 1, 16);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(1), 0x1004);
    assert_eq!(cpu.pc, 0x1010);

    // JALR x2, x1, offset 4 -> target (0x1004 + 4) & ~1 = 0x1008
    let inst = encode_i(0x67, 2, 0, 1, 4);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(2), 0x1014);
    assert_eq!(cpu.pc, 0x1008);

    // Branches
    cpu.write_reg(10, 50);
    cpu.write_reg(11, 50);
    cpu.write_reg(12, 100);

    // BEQ taken
    cpu.pc = 0x2000;
    let inst = encode_b(0x63, 0, 10, 11, 12);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x200C);

    // BNE taken
    cpu.pc = 0x2000;
    let inst = encode_b(0x63, 1, 10, 12, 8);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x2008);

    // BLT taken (50 < 100)
    cpu.pc = 0x2000;
    let inst = encode_b(0x63, 4, 10, 12, 16);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x2010);

    // BGE taken (100 >= 50)
    cpu.pc = 0x2000;
    let inst = encode_b(0x63, 5, 12, 10, 20);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x2014);

    // BLTU taken
    cpu.pc = 0x2000;
    let inst = encode_b(0x63, 6, 10, 12, 4);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x2004);

    // BGEU taken
    cpu.pc = 0x2000;
    let inst = encode_b(0x63, 7, 12, 10, 8);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x2008);
}

#[test]
fn test_rv32i_loads_and_stores() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    cpu.write_reg(1, 0x1000); // base address
    cpu.write_reg(2, 0x12345678);

    // Store Byte
    let inst = encode_s(0x23, 0, 1, 2, 0); // SB
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(mem.read_u8(0x1000), 0x78);

    // Store Halfword
    let inst = encode_s(0x23, 1, 1, 2, 2); // SH at offset 2
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(mem.read_u16(0x1002), 0x5678);

    // Store Word
    let inst = encode_s(0x23, 2, 1, 2, 4); // SW at offset 4
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(mem.read_u32(0x1004), 0x12345678);

    // Load Byte Signed / Unsigned
    mem.write_u8(0x2000, 0x80);
    cpu.write_reg(3, 0x2000);

    let inst = encode_i(0x03, 4, 0, 3, 0); // LB
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(4), 0xFFFFFF80);

    let inst = encode_i(0x03, 5, 4, 3, 0); // LBU
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(5), 0x00000080);

    // Load Halfword Signed / Unsigned
    mem.write_u16(0x2004, 0x8000);
    let inst = encode_i(0x03, 6, 1, 3, 4); // LH
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(6), 0xFFFF8000);

    let inst = encode_i(0x03, 7, 5, 3, 4); // LHU
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(7), 0x00008000);

    // Load Word
    let inst = encode_i(0x03, 8, 2, 1, 4); // LW at 0x1004
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 0x12345678);
}

// ---------------------------------------------------------------------------
// 2. RV32M Extension Tests
// ---------------------------------------------------------------------------

#[test]
fn test_rv32m_multiply_divide() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    cpu.write_reg(1, 0xFFFFFFFF); // -1 (signed) / 4294967295 (unsigned)
    cpu.write_reg(2, 5);

    // MUL: -1 * 5 = -5
    let inst = encode_r(0x33, 3, 0, 1, 2, 0x01);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(3) as i32, -5);

    // MULH: signed(-1) * signed(5) = -5 -> high 32 bits is -1 (0xFFFFFFFF)
    let inst = encode_r(0x33, 4, 1, 1, 2, 0x01);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(4), 0xFFFFFFFF);

    // MULHSU: signed(-1) * unsigned(5) = -5 -> high 32 bits is -1 (0xFFFFFFFF)
    let inst = encode_r(0x33, 5, 2, 1, 2, 0x01);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(5), 0xFFFFFFFF);

    // MULHU: unsigned(4294967295) * unsigned(5) -> high 32 bits is 4
    let inst = encode_r(0x33, 6, 3, 1, 2, 0x01);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(6), 4);

    // DIV / DIVU / REM / REMU
    cpu.write_reg(10, 20);
    cpu.write_reg(11, 3);

    // DIV 20 / 3 = 6
    let inst = encode_r(0x33, 12, 4, 10, 11, 0x01);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(12), 6);

    // REM 20 % 3 = 2
    let inst = encode_r(0x33, 13, 6, 10, 11, 0x01);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(13), 2);

    // Division by zero behavior according to RISC-V spec:
    cpu.write_reg(14, 0);

    // DIV by 0 returns -1 (0xFFFFFFFF)
    let inst = encode_r(0x33, 15, 4, 10, 14, 0x01);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(15), 0xFFFFFFFF);

    // DIVU by 0 returns 0xFFFFFFFF
    let inst = encode_r(0x33, 16, 5, 10, 14, 0x01);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(16), 0xFFFFFFFF);

    // REM by 0 returns dividend (20)
    let inst = encode_r(0x33, 17, 6, 10, 14, 0x01);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(17), 20);
}

// ---------------------------------------------------------------------------
// 3. RV32A Extension Tests
// ---------------------------------------------------------------------------

#[test]
fn test_rv32a_atomics() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    let addr = 0x3000;
    mem.write_u32(addr, 100);
    cpu.write_reg(1, addr);
    cpu.write_reg(2, 50);

    // LR.W x3, (x1)
    let inst = encode_r(0x2F, 3, 2, 1, 0, 0x02 << 2);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(3), 100);

    // SC.W x4, x2, (x1) -> x4 = 0 (success flag), mem[addr] = 50
    let inst = encode_r(0x2F, 4, 2, 1, 2, 0x03 << 2);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(4), 0);
    assert_eq!(mem.read_u32(addr), 50);

    // AMOSWAP.W x5, x2, (x1) -> x5 = 50, mem[addr] = 50
    cpu.write_reg(2, 200);
    let inst = encode_r(0x2F, 5, 2, 1, 2, 0x01 << 2);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(5), 50);
    assert_eq!(mem.read_u32(addr), 200);

    // AMOADD.W x6, x2 (50), (x1) -> x6 = 200, mem[addr] = 250
    cpu.write_reg(2, 50);
    let inst = encode_r(0x2F, 6, 2, 1, 2, 0x00 << 2);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(6), 200);
    assert_eq!(mem.read_u32(addr), 250);

    // AMOAND.W, AMOOR.W, AMOXOR.W
    cpu.write_reg(2, 0x0F);
    let inst = encode_r(0x2F, 7, 2, 1, 2, 0x0C << 2); // AMOAND
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(7), 250);
    assert_eq!(mem.read_u32(addr), 250 & 0x0F);

    let inst = encode_r(0x2F, 8, 2, 1, 2, 0x08 << 2); // AMOOR
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(mem.read_u32(addr), (250 & 0x0F) | 0x0F);

    // AMOMIN.W / AMOMAX.W
    mem.write_u32(addr, 0xFFFFFFFF); // -1 signed
    cpu.write_reg(2, 10); // 10 signed

    let inst = encode_r(0x2F, 9, 2, 1, 2, 0x10 << 2); // AMOMIN (signed) -> -1 < 10 -> -1
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(9), 0xFFFFFFFF);
    assert_eq!(mem.read_u32(addr), 0xFFFFFFFF);

    let inst = encode_r(0x2F, 10, 2, 1, 2, 0x1C << 2); // AMOMAXU (unsigned) -> 0xFFFFFFFF > 10 -> 0xFFFFFFFF
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(mem.read_u32(addr), 0xFFFFFFFF);
}

// ---------------------------------------------------------------------------
// 4. RV32F & RV32D Floating-Point Tests
// ---------------------------------------------------------------------------

#[test]
fn test_rv32f_single_precision() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // FLW f1 from memory
    let val_f32: f32 = 3.5;
    mem.write_u32(0x4000, val_f32.to_bits());
    cpu.write_reg(1, 0x4000);

    let inst = encode_i(0x07, 1, 2, 1, 0); // FLW f1, 0(x1)
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(1), 3.5);

    // FADD.S f3 = f1 (3.5) + f2 (1.5)
    cpu.write_f32(2, 1.5);
    let inst = encode_r(0x53, 3, 0, 1, 2, 0x00); // FADD.S
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(3), 5.0);

    // FSUB.S f4 = f1 (3.5) - f2 (1.5) = 2.0
    let inst = encode_r(0x53, 4, 0, 1, 2, 0x04); // FSUB.S
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(4), 2.0);

    // FMUL.S f5 = f1 (3.5) * f2 (1.5) = 5.25
    let inst = encode_r(0x53, 5, 0, 1, 2, 0x08); // FMUL.S
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(5), 5.25);

    // FDIV.S f6 = f3 (5.0) / f4 (2.0) = 2.5
    let inst = encode_r(0x53, 6, 0, 3, 4, 0x0C); // FDIV.S
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(6), 2.5);

    // FSQRT.S f7 = sqrt(f5 (5.25) -> f4 (4.0)) -> 2.0
    cpu.write_f32(8, 4.0);
    let inst = encode_r(0x53, 7, 0, 8, 0, 0x2C); // FSQRT.S
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(7), 2.0);

    // FSW f3 (5.0) to memory
    let inst = encode_s(0x27, 2, 1, 3, 4); // FSW f3 at offset 4
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(f32::from_bits(mem.read_u32(0x4004)), 5.0);

    // FMV.X.W x10, f3
    let inst = encode_r(0x53, 10, 0, 3, 0, 0x70);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(10), (5.0f32).to_bits());

    // FMV.W.X f9, x10
    let inst = encode_r(0x53, 9, 0, 10, 0, 0x78);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(9), 5.0);

    // FCVT.W.S x11, f3 (5.0) -> 5
    let inst = encode_r(0x53, 11, 0, 3, 0, 0x60);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(11), 5);

    // FEQ.S, FLT.S, FLE.S
    let inst = encode_r(0x53, 12, 2, 1, 2, 0x50); // FEQ.S f1(3.5), f2(1.5) -> 0
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(12), 0);

    let inst = encode_r(0x53, 13, 1, 2, 1, 0x50); // FLT.S f2(1.5) < f1(3.5) -> 1
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(13), 1);
}

#[test]
fn test_rv32d_double_precision() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // FLD f1 from memory
    let val_f64: f64 = 123.456789;
    let bits = val_f64.to_bits();
    mem.write_u32(0x5000, bits as u32);
    mem.write_u32(0x5004, (bits >> 32) as u32);
    cpu.write_reg(1, 0x5000);

    let inst = encode_i(0x07, 1, 3, 1, 0); // FLD f1, 0(x1)
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.fregs[1], 123.456789);

    // FADD.D f3 = f1 + f2 (10.0)
    cpu.fregs[2] = 10.0;
    let inst = encode_r(0x53, 3, 0, 1, 2, 0x01); // FADD.D
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert!((cpu.fregs[3] - 133.456789).abs() < 1e-9);

    // FSUB.D f4 = f3 - f2 = 123.456789
    let inst = encode_r(0x53, 4, 0, 3, 2, 0x05); // FSUB.D
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert!((cpu.fregs[4] - 123.456789).abs() < 1e-9);

    // FMUL.D, FDIV.D
    cpu.fregs[10] = 6.0;
    cpu.fregs[11] = 3.0;

    let inst = encode_r(0x53, 12, 0, 10, 11, 0x09); // FMUL.D
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.fregs[12], 18.0);

    let inst = encode_r(0x53, 13, 0, 10, 11, 0x0D); // FDIV.D
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.fregs[13], 2.0);

    // FSD f3 to memory
    let inst = encode_s(0x27, 3, 1, 3, 8); // FSD f3 at offset 8
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    let low = mem.read_u32(0x5008) as u64;
    let high = mem.read_u32(0x500C) as u64;
    assert!((f64::from_bits(low | (high << 32)) - 133.456789).abs() < 1e-9);
}

#[test]
fn test_rv32f_rv32d_fused_multiply_add() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    cpu.write_f32(1, 2.0);
    cpu.write_f32(2, 3.0);
    cpu.write_f32(3, 4.0);

    // FMADD.S f4 = (2.0 * 3.0) + 4.0 = 10.0
    let inst = encode_r4(0x43, 4, 0, 1, 2, 3, 0); // fmt=0 (.s)
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(4), 10.0);

    // FMSUB.S f5 = (2.0 * 3.0) - 4.0 = 2.0
    let inst = encode_r4(0x47, 5, 0, 1, 2, 3, 0);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(5), 2.0);

    // FMADD.D f6 = (2.0 * 3.0) + 4.0 = 10.0
    cpu.write_f64(1, 2.0);
    cpu.write_f64(2, 3.0);
    cpu.write_f64(3, 4.0);
    let inst = encode_r4(0x43, 6, 0, 1, 2, 3, 1); // fmt=1 (.d)
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.fregs[6], 10.0);
}

// ---------------------------------------------------------------------------
// 5. CSR & Privilege System Tests
// ---------------------------------------------------------------------------

#[test]
fn test_csr_instructions_and_mret() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    cpu.write_reg(1, 0b1010);

    // CSRRW x2, 0x300 (mstatus), x1 (write 0b1010, return old 0)
    let inst = encode_i(0x73, 2, 1, 1, 0x300);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(2), 0);
    assert_eq!(*cpu.csrs.get(&0x300).unwrap(), 0b1010);

    // CSRRS x3, 0x300, x1 with x1 = 0b0100 -> set bits
    cpu.write_reg(1, 0b0100);
    let inst = encode_i(0x73, 3, 2, 1, 0x300);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(3), 0b1010);
    assert_eq!(*cpu.csrs.get(&0x300).unwrap(), 0b1110);

    // CSRRC x4, 0x300, x1 with x1 = 0b0010 -> clear bit 1
    cpu.write_reg(1, 0b0010);
    let inst = encode_i(0x73, 4, 3, 1, 0x300);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(4), 0b1110);
    assert_eq!(*cpu.csrs.get(&0x300).unwrap(), 0b1100);

    // MRET instruction (0x30200073) restores PC from mepc (0x341)
    cpu.csrs.insert(0x341, 0x8000);
    let inst_mret = 0x30200073;
    assert!(cpu.execute_inst32(inst_mret, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x8000);
}

// ---------------------------------------------------------------------------
// 6. RV32C Compressed Instructions Tests
// ---------------------------------------------------------------------------

#[test]
fn test_rv32c_compressed_instructions() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // C.LI x1, 15 -> opcode 01, funct3 010 (inst: 0x403D)
    // format: [15:13=010][12=imm5][11:7=rd][6:2=imm4-0][1:0=01]
    // C.LI rd=1, imm=15 -> 010_0_00001_01111_01 = 0x40BD
    let c_li = 0x40BDu16;
    cpu.pc = 0x1000;
    assert!(cpu.execute_inst16(c_li, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(1), 15);
    assert_eq!(cpu.pc, 0x1002);

    // C.MV x2, x1 (C.MV has op=2, funct3=4, bit12=0) -> 0x8106 (x2 = 15)
    let c_mv = 0x8106u16;
    assert!(cpu.execute_inst16(c_mv, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(2), 15);

    // C.ADD x2, x1 -> 15 + 15 = 30 (op=2, funct3=4, bit12=1) -> 0x9106
    let c_add = 0x9106u16;
    assert!(cpu.execute_inst16(c_add, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(2), 30);

    // C.J offset 4 (op=1, funct3=5) -> 0xA011
    cpu.pc = 0x2000;
    let c_j = 0xA011u16;
    assert!(cpu.execute_inst16(c_j, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x2004);

    // C.BEQZ (op=1, funct3=6) with rs1c=x8 (index 0 for compressed 8-15)
    cpu.write_reg(8, 0);
    let c_beqz = 0xC001u16; // C.BEQZ x8, offset 0
    assert!(cpu.execute_inst16(c_beqz, &mut mem).is_ok());

    // C.SLLI x1, shamt 2 (op=2, funct3=0)
    cpu.write_reg(1, 5);
    let c_slli = 0x008A; // C.SLLI x1, 2
    assert!(cpu.execute_inst16(c_slli, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(1), 20);
}

#[test]
fn test_fp_sign_injection_min_max_class_conversions() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // Sign Injection .S
    cpu.write_f32(1, 2.5); // positive
    cpu.write_f32(2, -1.0); // negative

    // FSGNJ.S f3 = mag(f1), sign(f2) -> -2.5
    let inst = encode_r(0x53, 3, 0, 1, 2, 0x10);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(3), -2.5);

    // FSGNJN.S f4 = mag(f1), -sign(f2) -> +2.5
    let inst = encode_r(0x53, 4, 1, 1, 2, 0x10);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(4), 2.5);

    // FSGNJX.S f5 = mag(f1), sign(f1)^sign(f2) -> -2.5
    let inst = encode_r(0x53, 5, 2, 1, 2, 0x10);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(5), -2.5);

    // FMIN.S / FMAX.S
    let inst = encode_r(0x53, 6, 0, 1, 2, 0x14); // FMIN.S (2.5 vs -1.0) -> -1.0
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(6), -1.0);

    let inst = encode_r(0x53, 7, 1, 1, 2, 0x14); // FMAX.S (2.5 vs -1.0) -> 2.5
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(7), 2.5);

    // FCLASS.S
    let inst = encode_r(0x53, 8, 1, 1, 0, 0x70); // FCLASS.S f1 (2.5, positive normal) -> bit 6 = 1 << 6 = 64
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 64);

    // FCVT.S.W f9, x10 (100)
    cpu.write_reg(10, 100);
    let inst = encode_r(0x53, 9, 0, 10, 0, 0x68);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(9), 100.0);

    // FCVT.D.S f10, f9 (100.0) -> 100.0
    let inst = encode_r(0x53, 10, 0, 9, 0, 0x21);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f64(10), 100.0);

    // FCVT.S.D f11, f10 (100.0) -> 100.0
    let inst = encode_r(0x53, 11, 0, 10, 1, 0x20);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(11), 100.0);
}

#[test]
fn test_csr_immediate_instructions() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // CSRRWI x1, 0x300 (mstatus), zimm 5 -> sets mstatus to 5
    let inst = encode_i(0x73, 1, 5, 5, 0x300);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(*cpu.csrs.get(&0x300).unwrap(), 5);

    // CSRRSI x2, 0x300, zimm 8 -> sets bit 3 (0b0101 | 0b1000 = 0b1101 = 13)
    let inst = encode_i(0x73, 2, 6, 8, 0x300);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(2), 5);
    assert_eq!(*cpu.csrs.get(&0x300).unwrap(), 13);

    // CSRRCI x3, 0x300, zimm 4 -> clears bit 2 (0b1101 & ~0b0100 = 0b1001 = 9)
    let inst = encode_i(0x73, 3, 7, 4, 0x300);
    assert!(cpu.execute_inst32(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(3), 13);
    assert_eq!(*cpu.csrs.get(&0x300).unwrap(), 9);
}

// ---------------------------------------------------------------------------
// 7. RV32C Decoder Coverage (offsets, quadrant 1 funct3 4, compressed FP,
//    c.ebreak and reserved encodings)
// ---------------------------------------------------------------------------

/// Assemble a 16-bit compressed instruction. Each field is `(value, hi, lo)`
/// where `hi`/`lo` are inclusive instruction bit positions.
fn rvc(op: u16, funct3: u16, fields: &[(u16, u32, u32)]) -> u16 {
    let mut inst = op | (funct3 << 13);
    for &(val, hi, lo) in fields {
        let width = hi - lo + 1;
        let mask = ((1u32 << width) - 1) as u16;
        inst |= (val & mask) << lo;
    }
    inst
}

#[test]
fn test_rvc_addi16sp_scales_immediate_once() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // c.addi16sp sp, -16. nzimm = -16 => 10-bit field 0b11_1111_0000, so
    // nzimm[9]=1, nzimm[8:7]=11, nzimm[6]=1, nzimm[5]=1, nzimm[4]=1.
    let inst = rvc(
        1,
        3,
        &[
            (1, 12, 12),
            (2, 11, 7),
            (1, 6, 6),
            (1, 5, 5),
            (3, 4, 3),
            (1, 2, 2),
        ],
    );
    cpu.write_reg(2, 0x2000);
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(2), 0x2000 - 16);

    // c.addi16sp sp, 32: nzimm[5] = 1 only.
    let inst = rvc(1, 3, &[(2, 11, 7), (1, 2, 2)]);
    cpu.write_reg(2, 0x2000);
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(2), 0x2020);
}

#[test]
fn test_rvc_memory_offsets() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // c.lw x8, 4(x8): uimm[2] = inst[6].
    cpu.write_reg(8, 0x1000);
    mem.write_u32(0x1004, 0xDEADBEEF);
    let inst = rvc(
        0,
        2,
        &[(0, 12, 10), (0, 9, 7), (1, 6, 6), (0, 5, 5), (0, 4, 2)],
    );
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 0xDEADBEEF);

    // c.sw x9, 64(x8): uimm[6] = inst[5].
    cpu.write_reg(8, 0x1000);
    cpu.write_reg(9, 0x11223344);
    let inst = rvc(
        0,
        6,
        &[(0, 12, 10), (0, 9, 7), (0, 6, 6), (1, 5, 5), (1, 4, 2)],
    );
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(mem.read_u32(0x1040), 0x11223344);

    // c.lwsp x9, 4(sp): uimm[4:2] = inst[6:4].
    cpu.write_reg(2, 0x3000);
    mem.write_u32(0x3004, 0x55667788);
    let inst = rvc(2, 2, &[(0, 12, 12), (9, 11, 7), (1, 6, 4), (0, 3, 2)]);
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(9), 0x55667788);

    // c.lwsp x9, 192(sp): uimm[7:6] = inst[3:2].
    cpu.write_reg(2, 0x3000);
    mem.write_u32(0x30C0, 0x99AABBCC);
    let inst = rvc(2, 2, &[(0, 12, 12), (9, 11, 7), (0, 6, 4), (3, 3, 2)]);
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(9), 0x99AABBCC);

    // c.swsp x8, 4(sp): uimm[5:2] = inst[12:9].
    cpu.write_reg(2, 0x4000);
    cpu.write_reg(8, 0xCAFEBABE);
    let inst = rvc(2, 6, &[(1, 12, 9), (0, 8, 7), (8, 6, 2)]);
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(mem.read_u32(0x4004), 0xCAFEBABE);
}

#[test]
fn test_rvc_quadrant1_funct3_4_group() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // c.srli x8, 3
    cpu.write_reg(8, 0x80);
    let inst = rvc(1, 4, &[(0, 12, 12), (0, 11, 10), (0, 9, 7), (3, 6, 2)]);
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 0x10);

    // c.srai x8, 4 (arithmetic: sign is preserved)
    cpu.write_reg(8, 0xFFFFFF00);
    let inst = rvc(1, 4, &[(0, 12, 12), (1, 11, 10), (0, 9, 7), (4, 6, 2)]);
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 0xFFFFFFF0);

    // c.andi x8, -8
    cpu.write_reg(8, 0xFF);
    let inst = rvc(1, 4, &[(1, 12, 12), (2, 11, 10), (0, 9, 7), (0x18, 6, 2)]);
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 0xF8);

    // c.sub x8, x9
    cpu.write_reg(8, 100);
    cpu.write_reg(9, 30);
    let inst = rvc(
        1,
        4,
        &[(0, 12, 12), (3, 11, 10), (0, 9, 7), (0, 6, 5), (1, 4, 2)],
    );
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 70);

    // c.xor x8, x9
    cpu.write_reg(8, 0b1100);
    cpu.write_reg(9, 0b1010);
    let inst = rvc(
        1,
        4,
        &[(0, 12, 12), (3, 11, 10), (0, 9, 7), (1, 6, 5), (1, 4, 2)],
    );
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 0b0110);

    // c.or x8, x9
    cpu.write_reg(8, 0b1100);
    let inst = rvc(
        1,
        4,
        &[(0, 12, 12), (3, 11, 10), (0, 9, 7), (2, 6, 5), (1, 4, 2)],
    );
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 0b1110);

    // c.and x8, x9
    cpu.write_reg(8, 0b1100);
    let inst = rvc(
        1,
        4,
        &[(0, 12, 12), (3, 11, 10), (0, 9, 7), (3, 6, 5), (1, 4, 2)],
    );
    assert!(cpu.execute_inst16(inst, &mut mem).is_ok());
    assert_eq!(cpu.read_reg(8), 0b1000);
}

#[test]
fn test_rvc_compressed_fp_memory() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // c.fsd f8, 8(x8) then c.fld f9, 8(x8)
    cpu.write_reg(8, 0x1000);
    cpu.write_f64(8, 12.5);
    let fsd = rvc(0, 5, &[(1, 12, 10), (0, 9, 7), (0, 6, 5), (0, 4, 2)]);
    assert!(cpu.execute_inst16(fsd, &mut mem).is_ok());
    assert_eq!(mem.read_u32(0x1008), 12.5f64.to_bits() as u32);
    assert_eq!(mem.read_u32(0x100C), (12.5f64.to_bits() >> 32) as u32);

    let fld = rvc(0, 1, &[(1, 12, 10), (0, 9, 7), (0, 6, 5), (1, 4, 2)]);
    assert!(cpu.execute_inst16(fld, &mut mem).is_ok());
    assert_eq!(cpu.read_f64(9), 12.5);

    // c.fsw f8, 4(x8) then c.flw f9, 4(x8)
    cpu.write_f32(8, -2.25);
    let fsw = rvc(
        0,
        7,
        &[(0, 12, 10), (0, 9, 7), (1, 6, 6), (0, 5, 5), (0, 4, 2)],
    );
    assert!(cpu.execute_inst16(fsw, &mut mem).is_ok());
    assert_eq!(mem.read_u32(0x1004), (-2.25f32).to_bits());

    let flw = rvc(
        0,
        3,
        &[(0, 12, 10), (0, 9, 7), (1, 6, 6), (0, 5, 5), (1, 4, 2)],
    );
    assert!(cpu.execute_inst16(flw, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(9), -2.25);
}

#[test]
fn test_rvc_compressed_fp_stack_memory() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    cpu.write_reg(2, 0x2000);

    // c.fsdsp f5, 8(sp) then c.fldsp f6, 8(sp)
    cpu.write_f64(5, -3.75);
    let fsdsp = rvc(2, 5, &[(1, 12, 10), (0, 9, 7), (5, 6, 2)]);
    assert!(cpu.execute_inst16(fsdsp, &mut mem).is_ok());
    assert_eq!(mem.read_u32(0x2008), (-3.75f64).to_bits() as u32);

    let fldsp = rvc(2, 1, &[(0, 12, 12), (6, 11, 7), (1, 6, 5), (0, 4, 2)]);
    assert!(cpu.execute_inst16(fldsp, &mut mem).is_ok());
    assert_eq!(cpu.read_f64(6), -3.75);

    // c.fswsp f5, 4(sp) then c.flwsp f7, 4(sp)
    cpu.write_f32(5, 6.5);
    let fswsp = rvc(2, 7, &[(1, 12, 9), (0, 8, 7), (5, 6, 2)]);
    assert!(cpu.execute_inst16(fswsp, &mut mem).is_ok());
    assert_eq!(mem.read_u32(0x2004), 6.5f32.to_bits());

    let flwsp = rvc(2, 3, &[(0, 12, 12), (7, 11, 7), (1, 6, 4), (0, 3, 2)]);
    assert!(cpu.execute_inst16(flwsp, &mut mem).is_ok());
    assert_eq!(cpu.read_f32(7), 6.5);
}

#[test]
fn test_rvc_ebreak_halts() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    let c_ebreak = rvc(2, 4, &[(1, 12, 12), (0, 11, 7), (0, 6, 2)]);
    assert_eq!(c_ebreak, 0x9002);
    assert!(cpu.execute_inst16(c_ebreak, &mut mem).is_ok());
    assert!(cpu.is_halted);
}

#[test]
fn test_rvc_reserved_encodings_trap() {
    let mut mem = Memory::new();

    let reserved: [(&str, u16); 9] = [
        // c.addi4spn with a zero immediate (and the all-zero halfword).
        ("c.addi4spn imm=0", rvc(0, 0, &[(1, 4, 2)])),
        // Quadrant 1 funct3 3 with rd = x0.
        (
            "c.lui/c.addi16sp rd=0",
            rvc(1, 3, &[(1, 12, 12), (0, 11, 7), (1, 2, 2)]),
        ),
        // c.lui with a zero immediate.
        (
            "c.lui imm=0",
            rvc(1, 3, &[(0, 12, 12), (5, 11, 7), (0, 6, 2)]),
        ),
        // c.addi16sp with a zero immediate.
        (
            "c.addi16sp imm=0",
            rvc(1, 3, &[(0, 12, 12), (2, 11, 7), (0, 6, 2)]),
        ),
        // c.jr with rs1 = x0.
        (
            "c.jr rd=0",
            rvc(2, 4, &[(0, 12, 12), (0, 11, 7), (0, 6, 2)]),
        ),
        // RV32C shift encodings with shamt[5] set.
        (
            "c.srli shamt[5]=1",
            rvc(1, 4, &[(1, 12, 12), (0, 11, 10), (0, 9, 7), (1, 6, 2)]),
        ),
        (
            "c.slli shamt[5]=1",
            rvc(2, 0, &[(1, 12, 12), (8, 11, 7), (1, 6, 2)]),
        ),
        // c.subw / c.addw are RV64-only.
        (
            "c.subw",
            rvc(
                1,
                4,
                &[(1, 12, 12), (3, 11, 10), (0, 9, 7), (0, 6, 5), (1, 4, 2)],
            ),
        ),
        // c.lwsp with rd = x0.
        (
            "c.lwsp rd=0",
            rvc(2, 2, &[(0, 12, 12), (0, 11, 7), (1, 6, 4), (0, 3, 2)]),
        ),
    ];

    for (name, inst) in reserved {
        let mut cpu = Cpu::new();
        cpu.pc = 0x1000;
        let res = cpu.execute_inst16(inst, &mut mem);
        assert!(
            res.is_err(),
            "{} ({:#06x}) should trap, but it was accepted",
            name,
            inst
        );
        assert_eq!(
            cpu.pc, 0x1000,
            "{} ({:#06x}) advanced the PC instead of trapping",
            name, inst
        );
    }
}

// ---------------------------------------------------------------------------
// 8. Machine-Mode Trap State (mcause / mepc / mstatus)
// ---------------------------------------------------------------------------

const CSR_MSTATUS: u16 = 0x300;
const CSR_MTVEC: u16 = 0x305;
const CSR_MEPC: u16 = 0x341;
const CSR_MCAUSE: u16 = 0x342;
const MSTATUS_MIE: u32 = 1 << 3;
const MSTATUS_MPIE: u32 = 1 << 7;

#[test]
fn test_interrupt_records_cause_and_epc_separately() {
    let mut cpu = Cpu::new();
    cpu.csrs.insert(CSR_MTVEC, 0x8000);
    cpu.csrs.insert(CSR_MSTATUS, MSTATUS_MIE);
    cpu.pc = 0x1234;

    cpu.handle_interrupt(7);

    // mepc and mcause live in different registers, so the cause does not
    // overwrite the return address.
    assert_eq!(*cpu.csrs.get(&CSR_MEPC).unwrap(), 0x1234);
    assert_eq!(*cpu.csrs.get(&CSR_MCAUSE).unwrap(), 0x8000_0007);
    assert_eq!(cpu.pc, 0x8000);
}

#[test]
fn test_interrupt_disables_further_interrupts_until_mret() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    cpu.csrs.insert(CSR_MTVEC, 0x8000);
    cpu.csrs.insert(CSR_MSTATUS, MSTATUS_MIE);
    cpu.pc = 0x1234;

    assert!(cpu.interrupts_enabled());
    cpu.handle_interrupt(7);

    // Inside the handler MIE is clear and the previous value is kept in MPIE.
    let mstatus = *cpu.csrs.get(&CSR_MSTATUS).unwrap();
    assert_eq!(mstatus & MSTATUS_MIE, 0);
    assert_eq!(mstatus & MSTATUS_MPIE, MSTATUS_MPIE);
    assert!(!cpu.interrupts_enabled());

    // A second interrupt while the handler runs must not clobber mepc.
    // (`run()` gates delivery on `interrupts_enabled`.)
    assert!(!cpu.interrupts_enabled());
    assert_eq!(*cpu.csrs.get(&CSR_MEPC).unwrap(), 0x1234);

    // MRET returns to the interrupted PC and re-enables interrupts.
    cpu.pc = 0x8000;
    assert!(cpu.execute_inst32(0x30200073, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x1234);
    let mstatus = *cpu.csrs.get(&CSR_MSTATUS).unwrap();
    assert_eq!(mstatus & MSTATUS_MIE, MSTATUS_MIE);
    assert_eq!(mstatus & MSTATUS_MPIE, MSTATUS_MPIE);
    assert!(cpu.interrupts_enabled());
}

#[test]
fn test_mret_leaves_interrupts_disabled_when_mpie_is_clear() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    cpu.csrs.insert(CSR_MEPC, 0x2000);
    cpu.csrs.insert(CSR_MSTATUS, 0);

    assert!(cpu.execute_inst32(0x30200073, &mut mem).is_ok());
    assert_eq!(cpu.pc, 0x2000);
    assert!(!cpu.interrupts_enabled());
}

#[test]
fn test_interrupt_handler_returns_to_interrupted_instruction() {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    // Handler at 0x8000: addi x5, x5, 1 ; mret
    mem.write_u32(0x8000, encode_i(0x13, 5, 0, 5, 1));
    mem.write_u32(0x8004, 0x30200073);
    // Interrupted code at 0x1000: addi x6, x6, 1
    mem.write_u32(0x1000, encode_i(0x13, 6, 0, 6, 1));

    cpu.csrs.insert(CSR_MTVEC, 0x8000);
    cpu.csrs.insert(CSR_MSTATUS, MSTATUS_MIE);
    cpu.pc = 0x1000;

    cpu.handle_interrupt(11);
    // Run the handler to completion.
    while cpu.pc != 0x1000 {
        assert!(cpu.execute_inst32(mem.read_u32(cpu.pc), &mut mem).is_ok());
    }
    assert_eq!(cpu.read_reg(5), 1);

    // The interrupted instruction has not run yet, and now does.
    assert_eq!(cpu.read_reg(6), 0);
    assert!(cpu.execute_inst32(mem.read_u32(cpu.pc), &mut mem).is_ok());
    assert_eq!(cpu.read_reg(6), 1);
    assert_eq!(cpu.pc, 0x1004);
}
