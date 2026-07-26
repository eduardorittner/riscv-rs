use rust_whisper::{Cpu, Memory, MemoryOps};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x123456789ABCDEF0 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn range(&mut self, min: u32, max: u32) -> u32 {
        if min >= max {
            return min;
        }
        min + ((self.next_u64() as u32) % (max - min + 1))
    }
}

fn encode_r(opcode: u32, rd: usize, funct3: u32, rs1: usize, rs2: usize, funct7: u32) -> u32 {
    (funct7 << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | opcode
}

fn encode_i(opcode: u32, rd: usize, funct3: u32, rs1: usize, imm: i32) -> u32 {
    (((imm as u32) & 0xFFF) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | opcode
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

fn encode_u(opcode: u32, rd: usize, imm: u32) -> u32 {
    (imm & 0xFFFFF000) | ((rd as u32) << 7) | opcode
}

fn encode_r4(opcode: u32, rd: usize, funct3: u32, rs1: usize, rs2: usize, rs3: usize, fmt: u32) -> u32 {
    ((rs3 as u32) << 27)
        | (fmt << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7) | opcode
}

/// Generates a list of valid, non-trapping RV32IMAFD instructions.
fn generate_random_instructions(count: usize, rng: &mut SimpleRng) -> Vec<u32> {
    let mut insts = Vec::with_capacity(count);

    for _ in 0..count {
        let cat = rng.range(0, 7);
        let mut rd = rng.range(1, 31) as usize;
        if rd == 10 {
            rd = 11;
        }
        let rs1 = rng.range(1, 31) as usize;
        let rs2 = rng.range(1, 31) as usize;
        let _rs3 = rng.range(1, 31) as usize;
        let frd = rng.range(0, 31) as usize;
        let frs1 = rng.range(0, 31) as usize;
        let frs2 = rng.range(0, 31) as usize;
        let frs3 = rng.range(0, 31) as usize;

        let inst = match cat {
            0 => {
                // OP Integer R-Type
                let funct3_ops = [
                    (0, 0x00), // ADD
                    (0, 0x20), // SUB
                    (1, 0x00), // SLL
                    (2, 0x00), // SLT
                    (3, 0x00), // SLTU
                    (4, 0x00), // XOR
                    (5, 0x00), // SRL
                    (5, 0x20), // SRA
                    (6, 0x00), // OR
                    (7, 0x00), // AND
                    // RV32M
                    (0, 0x01), // MUL
                    (1, 0x01), // MULH
                    (2, 0x01), // MULHSU
                    (3, 0x01), // MULHU
                    (4, 0x01), // DIV
                    (5, 0x01), // DIVU
                    (6, 0x01), // REM
                    (7, 0x01), // REMU
                ];
                let (f3, f7) = funct3_ops[rng.range(0, (funct3_ops.len() - 1) as u32) as usize];
                encode_r(0x33, rd, f3, rs1, rs2, f7)
            }
            1 => {
                // OP-IMM Integer I-Type
                let op = rng.range(0, 8);
                match op {
                    0 => encode_i(0x13, rd, 0, rs1, rng.range(0, 2047) as i32), // ADDI
                    1 => encode_i(0x13, rd, 2, rs1, rng.range(0, 2047) as i32), // SLTI
                    2 => encode_i(0x13, rd, 3, rs1, rng.range(0, 2047) as i32), // SLTIU
                    3 => encode_i(0x13, rd, 4, rs1, rng.range(0, 2047) as i32), // XORI
                    4 => encode_i(0x13, rd, 6, rs1, rng.range(0, 2047) as i32), // ORI
                    5 => encode_i(0x13, rd, 7, rs1, rng.range(0, 2047) as i32), // ANDI
                    6 => encode_i(0x13, rd, 1, rs1, (rng.range(0, 31) & 0x1F) as i32), // SLLI
                    7 => encode_i(0x13, rd, 5, rs1, (rng.range(0, 31) & 0x1F) as i32), // SRLI
                    _ => encode_i(0x13, rd, 5, rs1, 0x400 | ((rng.range(0, 31) & 0x1F) as i32)), // SRAI
                }
            }
            2 => {
                // LUI / AUIPC
                let opcode = if rng.range(0, 1) == 0 { 0x37 } else { 0x17 };
                let imm = rng.range(1, 0xFFFFF) << 12;
                encode_u(opcode, rd, imm)
            }
            3 => {
                // Single-Precision FP Compute
                let op = rng.range(0, 5);
                match op {
                    0 => encode_r(0x53, frd, 0, frs1, frs2, 0x00), // FADD.S
                    1 => encode_r(0x53, frd, 0, frs1, frs2, 0x04), // FSUB.S
                    2 => encode_r(0x53, frd, 0, frs1, frs2, 0x08), // FMUL.S
                    3 => encode_r(0x53, frd, 0, frs1, frs2, 0x0C), // FDIV.S
                    _ => encode_r(0x53, frd, rng.range(0, 2), frs1, frs2, 0x10), // FSGNJ.S
                }
            }
            4 => {
                // Double-Precision FP Compute
                let op = rng.range(0, 5);
                match op {
                    0 => encode_r(0x53, frd, 0, frs1, frs2, 0x01), // FADD.D
                    1 => encode_r(0x53, frd, 0, frs1, frs2, 0x05), // FSUB.D
                    2 => encode_r(0x53, frd, 0, frs1, frs2, 0x09), // FMUL.D
                    3 => encode_r(0x53, frd, 0, frs1, frs2, 0x0D), // FDIV.D
                    _ => encode_r(0x53, frd, rng.range(0, 2), frs1, frs2, 0x11), // FSGNJ.D
                }
            }
            5 => {
                // FMA (FMADD.S / FMADD.D)
                let opcode = match rng.range(0, 3) {
                    0 => 0x43, // FMADD
                    1 => 0x47, // FMSUB
                    2 => 0x4B, // FNMSUB
                    _ => 0x4F, // FNMADD
                };
                let fmt = rng.range(0, 1);
                encode_r4(opcode, frd, 0, frs1, frs2, frs3, fmt)
            }
            6 => {
                // Memory Load/Store at base x10 (0x10000)
                let offset = ((rng.range(0, 15) * 4) as i32) & 0x3C; // aligned 0..60 offset
                let op = rng.range(0, 3);
                match op {
                    0 => encode_s(0x23, 2, 10, rs2, offset), // SW at x10 + offset
                    1 => encode_i(0x03, rd, 2, 10, offset),  // LW at x10 + offset
                    2 => encode_s(0x27, 2, 10, frs2, offset), // FSW at x10 + offset
                    _ => encode_i(0x07, frd, 2, 10, offset), // FLW at x10 + offset
                }
            }
            _ => {
                // Default ADDI fallback
                encode_i(0x13, rd, 0, rs1, 1)
            }
        };

        insts.push(inst);
    }

    insts
}

fn ensure_whisper_oracle_built() -> String {
    let oracle_path = std::path::Path::new("emulators/SweRV-ISS-1/opt/whisper");
    if !oracle_path.exists() {
        println!("Building SweRV-ISS-1 oracle emulator binary...");
        let status = Command::new("make")
            .args(["-f", "GNUmakefile.wdc", "BOOST_DIR=/opt/homebrew/opt/boost", "opt"])
            .current_dir("emulators/SweRV-ISS-1")
            .status()
            .expect("Failed to execute make in emulators/SweRV-ISS-1");
        assert!(status.success(), "Building SweRV-ISS-1 failed!");
    }
    oracle_path.to_str().unwrap().to_string()
}

#[test]
fn test_differential_harness_against_cpp_oracle() {
    let oracle_bin = ensure_whisper_oracle_built();

    let seed: u64 = 0xDEADBEEF12345678;
    let mut rng = SimpleRng::new(seed);
    let num_insts = 100;
    let instructions = generate_random_instructions(num_insts, &mut rng);

    // 1. Write hex file for C++ whisper oracle
    let hex_path = "/tmp/test_harness_100.hex";
    {
        let mut f = File::create(hex_path).expect("Failed to create hex file");
        writeln!(f, "@00000000").unwrap();
        for &inst in &instructions {
            let bytes = inst.to_le_bytes();
            writeln!(f, "{:02x} {:02x} {:02x} {:02x}", bytes[0], bytes[1], bytes[2], bytes[3]).unwrap();
        }
    }

    // Initial state definitions
    let mut initial_regs = [0u32; 32];
    for i in 1..32 {
        initial_regs[i] = (i as u32) * 0x01010101;
    }
    initial_regs[10] = 0x10000; // x10 points to valid data memory region

    let mut initial_fregs = [0u64; 32];
    for i in 0..32 {
        initial_fregs[i] = 0xFFFFFFFF00000000 | (0x3F800000 + (i as u64) * 0x1000);
    }

    // 2. Run Rust Emulator
    let mut rust_cpu = Cpu::new();
    let mut rust_mem = Memory::new();

    // Set initial registers in rust_cpu
    for i in 1..32 {
        rust_cpu.write_reg(i, initial_regs[i]);
    }
    for i in 0..32 {
        rust_cpu.fregs[i] = f64::from_bits(initial_fregs[i]);
    }

    // Load 100 instructions into rust_mem at 0x0
    for (i, &inst) in instructions.iter().enumerate() {
        rust_mem.write_u32((i * 4) as u32, inst);
    }

    // Run 100 instruction steps in rust_cpu
    rust_cpu.pc = 0;
    for _ in 0..num_insts {
        let inst = rust_mem.read_u32(rust_cpu.pc);
        rust_cpu.execute_inst(inst, &mut rust_mem).expect("Rust CPU execution error");
    }

    // 3. Run C++ Whisper Oracle
    let mut whisper_proc = Command::new(&oracle_bin)
        .args(["--isa", "imafdc", "--raw", "--interactive"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn whisper oracle binary");

    {
        let stdin = whisper_proc.stdin.as_mut().expect("Failed to open stdin");

        // Enable FP unit in mstatus.FS (CSR 0x300)
        writeln!(stdin, "poke c 0x300 0x6000").unwrap();

        // Poke initial integer registers
        for i in 1..32 {
            writeln!(stdin, "poke r x{} {:#x}", i, initial_regs[i]).unwrap();
        }
        // Poke initial FP registers
        for i in 0..32 {
            writeln!(stdin, "poke f f{} {:#x}", i, initial_fregs[i]).unwrap();
        }

        // Load hex file & step 100 instructions
        writeln!(stdin, "hex {}", hex_path).unwrap();
        writeln!(stdin, "step {}", num_insts).unwrap();

        // Query states
        writeln!(stdin, "peek pc").unwrap();
        for i in 1..32 {
            writeln!(stdin, "peek r x{}", i).unwrap();
        }
        for i in 0..32 {
            writeln!(stdin, "peek f f{}", i).unwrap();
        }
        writeln!(stdin, "quit").unwrap();
    }

    let output = whisper_proc.wait_with_output().expect("Failed to read whisper stdout");
    let stdout_reader = BufReader::new(&output.stdout[..]);

    let oracle_pc: u32;
    let mut oracle_regs = [0u32; 32];
    let mut oracle_fregs = [0u64; 32];

    let mut lines_parsed = Vec::new();
    for line in stdout_reader.lines().flatten() {
        let trimmed = line.trim();
        if trimmed.starts_with("0x") {
            lines_parsed.push(trimmed.to_string());
        }
    }

    assert!(
        lines_parsed.len() >= 64,
        "Seed: {:#018x} - Insufficient output lines from whisper oracle! Received {}",
        seed,
        lines_parsed.len()
    );

    oracle_pc = u32::from_str_radix(lines_parsed[0].trim_start_matches("0x"), 16).unwrap();
    for i in 1..32 {
        oracle_regs[i] = u32::from_str_radix(lines_parsed[i].trim_start_matches("0x"), 16).unwrap();
    }
    for i in 0..32 {
        oracle_fregs[i] = u64::from_str_radix(lines_parsed[32 + i].trim_start_matches("0x"), 16).unwrap();
    }

    // 4. Differential State Comparison between Rust-Whisper & C++ Oracle
    assert_eq!(
        rust_cpu.pc, oracle_pc,
        "Seed: {:#018x} - PC Mismatch! Rust PC: {:#010x}, Oracle PC: {:#010x}",
        seed, rust_cpu.pc, oracle_pc
    );

    for i in 1..32 {
        assert_eq!(
            rust_cpu.read_reg(i),
            oracle_regs[i],
            "Seed: {:#018x} - Register x{} mismatch! Rust: {:#010x}, Oracle: {:#010x}",
            seed, i, rust_cpu.read_reg(i), oracle_regs[i]
        );
    }

    for i in 0..32 {
        assert_eq!(
            rust_cpu.fregs[i].to_bits(),
            oracle_fregs[i],
            "Seed: {:#018x} - FP Register f{} mismatch! Rust: {:#018x}, Oracle: {:#018x}",
            seed, i, rust_cpu.fregs[i].to_bits(), oracle_fregs[i]
        );
    }
}
