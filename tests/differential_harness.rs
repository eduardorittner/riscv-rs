use proptest::prelude::*;
use rust_whisper::{Cpu, Memory, MemoryOps};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

static ORACLE_BUILD_RESULT: OnceLock<Result<String, String>> = OnceLock::new();

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

fn encode_u(opcode: u32, rd: usize, imm: u32) -> u32 {
    (imm & 0xFFFFF000) | ((rd as u32) << 7) | opcode
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

fn manifest_dir() -> std::path::PathBuf {
    std::env::var("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap())
}

fn ensure_whisper_oracle_built() -> &'static str {
    let res = ORACLE_BUILD_RESULT.get_or_init(|| {
        let base_dir = manifest_dir();
        let swerv_dir = base_dir.join("SweRV-ISS-1");
        let makefile_path = swerv_dir.join("GNUmakefile.wdc");

        if !makefile_path.exists() {
            println!("SweRV-ISS-1 submodule not found or uninitialized. Initializing git submodule...");
            let _ = Command::new("git")
                .args(["submodule", "update", "--init", "--recursive"])
                .current_dir(&base_dir)
                .status();
        }

        if !swerv_dir.exists() {
            return Err(format!(
                "SweRV-ISS-1 directory does not exist at {}",
                swerv_dir.display()
            ));
        }

        if !makefile_path.exists() {
            return Err(format!(
                "SweRV-ISS-1 GNUmakefile.wdc not found at {}",
                makefile_path.display()
            ));
        }

        let oracle_path = swerv_dir.join("opt/whisper");
        if oracle_path.exists() {
            return Ok(oracle_path.to_str().unwrap().to_string());
        }

        let vendor_dir = base_dir.join("vendor");
        if !vendor_dir.exists() {
            return Err(format!("vendor directory does not exist at {}", vendor_dir.display()));
        }
        let vendor_dir_str = vendor_dir.to_str().unwrap();
        let po_lib = vendor_dir.join("stage/lib/libboost_program_options.a");
        if !po_lib.exists() {
            println!("Building vendored libboost_program_options.a...");
            let status = Command::new("make")
                .args(["-f", "vendor/GNUmakefile.vendor"])
                .current_dir(&base_dir)
                .status();
            match status {
                Ok(s) if s.success() => {}
                Ok(s) => {
                    return Err(format!(
                        "Building vendored libboost_program_options.a failed with status: {s}"
                    ))
                }
                Err(e) => {
                    return Err(format!("Failed to execute make for vendored boost: {e}"))
                }
            }
        }

        let cxx_base = std::env::var("CXX").unwrap_or_else(|_| "g++".to_string());
        let cxx = format!(
            "{} -include cstdint -include optional -include limits -Wno-unknown-warning-option -Wno-invalid-specialization",
            cxx_base
        );
        println!("Building SweRV-ISS-1 oracle emulator binary...");
        let status = Command::new("make")
            .args([
                "-f",
                "GNUmakefile.wdc",
                &format!("BOOST_DIR={}", vendor_dir_str),
                "opt",
            ])
            .env("CXX", cxx)
            .current_dir(&swerv_dir)
            .status();

        match status {
            Ok(s) if s.success() => {
                if oracle_path.exists() {
                    Ok(oracle_path.to_str().unwrap().to_string())
                } else {
                    Err(format!("Make reported success but {} was not created", oracle_path.display()))
                }
            }
            Ok(s) => Err(format!("Building SweRV-ISS-1 oracle emulator failed with exit status: {s}")),
            Err(e) => Err(format!("Failed to execute make in {}: {e}", swerv_dir.display())),
        }
    });

    match res {
        Ok(path) => path.as_str(),
        Err(err_msg) => {
            panic!("SweRV-ISS-1 compilation failed: {}\nAborting test.", err_msg);
        }
    }
}

fn run_differential_test(instructions: &[u32]) {
    let oracle_bin = ensure_whisper_oracle_built();
    let num_insts = instructions.len();
    if num_insts == 0 {
        return;
    }

    let temp_dir = std::env::temp_dir();
    let thread_id = std::thread::current().id();
    let hex_path = temp_dir.join(format!("test_harness_{:?}_{}.hex", thread_id, num_insts));
    {
        let mut f = File::create(&hex_path).expect("Failed to create hex file");
        writeln!(f, "@00000000").unwrap();
        for &inst in instructions {
            let bytes = inst.to_le_bytes();
            writeln!(
                f,
                "{:02x} {:02x} {:02x} {:02x}",
                bytes[0], bytes[1], bytes[2], bytes[3]
            )
            .unwrap();
        }
    }

    // Initial state definitions
    let mut initial_regs = [0u32; 32];
    for (i, reg) in initial_regs.iter_mut().enumerate().skip(1) {
        *reg = (i as u32) * 0x01010101;
    }
    initial_regs[10] = 0x10000; // x10 points to valid data memory region

    let mut initial_fregs = [0u64; 32];
    for (i, freg) in initial_fregs.iter_mut().enumerate() {
        *freg = 0xFFFFFFFF00000000 | (0x3F800000 + (i as u64) * 0x1000);
    }

    // 1. Run Rust Emulator
    let mut rust_cpu = Cpu::new();
    let mut rust_mem = Memory::new();

    for (i, &reg) in initial_regs.iter().enumerate().skip(1) {
        rust_cpu.write_reg(i, reg);
    }
    for (i, &freg) in initial_fregs.iter().enumerate() {
        rust_cpu.fregs[i] = f64::from_bits(freg);
    }

    for (i, &inst) in instructions.iter().enumerate() {
        rust_mem.write_u32((i * 4) as u32, inst);
    }

    rust_cpu.pc = 0;
    for _ in 0..num_insts {
        let inst = rust_mem.read_u32(rust_cpu.pc);
        rust_cpu
            .execute_inst32(inst, &mut rust_mem)
            .expect("Rust CPU execution error");
    }

    // 2. Run C++ Whisper Oracle
    let mut whisper_proc = Command::new(oracle_bin)
        .args(["--isa", "imafdc", "--raw", "--interactive"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to spawn whisper oracle binary");

    {
        let stdin = whisper_proc.stdin.as_mut().expect("Failed to open stdin");

        // Enable FP unit in mstatus.FS (CSR 0x300)
        writeln!(stdin, "poke c 0x300 0x6000").unwrap();

        for (i, &reg) in initial_regs.iter().enumerate().skip(1) {
            writeln!(stdin, "poke r x{} {:#x}", i, reg).unwrap();
        }
        for (i, &freg) in initial_fregs.iter().enumerate() {
            writeln!(stdin, "poke f f{} {:#x}", i, freg).unwrap();
        }

        writeln!(stdin, "hex {}", hex_path.to_str().unwrap()).unwrap();
        writeln!(stdin, "step {}", num_insts).unwrap();

        writeln!(stdin, "peek pc").unwrap();
        for i in 1..32 {
            writeln!(stdin, "peek r x{}", i).unwrap();
        }
        for i in 0..32 {
            writeln!(stdin, "peek f f{}", i).unwrap();
        }
        writeln!(stdin, "quit").unwrap();
    }

    let output = whisper_proc
        .wait_with_output()
        .expect("Failed to read whisper stdout");
    let stdout_reader = BufReader::new(&output.stdout[..]);

    let mut lines_parsed = Vec::new();
    for line in stdout_reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.starts_with("0x") {
            lines_parsed.push(trimmed.to_string());
        }
    }

    let _ = std::fs::remove_file(&hex_path);

    assert!(
        lines_parsed.len() >= 64,
        "Insufficient output lines from whisper oracle! Received {}",
        lines_parsed.len()
    );

    let oracle_pc = u32::from_str_radix(lines_parsed[0].trim_start_matches("0x"), 16).unwrap();
    let mut oracle_regs = [0u32; 32];
    for i in 1..32 {
        oracle_regs[i] = u32::from_str_radix(lines_parsed[i].trim_start_matches("0x"), 16).unwrap();
    }
    let mut oracle_fregs = [0u64; 32];
    for i in 0..32 {
        oracle_fregs[i] =
            u64::from_str_radix(lines_parsed[32 + i].trim_start_matches("0x"), 16).unwrap();
    }

    // 3. Differential State Comparison
    assert_eq!(
        rust_cpu.pc, oracle_pc,
        "PC Mismatch! Rust PC: {:#010x}, Oracle PC: {:#010x}",
        rust_cpu.pc, oracle_pc
    );

    for (i, &oracle_reg) in oracle_regs.iter().enumerate().skip(1) {
        assert_eq!(
            rust_cpu.read_reg(i),
            oracle_reg,
            "Register x{} mismatch! Rust: {:#010x}, Oracle: {:#010x}",
            i,
            rust_cpu.read_reg(i),
            oracle_reg
        );
    }

    for (i, &oracle_bits) in oracle_fregs.iter().enumerate() {
        let rust_bits = rust_cpu.fregs[i].to_bits();
        if rust_bits != oracle_bits {
            if (rust_bits >> 32) == 0xFFFFFFFF && (oracle_bits >> 32) == 0xFFFFFFFF {
                let r_f32 = f32::from_bits(rust_bits as u32);
                let o_f32 = f32::from_bits(oracle_bits as u32);
                if r_f32.is_nan() && o_f32.is_nan() {
                    continue;
                }
                let diff = (rust_bits as u32 as i64 - oracle_bits as u32 as i64).abs();
                if (r_f32 - o_f32).abs() < 1e-3 || diff <= 1024 {
                    continue;
                }
            } else {
                let r_f64 = f64::from_bits(rust_bits);
                let o_f64 = f64::from_bits(oracle_bits);
                if r_f64.is_nan() && o_f64.is_nan() {
                    continue;
                }
                let diff = ((rust_bits as i128) - (oracle_bits as i128)).abs();
                if (r_f64 - o_f64).abs() < 1e-3 || diff <= 1024 {
                    continue;
                }
            }
            assert_eq!(
                rust_bits, oracle_bits,
                "FP Register f{} mismatch! Rust: {:#018x}, Oracle: {:#018x}",
                i, rust_bits, oracle_bits
            );
        }
    }
}

fn arb_reg() -> impl Strategy<Value = usize> {
    (1..32usize).prop_map(|r| if r == 10 { 11 } else { r })
}

fn arb_any_reg() -> impl Strategy<Value = usize> {
    1..32usize
}

fn arb_freg() -> impl Strategy<Value = usize> {
    0..32usize
}

fn arb_r_type_instruction() -> impl Strategy<Value = u32> {
    (arb_reg(), arb_any_reg(), arb_any_reg(), 0..18usize).prop_map(|(rd, rs1, rs2, op_idx)| {
        let funct3_ops = [
            (0, 0x00),
            (0, 0x20),
            (1, 0x00),
            (2, 0x00),
            (3, 0x00),
            (4, 0x00),
            (5, 0x00),
            (5, 0x20),
            (6, 0x00),
            (7, 0x00),
            (0, 0x01),
            (1, 0x01),
            (2, 0x01),
            (3, 0x01),
            (4, 0x01),
            (5, 0x01),
            (6, 0x01),
            (7, 0x01),
        ];
        let (f3, f7) = funct3_ops[op_idx];
        encode_r(0x33, rd, f3, rs1, rs2, f7)
    })
}

fn arb_i_type_instruction() -> impl Strategy<Value = u32> {
    (arb_reg(), arb_any_reg(), 0..9u32, 0..2048i32).prop_map(|(rd, rs1, op, raw_imm)| match op {
        0 => encode_i(0x13, rd, 0, rs1, raw_imm),
        1 => encode_i(0x13, rd, 2, rs1, raw_imm),
        2 => encode_i(0x13, rd, 3, rs1, raw_imm),
        3 => encode_i(0x13, rd, 4, rs1, raw_imm),
        4 => encode_i(0x13, rd, 6, rs1, raw_imm),
        5 => encode_i(0x13, rd, 7, rs1, raw_imm),
        6 => encode_i(0x13, rd, 1, rs1, raw_imm & 0x1F),
        7 => encode_i(0x13, rd, 5, rs1, raw_imm & 0x1F),
        _ => encode_i(0x13, rd, 5, rs1, 0x400 | (raw_imm & 0x1F)),
    })
}

fn arb_fp_instruction() -> impl Strategy<Value = u32> {
    prop_oneof![
        // Single-Precision FP Compute
        (arb_freg(), arb_freg(), arb_freg(), 0..5u32, 0..2u32).prop_map(
            |(frd, frs1, frs2, op, rm)| {
                match op {
                    0 => encode_r(0x53, frd, 0, frs1, frs2, 0x00),
                    1 => encode_r(0x53, frd, 0, frs1, frs2, 0x04),
                    2 => encode_r(0x53, frd, 0, frs1, frs2, 0x08),
                    3 => encode_r(0x53, frd, 0, frs1, frs2, 0x0C),
                    _ => encode_r(0x53, frd, rm, frs1, frs2, 0x10),
                }
            }
        ),
        // Double-Precision FP Compute
        (arb_freg(), arb_freg(), arb_freg(), 0..5u32, 0..2u32).prop_map(
            |(frd, frs1, frs2, op, rm)| {
                match op {
                    0 => encode_r(0x53, frd, 0, frs1, frs2, 0x01),
                    1 => encode_r(0x53, frd, 0, frs1, frs2, 0x05),
                    2 => encode_r(0x53, frd, 0, frs1, frs2, 0x09),
                    3 => encode_r(0x53, frd, 0, frs1, frs2, 0x0D),
                    _ => encode_r(0x53, frd, rm, frs1, frs2, 0x11),
                }
            }
        ),
        // FMA
        (
            arb_freg(),
            arb_freg(),
            arb_freg(),
            arb_freg(),
            0..4u32,
            0..2u32
        )
            .prop_map(|(frd, frs1, frs2, frs3, op, fmt)| {
                let opcode = match op {
                    0 => 0x43,
                    1 => 0x47,
                    2 => 0x4B,
                    _ => 0x4F,
                };
                encode_r4(opcode, frd, 0, frs1, frs2, frs3, fmt)
            }),
    ]
}

fn arb_instruction() -> impl Strategy<Value = u32> {
    prop_oneof![
        arb_r_type_instruction(),
        arb_i_type_instruction(),
        (arb_reg(), prop::bool::ANY, 1..0xFFFFF32u32).prop_map(|(rd, is_lui, raw_imm)| {
            let opcode = if is_lui { 0x37 } else { 0x17 };
            let imm = (raw_imm & 0xFFFFF) << 12;
            encode_u(opcode, rd, imm)
        }),
        arb_fp_instruction(),
        (arb_reg(), arb_any_reg(), arb_freg(), 0..16i32, 0..4u32).prop_map(
            |(rd, rs2, frd, offset_idx, op)| {
                let offset = (offset_idx * 4) & 0x3C;
                match op {
                    0 => encode_s(0x23, 2, 10, rs2, offset), // SW
                    1 => encode_i(0x03, rd, 2, 10, offset),  // LW
                    2 => encode_s(0x27, 2, 10, frd, offset), // FSW
                    _ => encode_i(0x07, frd, 2, 10, offset), // FLW
                }
            }
        ),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn test_differential_integer_instructions(
        instructions in proptest::collection::vec(
            prop_oneof![arb_r_type_instruction(), arb_i_type_instruction()],
            10..40
        )
    ) {
        run_differential_test(&instructions);
    }

    #[test]
    fn test_differential_floating_point_instructions(
        instructions in proptest::collection::vec(arb_fp_instruction(), 10..40)
    ) {
        run_differential_test(&instructions);
    }

    #[test]
    fn test_differential_random_instruction_sequences(
        instructions in proptest::collection::vec(arb_instruction(), 10..50)
    ) {
        run_differential_test(&instructions);
    }
}
