use proptest::prelude::*;
use riscv_rs::{Cpu, Memory, MemoryOps};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

static ORACLE_BUILD_RESULT: OnceLock<Result<String, String>> = OnceLock::new();

/// Base address of the data region both emulators use for loads and stores.
/// `x10` is initialised to this address and no generated instruction writes to
/// `x10`, so every memory access lands inside the compared window.
const DATA_BASE: u32 = 0x10000;

/// Size of the memory window compared after each program. It covers the whole
/// range the store strategies can reach (`DATA_BASE + 0x00 .. + 0x3C`).
const DATA_WINDOW_BYTES: u32 = 0x40;

/// Largest forward displacement a generated branch or jump can take, in bytes.
/// Execution is padded with this much room per step, so the program counter can
/// never leave the loaded image.
const MAX_BRANCH_BYTES: u32 = 16;

/// `nop` (`addi x0, x0, 0`), used to pad the image past the generated body.
const NOP: u32 = 0x0000_0013;

/// Machine scratch register. It is the one CSR that both the simulator and the
/// oracle treat as plain read/write storage, with no WARL masking and no effect
/// on how following instructions execute, so it is the only one a random
/// instruction stream can safely write.
const CSR_MSCRATCH: u32 = 0x340;

/// How far the floating-point results of the two emulators may differ, in units
/// in the last place. Zero would be correct for add, sub, mul, div and sqrt,
/// which IEEE 754 specifies exactly and both implementations round to nearest.
/// The slack exists for the fused multiply-add family: the simulator evaluates
/// it as a separate multiply and add, so it rounds twice where the oracle
/// rounds once. Override with `RISCV_RS_FP_ULP_TOLERANCE` to tighten or widen.
fn fp_ulp_tolerance() -> i128 {
    std::env::var("RISCV_RS_FP_ULP_TOLERANCE")
        .ok()
        .and_then(|v| v.parse::<i128>().ok())
        .unwrap_or(4)
}

/// True when a missing or unbuildable oracle must fail the run instead of
/// skipping it. CI sets this so the differential tests cannot pass vacuously.
fn oracle_required() -> bool {
    std::env::var("RISCV_RS_REQUIRE_ORACLE").as_deref() == Ok("1")
}

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

/// Pack two compressed instructions into one 4-byte image word. Emitting them
/// in pairs keeps every instruction boundary 4-byte aligned, so a branch or
/// jump with a 4-byte-aligned target can never land in the middle of one.
fn rvc_pair(lo: u16, hi: u16) -> u32 {
    ((hi as u32) << 16) | (lo as u32)
}

fn manifest_dir() -> std::path::PathBuf {
    std::env::var("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap())
}

fn ensure_whisper_oracle_built() -> Option<&'static str> {
    if cfg!(target_os = "windows") {
        if oracle_required() {
            panic!(
                "RISCV_RS_REQUIRE_ORACLE=1 but the SweRV-ISS oracle is not supported on Windows."
            );
        }
        println!(
            "Skipping SweRV-ISS oracle build and process-spawning differential tests on Windows."
        );
        return None;
    }

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

        let oracle_path = if swerv_dir.join("opt/whisper.exe").exists() {
            swerv_dir.join("opt/whisper.exe")
        } else {
            swerv_dir.join("opt/whisper")
        };
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
                let oracle_path = if swerv_dir.join("opt/whisper.exe").exists() {
                    swerv_dir.join("opt/whisper.exe")
                } else {
                    swerv_dir.join("opt/whisper")
                };
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
        Ok(path) => Some(path.as_str()),
        Err(err_msg) => {
            if oracle_required() {
                panic!(
                    "RISCV_RS_REQUIRE_ORACLE=1 but the SweRV-ISS-1 oracle is unavailable: {}",
                    err_msg
                );
            }
            eprintln!(
                "SweRV-ISS-1 oracle compilation skipped/failed: {}\nSkipping differential test execution.",
                err_msg
            );
            None
        }
    }
}

/// The image executed by both emulators: the generated body followed by enough
/// `nop`s that the program counter stays inside it for the whole run.
struct Program {
    /// One entry per 4-byte image word, written to the hex file verbatim.
    words: Vec<u32>,
    /// Number of instructions to step, which is not the number of words: a
    /// word may hold one 32-bit instruction or two compressed ones.
    instruction_count: usize,
}

fn build_program(body: &[u32]) -> Program {
    // Count instructions by walking the body the way a fetch unit would.
    let mut bytes = Vec::with_capacity(body.len() * 4);
    for &word in body {
        bytes.extend_from_slice(&word.to_le_bytes());
    }

    let mut instruction_count = 0usize;
    let mut offset = 0usize;
    while offset + 1 < bytes.len() {
        let half = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        offset += if (half & 0x3) == 0x3 { 4 } else { 2 };
        instruction_count += 1;
    }

    // Each step advances the program counter by at most `MAX_BRANCH_BYTES`, so
    // this much padding guarantees every fetch hits a real instruction.
    let pad_words = (MAX_BRANCH_BYTES as usize / 4) * instruction_count + 8;
    let mut words = body.to_vec();
    words.extend(std::iter::repeat_n(NOP, pad_words));

    Program {
        words,
        instruction_count,
    }
}

/// Map floating-point bits onto a monotonic integer so that the distance
/// between two keys is the number of representable values between them.
fn f32_order_key(bits: u32) -> i128 {
    if (bits & 0x8000_0000) != 0 {
        -((bits & 0x7FFF_FFFF) as i128)
    } else {
        bits as i128
    }
}

fn f64_order_key(bits: u64) -> i128 {
    if (bits & 0x8000_0000_0000_0000) != 0 {
        -((bits & 0x7FFF_FFFF_FFFF_FFFF) as i128)
    } else {
        bits as i128
    }
}

fn run_differential_test(instructions: &[u32]) {
    let oracle_bin = match ensure_whisper_oracle_built() {
        Some(bin) => bin,
        None => return,
    };
    if instructions.is_empty() {
        return;
    }

    let program = build_program(instructions);
    let num_insts = program.instruction_count;
    if num_insts == 0 {
        return;
    }

    let temp_dir = std::env::temp_dir();
    let thread_id = std::thread::current().id();
    let hex_path = temp_dir.join(format!("test_harness_{:?}_{}.hex", thread_id, num_insts));
    {
        let mut f = File::create(&hex_path).expect("Failed to create hex file");
        writeln!(f, "@00000000").unwrap();
        for &inst in &program.words {
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
    initial_regs[10] = DATA_BASE; // x10 points to valid data memory region

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

    for (i, &inst) in program.words.iter().enumerate() {
        rust_mem.write_u32((i * 4) as u32, inst);
    }

    rust_cpu.pc = 0;
    for step in 0..num_insts {
        let pc = rust_cpu.pc;
        let half = rust_mem.read_u16(pc);
        let result = if (half & 0x3) != 0x3 {
            rust_cpu.execute_inst16(half, &mut rust_mem)
        } else {
            rust_cpu.execute_inst32(rust_mem.read_u32(pc), &mut rust_mem)
        };
        if let Err(e) = result {
            panic!(
                "Rust CPU execution error on step {} at PC {:#010x} (raw {:#010x}): {}",
                step,
                pc,
                rust_mem.read_u32(pc),
                e
            );
        }
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
        // Memory peeks print "<addr>: <word>", so they are told apart from the
        // register peeks above by the colon.
        writeln!(
            stdin,
            "peek m {:#x} {:#x}",
            DATA_BASE,
            DATA_BASE + DATA_WINDOW_BYTES - 4
        )
        .unwrap();
        writeln!(stdin, "quit").unwrap();
    }

    let output = whisper_proc
        .wait_with_output()
        .expect("Failed to read whisper stdout");
    let stdout_reader = BufReader::new(&output.stdout[..]);

    let mut reg_lines = Vec::new();
    let mut mem_words = Vec::new();
    for line in stdout_reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if !trimmed.starts_with("0x") {
            continue;
        }
        match trimmed.split_once(": ") {
            Some((_addr, value)) => {
                mem_words.push(u32::from_str_radix(value.trim_start_matches("0x"), 16).unwrap());
            }
            None => reg_lines.push(trimmed.to_string()),
        }
    }

    let _ = std::fs::remove_file(&hex_path);

    assert!(
        reg_lines.len() >= 64,
        "Insufficient register output from whisper oracle! Received {}",
        reg_lines.len()
    );
    assert_eq!(
        mem_words.len(),
        (DATA_WINDOW_BYTES / 4) as usize,
        "Insufficient memory output from whisper oracle! Received {} words",
        mem_words.len()
    );

    let oracle_pc = u32::from_str_radix(reg_lines[0].trim_start_matches("0x"), 16).unwrap();
    let mut oracle_regs = [0u32; 32];
    for i in 1..32 {
        oracle_regs[i] = u32::from_str_radix(reg_lines[i].trim_start_matches("0x"), 16).unwrap();
    }
    let mut oracle_fregs = [0u64; 32];
    for i in 0..32 {
        oracle_fregs[i] =
            u64::from_str_radix(reg_lines[32 + i].trim_start_matches("0x"), 16).unwrap();
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

    for (i, &oracle_word) in mem_words.iter().enumerate() {
        let addr = DATA_BASE + (i as u32) * 4;
        let rust_word = rust_mem.read_u32(addr);
        assert_eq!(
            rust_word, oracle_word,
            "Memory mismatch at {:#010x}! Rust: {:#010x}, Oracle: {:#010x}",
            addr, rust_word, oracle_word
        );
    }

    let tolerance = fp_ulp_tolerance();
    for (i, &oracle_bits) in oracle_fregs.iter().enumerate() {
        let rust_bits = rust_cpu.fregs[i].to_bits();
        if rust_bits == oracle_bits {
            continue;
        }
        let r_f64 = f64::from_bits(rust_bits);
        let o_f64 = f64::from_bits(oracle_bits);
        if r_f64.is_nan() && o_f64.is_nan() {
            // NaN payloads are not architecturally fixed for every operation.
            continue;
        }
        let ulps = if (rust_bits >> 32) == 0xFFFFFFFF && (oracle_bits >> 32) == 0xFFFFFFFF {
            // Both values are NaN-boxed single-precision floats.
            let r_f32 = f32::from_bits(rust_bits as u32);
            let o_f32 = f32::from_bits(oracle_bits as u32);
            if r_f32.is_nan() && o_f32.is_nan() {
                continue;
            }
            (f32_order_key(rust_bits as u32) - f32_order_key(oracle_bits as u32)).abs()
        } else {
            (f64_order_key(rust_bits) - f64_order_key(oracle_bits)).abs()
        };
        assert!(
            ulps <= tolerance,
            "FP Register f{} mismatch by {} ulp (tolerance {})! Rust: {:#018x}, Oracle: {:#018x}",
            i,
            ulps,
            tolerance,
            rust_bits,
            oracle_bits
        );
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

/// A compressed register field value that never selects `x10`, which holds the
/// data base address the memory strategies depend on.
fn arb_creg_field() -> impl Strategy<Value = u16> {
    (0..8u16).prop_map(|r| if r == 2 { 3 } else { r })
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

/// Conditional branches. Targets are forward, word aligned and no further than
/// `MAX_BRANCH_BYTES`, so the program counter stays inside the padded image.
fn arb_branch_instruction() -> impl Strategy<Value = u32> {
    (arb_any_reg(), arb_any_reg(), 0..6usize, 1..=4i32).prop_map(|(rs1, rs2, op, words)| {
        let funct3 = [0u32, 1, 4, 5, 6, 7][op];
        encode_b(0x63, funct3, rs1, rs2, words * 4)
    })
}

/// Unconditional jumps. `jal` moves forward by a bounded amount; `jalr` is
/// always taken from `x0` so its target is an absolute address near the start
/// of the image rather than whatever a random register happens to hold.
fn arb_jump_instruction() -> impl Strategy<Value = u32> {
    prop_oneof![
        (arb_reg(), 1..=4i32).prop_map(|(rd, words)| encode_j(0x6F, rd, words * 4)),
        (arb_reg(), 0..=8i32).prop_map(|(rd, words)| encode_i(0x67, rd, 0, 0, words * 4)),
    ]
}

/// Atomic memory operations on the shared data word at `x10`.
///
/// `lr.w` and `sc.w` are deliberately absent: the simulator does not model
/// reservations, so its `sc.w` always succeeds while the oracle's may not.
fn arb_amo_instruction() -> impl Strategy<Value = u32> {
    (arb_reg(), arb_any_reg(), 0..9usize).prop_map(|(rd, rs2, op)| {
        let funct5 = [0x01u32, 0x00, 0x04, 0x0C, 0x08, 0x10, 0x14, 0x18, 0x1C][op];
        encode_r(0x2F, rd, 2, 10, rs2, funct5 << 2)
    })
}

/// CSR reads and writes against `mscratch`, the one register both emulators
/// treat as plain storage.
fn arb_csr_instruction() -> impl Strategy<Value = u32> {
    (arb_reg(), arb_any_reg(), 0..6usize, 0..32i32).prop_map(|(rd, rs1, op, zimm)| {
        match op {
            0 => encode_i(0x73, rd, 1, rs1, CSR_MSCRATCH as i32), // csrrw
            1 => encode_i(0x73, rd, 2, rs1, CSR_MSCRATCH as i32), // csrrs
            2 => encode_i(0x73, rd, 3, rs1, CSR_MSCRATCH as i32), // csrrc
            3 => encode_i(0x73, rd, 5, zimm as usize, CSR_MSCRATCH as i32), // csrrwi
            4 => encode_i(0x73, rd, 6, zimm as usize, CSR_MSCRATCH as i32), // csrrsi
            _ => encode_i(0x73, rd, 7, zimm as usize, CSR_MSCRATCH as i32), // csrrci
        }
    })
}

/// A single compressed instruction. Control-flow forms are excluded so that
/// every branch target stays 4-byte aligned; the 32-bit strategies cover jumps.
fn arb_compressed_instruction() -> impl Strategy<Value = u16> {
    prop_oneof![
        // c.addi / c.li / c.slli, on the full register file.
        (arb_reg(), 1..32u16, 0..3usize).prop_map(|(rd, imm, op)| {
            let rd = rd as u16;
            match op {
                0 => rvc(1, 0, &[(0, 12, 12), (rd, 11, 7), (imm, 6, 2)]), // c.addi
                1 => rvc(1, 2, &[(0, 12, 12), (rd, 11, 7), (imm, 6, 2)]), // c.li
                _ => rvc(2, 0, &[(0, 12, 12), (rd, 11, 7), (imm, 6, 2)]), // c.slli
            }
        }),
        // c.lui, which may not target x0 or sp and may not use a zero immediate.
        (3..32u16, 1..32u16).prop_map(|(rd, imm)| {
            let rd = if rd == 10 { 11 } else { rd };
            rvc(1, 3, &[(0, 12, 12), (rd, 11, 7), (imm, 6, 2)])
        }),
        // c.addi16sp, the instruction whose immediate scaling this harness
        // exists to pin down.
        // `v` is nzimm[9:4]; bit k of `v` is nzimm bit k + 4.
        (1..64u16).prop_map(|v| {
            rvc(
                1,
                3,
                &[
                    ((v >> 5) & 1, 12, 12), // nzimm[9]
                    (2, 11, 7),             // rd = sp
                    (v & 1, 6, 6),          // nzimm[4]
                    ((v >> 2) & 1, 5, 5),   // nzimm[6]
                    ((v >> 3) & 3, 4, 3),   // nzimm[8:7]
                    ((v >> 1) & 1, 2, 2),   // nzimm[5]
                ],
            )
        }),
        // c.addi4spn, with a nonzero immediate.
        (arb_creg_field(), 1..256u16)
            .prop_map(|(rd, uimm)| { rvc(0, 0, &[(uimm, 12, 5), (rd, 4, 2)]) }),
        // c.mv / c.add.
        (arb_reg(), arb_any_reg(), prop::bool::ANY).prop_map(|(rd, rs2, is_add)| {
            rvc(
                2,
                4,
                &[
                    (if is_add { 1 } else { 0 }, 12, 12),
                    (rd as u16, 11, 7),
                    (rs2 as u16, 6, 2),
                ],
            )
        }),
        // c.srli / c.srai / c.andi.
        (arb_creg_field(), 1..32u16, 0..3usize).prop_map(|(rd, imm, op)| {
            let sel = op as u16;
            rvc(1, 4, &[(0, 12, 12), (sel, 11, 10), (rd, 9, 7), (imm, 6, 2)])
        }),
        // c.sub / c.xor / c.or / c.and.
        (arb_creg_field(), arb_creg_field(), 0..4u16).prop_map(|(rd, rs2, op)| {
            rvc(
                1,
                4,
                &[
                    (0, 12, 12),
                    (3, 11, 10),
                    (rd, 9, 7),
                    (op, 6, 5),
                    (rs2, 4, 2),
                ],
            )
        }),
        // c.lw / c.sw against the data region held in x10.
        (arb_creg_field(), 0..16u16, prop::bool::ANY).prop_map(|(reg, word, is_store)| {
            // uimm[5:3] = inst[12:10], uimm[2] = inst[6], uimm[6] = inst[5].
            let offset = word * 4;
            let fields = [
                ((offset >> 3) & 0x7, 12, 10),
                (2, 9, 7), // rs1' = x10
                ((offset >> 2) & 0x1, 6, 6),
                (0, 5, 5),
                (reg, 4, 2),
            ];
            if is_store {
                rvc(0, 6, &fields)
            } else {
                rvc(0, 2, &fields)
            }
        }),
    ]
}

/// Two compressed instructions packed into one image word.
fn arb_compressed_pair() -> impl Strategy<Value = u32> {
    (arb_compressed_instruction(), arb_compressed_instruction())
        .prop_map(|(lo, hi)| rvc_pair(lo, hi))
}

fn arb_fp_instruction() -> impl Strategy<Value = u32> {
    prop_oneof![
        // Single-Precision FP Compute
        (
            arb_freg(),
            arb_reg(),
            arb_freg(),
            arb_freg(),
            0..5u32,
            0..2u32
        )
            .prop_map(|(frd, ird, frs1, frs2, op, rm)| {
                match op {
                    0 => encode_r(0x53, frd, 0, frs1, frs2, 0x00),
                    1 => encode_r(0x53, frd, 0, frs1, frs2, 0x04),
                    2 => encode_r(0x53, frd, 0, frs1, frs2, 0x08),
                    3 => encode_r(0x53, frd, 0, frs1, frs2, 0x0C),
                    _ => encode_r(0x53, ird, rm, frs1, frs2, 0x10),
                }
            }),
        // Double-Precision FP Compute
        (
            arb_freg(),
            arb_reg(),
            arb_freg(),
            arb_freg(),
            0..5u32,
            0..2u32
        )
            .prop_map(|(frd, ird, frs1, frs2, op, rm)| {
                match op {
                    0 => encode_r(0x53, frd, 0, frs1, frs2, 0x01),
                    1 => encode_r(0x53, frd, 0, frs1, frs2, 0x05),
                    2 => encode_r(0x53, frd, 0, frs1, frs2, 0x09),
                    3 => encode_r(0x53, frd, 0, frs1, frs2, 0x0D),
                    _ => encode_r(0x53, ird, rm, frs1, frs2, 0x11),
                }
            }),
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

fn arb_memory_instruction() -> impl Strategy<Value = u32> {
    (arb_reg(), arb_any_reg(), arb_freg(), 0..16i32, 0..4u32).prop_map(
        |(rd, rs2, frd, offset_idx, op)| {
            let offset = (offset_idx * 4) & 0x3C;
            match op {
                0 => encode_s(0x23, 2, 10, rs2, offset), // SW
                1 => encode_i(0x03, rd, 2, 10, offset),  // LW
                2 => encode_s(0x27, 2, 10, frd, offset), // FSW
                _ => encode_i(0x07, frd, 2, 10, offset), // FLW
            }
        },
    )
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
        arb_memory_instruction(),
        arb_branch_instruction(),
        arb_jump_instruction(),
        arb_amo_instruction(),
        arb_csr_instruction(),
        arb_compressed_pair(),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

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
    fn test_differential_control_flow_instructions(
        instructions in proptest::collection::vec(
            prop_oneof![
                arb_branch_instruction(),
                arb_jump_instruction(),
                arb_r_type_instruction(),
                arb_memory_instruction(),
            ],
            10..40
        )
    ) {
        run_differential_test(&instructions);
    }

    #[test]
    fn test_differential_amo_and_csr_instructions(
        instructions in proptest::collection::vec(
            prop_oneof![arb_amo_instruction(), arb_csr_instruction(), arb_memory_instruction()],
            10..40
        )
    ) {
        run_differential_test(&instructions);
    }

    #[test]
    fn test_differential_compressed_instructions(
        instructions in proptest::collection::vec(arb_compressed_pair(), 10..40)
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

#[test]
fn test_failing_ci_sequence() {
    let instructions = vec![
        33554627, 34941991, 63250563, 59058435, 1782843719, 1631338131, 1141073939, 48570915,
        289540691, 548836435,
    ];
    run_differential_test(&instructions);
}

/// `c.addi16sp sp, -16` must move the stack pointer by exactly -16. This is a
/// direct guard on the immediate that used to be scaled twice.
#[test]
fn test_differential_addi16sp_scaling() {
    let c_addi16sp_minus_16 = rvc(
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
    let c_nop = rvc(1, 0, &[]);
    run_differential_test(&[rvc_pair(c_addi16sp_minus_16, c_nop); 4]);
}
