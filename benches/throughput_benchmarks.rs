//! Instruction-throughput benchmarks for the CPU core.
//!
//! `memory_benchmarks.rs` measures the memory model in isolation. This file
//! measures the thing the user actually waits for: how many guest instructions
//! per second `run_slice` retires.
//!
//! Every workload is a self-contained loop with a fixed, known instruction
//! count, so Criterion's throughput figures are instructions per second rather
//! than iterations per second. The three shapes stress different parts of the
//! hot path:
//!
//! * `alu_loop` — register-only work. Measures fetch, decode and dispatch with
//!   no memory traffic beyond the fetch.
//! * `memcpy_loop` — a load/store pair per iteration. Adds the data-side page
//!   lookup to the instruction-side one.
//! * `mixed_loop` — compressed and uncompressed instructions interleaved, so
//!   the 16/32-bit fetch split is on the measured path.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use riscv_rs::{Cpu, Memory, MemoryOps};

const PROGRAM_BASE: u32 = 0x1000;
/// Where `memcpy_loop` reads from and writes to. Far from the program so the
/// data accesses land on a different page than the instruction fetches.
const SRC_BASE: u32 = 0x100000;
const DST_BASE: u32 = 0x200000;

// ─── Encoders ───────────────────────────────────────────────────────────────

fn encode_i(opcode: u32, rd: usize, funct3: u32, rs1: usize, imm: i32) -> u32 {
    (((imm as u32) & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

fn encode_r(opcode: u32, rd: usize, funct3: u32, rs1: usize, rs2: usize, funct7: u32) -> u32 {
    (funct7 << 25)
        | ((rs2 as u32) << 20)
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

/// `addi rd, rs1, imm`
fn addi(rd: usize, rs1: usize, imm: i32) -> u32 {
    encode_i(0x13, rd, 0, rs1, imm)
}

/// `xori rd, rs1, imm`
fn xori(rd: usize, rs1: usize, imm: i32) -> u32 {
    encode_i(0x13, rd, 4, rs1, imm)
}

/// `slli rd, rs1, shamt`
fn slli(rd: usize, rs1: usize, shamt: u32) -> u32 {
    encode_i(0x13, rd, 1, rs1, shamt as i32)
}

/// `add rd, rs1, rs2`
fn add(rd: usize, rs1: usize, rs2: usize) -> u32 {
    encode_r(0x33, rd, 0, rs1, rs2, 0)
}

/// `lw rd, imm(rs1)`
fn lw(rd: usize, rs1: usize, imm: i32) -> u32 {
    encode_i(0x03, rd, 2, rs1, imm)
}

/// `sw rs2, imm(rs1)`
fn sw(rs1: usize, rs2: usize, imm: i32) -> u32 {
    encode_s(0x23, 2, rs1, rs2, imm)
}

/// `lui rd, imm20`
fn lui(rd: usize, imm20: u32) -> u32 {
    ((imm20 & 0xFFFFF) << 12) | ((rd as u32) << 7) | 0x37
}

/// `li rd, value` as the canonical `lui` + `addi` pair.
///
/// `addi` alone carries a 12-bit signed immediate, so anything above 2047
/// silently sign-extends to a negative number. A loop counter built that way
/// never reaches zero.
fn li(rd: usize, value: i32) -> [u32; 2] {
    let v = value as u32;
    // The `addi` immediate is sign-extended, so the high part is rounded up
    // when bit 11 of the value is set.
    let hi = v.wrapping_add(0x800) >> 12;
    let lo = (v.wrapping_sub(hi << 12)) as i32;
    [lui(rd, hi), addi(rd, rd, lo)]
}

const ECALL: u32 = 0x0000_0073;

/// `bne rs1, x0, offset`
fn bnez(rs1: usize, offset: i32) -> u32 {
    encode_b(0x63, 1, rs1, 0, offset)
}

/// `c.addi rd, imm` — a 16-bit instruction. `rd` must not be x0 and `imm` must
/// not be 0, or the encoding means `c.nop` instead.
fn c_addi(rd: usize, imm: i32) -> u16 {
    let imm_u = imm as u32;
    // funct3 is 0b000 for C.ADDI, so bits 15:13 contribute nothing.
    ((((imm_u >> 5) & 1) as u16) << 12) | ((rd as u16) << 7) | (((imm_u & 0x1F) as u16) << 2) | 0b01
}

// ─── Program construction ───────────────────────────────────────────────────

/// A guest program as a flat halfword stream, so a 16-bit instruction and a
/// 32-bit one can sit next to each other.
#[derive(Default)]
struct Program {
    halfwords: Vec<u16>,
}

impl Program {
    fn push32(&mut self, word: u32) {
        self.halfwords.push(word as u16);
        self.halfwords.push((word >> 16) as u16);
    }

    fn push_pair(&mut self, pair: [u32; 2]) {
        self.push32(pair[0]);
        self.push32(pair[1]);
    }

    fn push16(&mut self, half: u16) {
        self.halfwords.push(half);
    }

    /// Byte length of the program so far — the branch offsets need it.
    fn len_bytes(&self) -> i32 {
        (self.halfwords.len() * 2) as i32
    }

    /// `exit(0)`. Every workload ends with it: without a halt the CPU runs on
    /// into zeroed memory, and the resulting illegal-instruction error formats
    /// a message and pushes it to the mock stderr on every single sample.
    fn push_exit(&mut self) {
        self.push32(addi(17, 0, 93)); // a7 = SYS_exit
        self.push32(addi(10, 0, 0)); // a0 = 0
        self.push32(ECALL);
    }

    fn load(&self, mem: &mut Memory) {
        for (i, half) in self.halfwords.iter().enumerate() {
            mem.write_u16(PROGRAM_BASE + (i as u32) * 2, *half);
        }
    }
}

/// A workload: the program image and the exact number of instructions one full
/// run retires.
struct Workload {
    program: Program,
    instructions: u64,
}

/// `iterations` passes of a five-instruction register-only body.
///
/// Body: four ALU instructions plus the loop-closing branch. The counter uses
/// `addi x5, x5, -1`, which is one of the four.
fn alu_loop(iterations: i32) -> Workload {
    let mut p = Program::default();
    p.push_pair(li(5, iterations)); // x5 = iterations   (setup, 2 instructions)

    let body_start = p.len_bytes();
    p.push32(addi(5, 5, -1)); // x5 -= 1
    p.push32(xori(6, 6, 0x5A5)); // x6 ^= 0x5A5
    p.push32(slli(7, 6, 3)); // x7 = x6 << 3
    p.push32(add(8, 7, 6)); // x8 = x7 + x6
    let back = body_start - p.len_bytes();
    p.push32(bnez(5, back)); // loop while x5 != 0
    p.push_exit();

    Workload {
        program: p,
        // 2 setup instructions, 5 per iteration, then the 3-instruction exit.
        instructions: 2 + 5 * iterations as u64 + 3,
    }
}

/// `iterations` passes of a load, a store, two pointer bumps and the branch.
fn memcpy_loop(iterations: i32) -> Workload {
    let mut p = Program::default();
    p.push_pair(li(5, iterations)); // x5 = iterations (2 instructions)
    p.push32(addi(10, 0, 0)); // x10 = src cursor
    p.push32(addi(11, 0, 0)); // x11 = dst cursor
                              // The bases do not fit in a 12-bit immediate, so build them with shifts.
    p.push32(addi(12, 0, (SRC_BASE >> 12) as i32)); // x12 = SRC_BASE >> 12
    p.push32(slli(12, 12, 12)); // x12 = SRC_BASE
    p.push32(addi(13, 0, (DST_BASE >> 12) as i32)); // x13 = DST_BASE >> 12
    p.push32(slli(13, 13, 12)); // x13 = DST_BASE

    let body_start = p.len_bytes();
    p.push32(add(14, 12, 10)); // x14 = src + cursor
    p.push32(lw(15, 14, 0)); // x15 = [x14]
    p.push32(add(16, 13, 11)); // x16 = dst + cursor
    p.push32(sw(16, 15, 0)); // [x16] = x15
    p.push32(addi(10, 10, 4)); // src cursor += 4
    p.push32(addi(11, 11, 4)); // dst cursor += 4
    p.push32(addi(5, 5, -1)); // x5 -= 1
    let back = body_start - p.len_bytes();
    p.push32(bnez(5, back));
    p.push_exit();

    Workload {
        program: p,
        // 8 setup instructions, 8 per iteration, then the 3-instruction exit.
        instructions: 8 + 8 * iterations as u64 + 3,
    }
}

/// `iterations` passes of a body that interleaves 16-bit and 32-bit encodings,
/// so every iteration crosses the compressed/uncompressed fetch split twice.
fn mixed_loop(iterations: i32) -> Workload {
    let mut p = Program::default();
    p.push_pair(li(5, iterations)); // x5 = iterations (2 instructions)

    let body_start = p.len_bytes();
    p.push16(c_addi(6, 1)); // c.addi x6, 1     (16-bit)
    p.push32(xori(7, 7, 0x3C3)); // xori x7, x7, 0x3C3  (32-bit)
    p.push16(c_addi(8, -1)); // c.addi x8, -1    (16-bit)
    p.push32(add(9, 7, 6)); // add x9, x7, x6      (32-bit)
    p.push16(c_addi(5, -1)); // c.addi x5, -1    (16-bit)
    let back = body_start - p.len_bytes();
    p.push32(bnez(5, back)); // 32-bit
    p.push_exit();

    Workload {
        program: p,
        // 2 setup instructions, 6 per iteration, then the 3-instruction exit.
        instructions: 2 + 6 * iterations as u64 + 3,
    }
}

/// Run one workload to completion and return the instructions retired.
///
/// The budget is `u32::MAX`, so the whole workload runs inside one slice: the
/// benchmark measures the inner loop, not the slice scheduler.
fn run_workload(workload: &Workload) -> u64 {
    let mut cpu = Cpu::new();
    cpu.pc = PROGRAM_BASE;
    let mut mem = Memory::new();
    workload.program.load(&mut mem);

    let outcome = cpu.run_slice(&mut mem, u32::MAX);

    // The Criterion throughput figures are instructions per second only if the
    // declared count is the count the CPU really retired. An encoding mistake
    // that shortened a loop would otherwise show up as a free speedup.
    assert_eq!(
        outcome.steps, workload.instructions,
        "workload retired {} instructions, not the declared {}",
        outcome.steps, workload.instructions
    );
    assert!(cpu.is_halted, "workload did not reach its exit syscall");

    outcome.steps
}

fn bench_throughput(c: &mut Criterion) {
    // Enough iterations that the loop dominates the `Memory::new` allocation in
    // each sample, and few enough that a sample stays in the millisecond range.
    const ITERATIONS: i32 = 20_000;

    let cases: [(&str, Workload); 3] = [
        ("alu_loop", alu_loop(ITERATIONS)),
        ("memcpy_loop", memcpy_loop(ITERATIONS)),
        ("mixed_loop", mixed_loop(ITERATIONS)),
    ];

    let mut group = c.benchmark_group("cpu_throughput");
    for (name, workload) in &cases {
        // Reporting elements makes Criterion print instructions per second
        // directly, which is the number the optimisation work is chasing.
        group.throughput(Throughput::Elements(workload.instructions));
        group.bench_function(*name, |b| {
            b.iter(|| black_box(run_workload(black_box(workload))))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
