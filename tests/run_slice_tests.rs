//! Tests for the sliced run loop.
//!
//! The browser worker drives a run through `Cpu::run_slice` so it can answer
//! messages between slices. These tests pin the two properties that make that
//! safe: a slice never runs past its budget, and slicing does not change the
//! machine state a whole run produces.

use riscv_rs::{Cpu, Memory, MemoryOps, SliceStatus};

const PROGRAM_BASE: u32 = 0x1000;

fn encode_i(opcode: u32, rd: usize, funct3: u32, rs1: usize, imm: i32) -> u32 {
    (((imm as u32) & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
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

/// `bne rs1, x0, offset`
fn bnez(rs1: usize, offset: i32) -> u32 {
    encode_b(0x63, 1, rs1, 0, offset)
}

const ECALL: u32 = 0x0000_0073;

fn load(mem: &mut Memory, words: &[u32]) {
    for (i, word) in words.iter().enumerate() {
        mem.write_u32(PROGRAM_BASE + (i as u32) * 4, *word);
    }
}

fn new_cpu() -> Cpu {
    let mut cpu = Cpu::new();
    cpu.pc = PROGRAM_BASE;
    cpu
}

/// A loop that never leaves: `addi x1, x1, 1` followed by a branch back to it.
fn endless_loop() -> Vec<u32> {
    vec![addi(1, 1, 1), addi(5, 0, 1), bnez(5, -8)]
}

/// Counts x5 down from `iterations` and then exits with `exit_code`.
fn counting_program(iterations: i32, exit_code: i32) -> Vec<u32> {
    vec![
        addi(5, 0, iterations),
        addi(5, 5, -1),
        bnez(5, -4),
        addi(17, 0, 93), // a7 = SYS_exit
        addi(10, 0, exit_code),
        ECALL,
    ]
}

#[test]
fn a_slice_stops_at_its_budget_and_reports_running() {
    let mut cpu = new_cpu();
    let mut mem = Memory::new();
    load(&mut mem, &endless_loop());

    let outcome = cpu.run_slice(&mut mem, 100);

    assert_eq!(outcome.status, SliceStatus::Running);
    assert_eq!(outcome.steps, 100);
    assert_eq!(cpu.step_counter, 100);
    assert!(!cpu.is_halted);
}

#[test]
fn slicing_one_instruction_at_a_time_matches_a_whole_run() {
    let program = counting_program(25, 7);

    let mut whole_cpu = new_cpu();
    let mut whole_mem = Memory::new();
    load(&mut whole_mem, &program);
    let whole_exit = whole_cpu.run(&mut whole_mem);

    let mut sliced_cpu = new_cpu();
    let mut sliced_mem = Memory::new();
    load(&mut sliced_mem, &program);
    let mut slices = 0;
    loop {
        let outcome = sliced_cpu.run_slice(&mut sliced_mem, 1);
        slices += 1;
        assert!(slices < 10_000, "the sliced run did not terminate");
        if outcome.status != SliceStatus::Running {
            break;
        }
    }

    assert_eq!(sliced_cpu.regs, whole_cpu.regs);
    assert_eq!(sliced_cpu.pc, whole_cpu.pc);
    assert_eq!(sliced_cpu.exit_code, whole_cpu.exit_code);
    assert_eq!(sliced_cpu.exit_code, whole_exit);
    assert_eq!(sliced_cpu.exit_code, 7);
    assert_eq!(sliced_cpu.step_counter, whole_cpu.step_counter);
    assert!(sliced_cpu.is_halted && whole_cpu.is_halted);
}

#[test]
fn a_slice_reports_a_halt() {
    let mut cpu = new_cpu();
    let mut mem = Memory::new();
    load(&mut mem, &counting_program(3, 0));

    let outcome = cpu.run_slice(&mut mem, 1000);

    assert_eq!(outcome.status, SliceStatus::Halted);
    assert_eq!(outcome.exit_code, 0);
    assert!(cpu.is_halted);
}

#[test]
fn a_slice_reports_a_trap_on_an_illegal_instruction() {
    let mut cpu = new_cpu();
    let mut mem = Memory::new();
    // The all-zero halfword is the canonical illegal instruction, and memory
    // reads back as zero where nothing was written.
    load(&mut mem, &[addi(1, 0, 1), 0x0000_0000]);

    let outcome = cpu.run_slice(&mut mem, 1000);

    assert_eq!(outcome.status, SliceStatus::Trapped);
    assert_eq!(outcome.pc, PROGRAM_BASE + 4);
    assert!(cpu.trapped);
    assert_ne!(outcome.exit_code, 0);
}

#[test]
fn a_slice_reports_a_breakpoint_and_then_runs_past_it() {
    let mut cpu = new_cpu();
    let mut mem = Memory::new();
    load(&mut mem, &counting_program(3, 5));

    let breakpoint = PROGRAM_BASE + 4;
    cpu.debug_enabled = true;
    cpu.breakpoints.insert(breakpoint);

    let outcome = cpu.run_slice(&mut mem, 1000);
    assert_eq!(outcome.status, SliceStatus::Breakpoint);
    assert_eq!(outcome.pc, breakpoint);
    assert_eq!(outcome.steps, 1);
    assert!(!cpu.is_halted);

    // The next slice must make progress rather than report the same breakpoint
    // for ever; the loop body comes back to it, so it stops there again.
    let outcome = cpu.run_slice(&mut mem, 1000);
    assert_eq!(outcome.status, SliceStatus::Breakpoint);
    assert!(outcome.steps > 0);
}

#[test]
fn stepping_leaves_a_breakpoint_it_already_reported() {
    let mut cpu = new_cpu();
    let mut mem = Memory::new();
    load(&mut mem, &counting_program(3, 5));

    let breakpoint = PROGRAM_BASE + 4;
    cpu.debug_enabled = true;
    cpu.breakpoints.insert(breakpoint);

    // Run up to the breakpoint.
    assert_eq!(
        cpu.run_slice(&mut mem, 1000).status,
        SliceStatus::Breakpoint
    );
    assert_eq!(cpu.pc, breakpoint);

    // A step from the breakpoint address must execute the instruction rather
    // than report the same hit again.
    let before = cpu.step_counter;
    assert_eq!(
        riscv_rs::StepResult::Ok,
        cpu.step_instruction(&mut mem),
        "the step reported the breakpoint instead of leaving it"
    );
    assert_eq!(cpu.step_counter, before + 1);
    assert_ne!(cpu.pc, breakpoint);
}

#[test]
fn a_breakpoint_is_ignored_when_debug_mode_is_off() {
    let mut cpu = new_cpu();
    let mut mem = Memory::new();
    load(&mut mem, &counting_program(3, 5));
    cpu.breakpoints.insert(PROGRAM_BASE + 4);

    let outcome = cpu.run_slice(&mut mem, 1000);

    assert_eq!(outcome.status, SliceStatus::Halted);
    assert_eq!(outcome.exit_code, 5);
}

/// A 32-bit instruction whose two halves live in different pages.
///
/// Instruction fetch reads one four-byte window with a single page lookup. That
/// window cannot span two pages, so the fetch reports itself as partial and the
/// CPU re-reads the full word. If it ever stopped doing that, the upper half of
/// the instruction would read as zero and the decode would be wrong.
#[test]
fn a_32_bit_instruction_across_a_page_boundary_executes() {
    const PAGE_SIZE: u32 = 65536;
    // The last halfword of page 0: the instruction's upper half is in page 1.
    let straddling = PAGE_SIZE - 2;

    let mut cpu = Cpu::new();
    cpu.pc = straddling;
    let mut mem = Memory::new();

    // `addi x7, x0, 42` written so that it crosses the page edge, followed by
    // the exit sequence on the far side.
    let inst = addi(7, 0, 42);
    mem.write_u16(straddling, inst as u16);
    mem.write_u16(straddling + 2, (inst >> 16) as u16);
    mem.write_u32(straddling + 4, addi(17, 0, 93)); // a7 = SYS_exit
    mem.write_u32(straddling + 8, addi(10, 0, 7)); // a0 = 7
    mem.write_u32(straddling + 12, ECALL);

    let outcome = cpu.run_slice(&mut mem, 100);

    assert_eq!(
        cpu.regs[7], 42,
        "the straddling instruction decoded to the wrong value"
    );
    assert_eq!(outcome.status, SliceStatus::Halted);
    assert_eq!(outcome.exit_code, 7);
}
