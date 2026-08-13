use proptest::prelude::*;
use riscv_rs::host_imports::{
    get_mock_stderr, get_mock_stdout, reset_mocks, set_mock_custom_syscall, set_mock_stdin,
};
use riscv_rs::syscall::handle_ecall;
use riscv_rs::{Cpu, Memory, MemoryOps};
use std::panic::{catch_unwind, AssertUnwindSafe};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Task 2.1: Arbitrary Syscall Exception Safety Fuzzer
    /// Verifies that no arbitrary combination of syscall ID and argument registers triggers a host panic.
    #[test]
    fn fuzz_syscall_panic_safety(
        sys_num in 0..1000u32,
        arg0 in 0..u32::MAX,
        arg1 in 0..u32::MAX,
        arg2 in 0..u32::MAX,
        arg3 in 0..u32::MAX,
    ) {
        reset_mocks();
        let mut cpu = Cpu::new();
        let mut mem = Memory::new();

        cpu.write_reg(17, sys_num);
        cpu.write_reg(10, arg0);
        cpu.write_reg(11, arg1);
        cpu.write_reg(12, arg2);
        cpu.write_reg(13, arg3);

        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = handle_ecall(&mut cpu, &mut mem);
        }));
        prop_assert!(res.is_ok(), "Host panicked during syscall {}", sys_num);
    }

    /// Task 2.2: SYS_write Boundary & Buffer Fuzzer
    /// Verifies that SYS_write handles varied file descriptors, buffer pointers, and counts safely.
    #[test]
    fn fuzz_sys_write_boundaries(
        fd in prop_oneof![Just(0u32), Just(1u32), Just(2u32), Just(3u32), Just(64u32), Just(u32::MAX)],
        buf_ptr in prop_oneof![0..10u32, 0xFFF0..0x10010u32, 0xFFFF0000..0xFFFF0010u32, 0xFFFFFFF0..u32::MAX],
        count in 0..100_000usize,
        data in proptest::collection::vec(any::<u8>(), 0..1024),
    ) {
        reset_mocks();
        let mut cpu = Cpu::new();
        let mut mem = Memory::new();

        if !data.is_empty() {
            mem.write_bytes(buf_ptr, &data);
        }

        cpu.write_reg(17, 64); // SYS_write
        cpu.write_reg(10, fd);
        cpu.write_reg(11, buf_ptr);
        cpu.write_reg(12, count as u32);

        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = handle_ecall(&mut cpu, &mut mem);
        }));
        prop_assert!(res.is_ok(), "SYS_write panicked");

        let ret_val = cpu.read_reg(10);
        prop_assert_eq!(ret_val, count as u32);

        if fd == 1 {
            let stdout_logs = get_mock_stdout();
            prop_assert_eq!(stdout_logs.len(), 1);
        } else if fd == 2 {
            let stderr_logs = get_mock_stderr();
            prop_assert_eq!(stderr_logs.len(), 1);
        } else {
            prop_assert!(get_mock_stdout().is_empty());
            prop_assert!(get_mock_stderr().is_empty());
        }
    }

    /// Task 2.3: SYS_read Stdin & Buffer Fuzzer
    /// Verifies that SYS_read transfers stdin data correctly to RAM or MMIO memory without panic.
    #[test]
    fn fuzz_sys_read_boundaries(
        stdin_bytes in proptest::collection::vec(any::<u8>(), 0..1000),
        buf_ptr in prop_oneof![0..10u32, 0x10000..0x10020u32, 0xFFFF0000..0xFFFF0010u32, 0xFFFFFFF0..u32::MAX],
        count in 0..100_000u32,
    ) {
        reset_mocks();
        set_mock_stdin(&stdin_bytes);

        let mut cpu = Cpu::new();
        let mut mem = Memory::new();

        cpu.write_reg(17, 63); // SYS_read
        cpu.write_reg(10, 0);  // fd = stdin
        cpu.write_reg(11, buf_ptr);
        cpu.write_reg(12, count);

        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = handle_ecall(&mut cpu, &mut mem);
        }));
        prop_assert!(res.is_ok(), "SYS_read panicked");

        let bytes_read = cpu.read_reg(10);
        let expected_read = (stdin_bytes.len() as u32).min(count);
        prop_assert_eq!(bytes_read, expected_read);

        if bytes_read > 0 && buf_ptr < 0xFFFF0000 {
            let mem_bytes = mem.read_bytes(buf_ptr, bytes_read as usize);
            prop_assert_eq!(mem_bytes, &stdin_bytes[..bytes_read as usize]);
        }
    }

    /// Task 2.4: SYS_brk Heap Invariant Fuzzer
    /// Verifies heap break pointer invariants across random sequences of SYS_brk requests.
    #[test]
    fn fuzz_sys_brk_invariants(
        req_addrs in proptest::collection::vec(any::<u32>(), 1..20),
    ) {
        reset_mocks();
        let mut cpu = Cpu::new();
        let mut mem = Memory::new();
        let initial_brk = mem.get_brk();
        let mut current_brk = initial_brk;

        for req_a0 in req_addrs {
            cpu.write_reg(17, 214); // SYS_brk
            cpu.write_reg(10, req_a0);

            let res = catch_unwind(AssertUnwindSafe(|| {
                let _ = handle_ecall(&mut cpu, &mut mem);
            }));
            prop_assert!(res.is_ok(), "SYS_brk panicked");

            let new_brk = cpu.read_reg(10);

            if req_a0 == 0 {
                prop_assert_eq!(new_brk, current_brk);
                prop_assert_eq!(mem.get_brk(), current_brk);
            } else if req_a0 >= initial_brk && req_a0 < 0xFFFF0000 {
                current_brk = req_a0;
                prop_assert_eq!(new_brk, current_brk);
                prop_assert_eq!(mem.get_brk(), current_brk);
            } else {
                prop_assert_eq!(new_brk, current_brk);
                prop_assert_eq!(mem.get_brk(), current_brk);
            }
        }
    }

    /// Task 2.5: Custom Syscall Host Hook Fuzzer
    /// Verifies that custom_syscall hook intercept behavior correctly returns non-zero status or falls through.
    #[test]
    fn fuzz_custom_syscall_hook(
        hook_return in any::<i32>(),
        sys_num in 0..100u32,
    ) {
        reset_mocks();
        set_mock_custom_syscall(move |_a0, _a1, _a2, _a3, _a7| Ok(hook_return));

        let mut cpu = Cpu::new();
        let mut mem = Memory::new();

        cpu.write_reg(17, sys_num);
        cpu.write_reg(10, 42);

        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = handle_ecall(&mut cpu, &mut mem);
        }));
        prop_assert!(res.is_ok(), "custom_syscall hook panicked");

        if hook_return != 0 {
            prop_assert_eq!(cpu.read_reg(10), hook_return as u32);
        }
    }
}

#[test]
fn test_unknown_syscall_returns_error() {
    use riscv_rs::syscall::{handle_ecall, UnknownSyscall};
    use riscv_rs::{Cpu, Memory};

    let mut cpu = Cpu::new();
    let mut mem = Memory::new();

    cpu.write_reg(17, 999); // Unknown syscall number
    cpu.write_reg(10, 0x11); // a0
    cpu.write_reg(11, 0x22); // a1
    cpu.write_reg(12, 0x33); // a2
    cpu.write_reg(13, 0x44); // a3

    let result = handle_ecall(&mut cpu, &mut mem);
    assert_eq!(
        result,
        Err(UnknownSyscall {
            sys_num: 999,
            a0: 0x11,
            a1: 0x22,
            a2: 0x33,
            a3: 0x44,
        })
    );
}
