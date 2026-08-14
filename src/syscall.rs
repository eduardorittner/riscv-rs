use crate::cpu::{Cpu, A0, A1, A2, A3, A7};
use crate::host_imports;
use crate::memory::MemoryOps;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Unknown syscall number along with the passed arguments
pub struct UnknownSyscall {
    pub sys_num: u32,
    pub a0: u32,
    pub a1: u32,
    pub a2: u32,
    pub a3: u32,
}

pub fn handle_ecall<M: MemoryOps>(cpu: &mut Cpu, mem: &mut M) -> Result<(), UnknownSyscall> {
    let a7 = cpu.read_reg(A7); // Syscall number
    let a0 = cpu.read_reg(A0); // Arg 0 / Return val
    let a1 = cpu.read_reg(A1); // Arg 1
    let a2 = cpu.read_reg(A2); // Arg 2
    let a3 = cpu.read_reg(A3); // Arg 3

    // Try host custom syscall handler first
    if let Ok(res) =
        host_imports::custom_syscall(a0 as i32, a1 as i32, a2 as i32, a3 as i32, a7 as i32)
    {
        if res != 0 {
            cpu.write_reg(A0, res as u32);
            return Ok(());
        }
    }

    // Fall back to our own implementation
    match a7 {
        // SYS_exit
        93 | 10 | 1 => {
            cpu.is_halted = true;
            cpu.exit_code = a0 as i32;
            Ok(())
        }

        // SYS_write
        64 | 4 => {
            let fd = a0;
            let buf_ptr = a1;
            let count = (a2 as usize).min(16 * 1024 * 1024);
            let mut bytes = mem.read_bytes(buf_ptr, count);
            while bytes.last() == Some(&0) {
                bytes.pop();
            }
            let text = String::from_utf8_lossy(&bytes);

            if fd == 1 {
                host_imports::js_print(&text);
            } else if fd == 2 {
                host_imports::js_print_err(&text);
            }
            cpu.write_reg(10, a2); // Return number of bytes written
            Ok(())
        }

        // SYS_read
        63 | 3 => {
            let _fd = a0;
            let buf_ptr = a1;
            let count = (a2 as usize).min(16 * 1024 * 1024);

            let mut scratch = vec![0u8; count];
            let bytes_read = host_imports::read_from_stdin(scratch.as_mut_ptr(), count as u32);
            if bytes_read > 0 {
                mem.write_bytes(buf_ptr, &scratch[..bytes_read as usize]);
                cpu.write_reg(10, bytes_read as u32);
            } else {
                cpu.write_reg(10, 0);
            }
            Ok(())
        }

        // SYS_brk
        214 | 45 => {
            if a0 != 0 {
                mem.set_brk(a0);
            }
            cpu.write_reg(10, mem.get_brk());
            Ok(())
        }

        // SYS_close (57), SYS_lseek (62), SYS_fstat (80)
        57 | 62 | 80 => {
            cpu.write_reg(10, 0);
            Ok(())
        }

        _ => Err(UnknownSyscall {
            sys_num: a7,
            a0,
            a1,
            a2,
            a3,
        }),
    }
}
