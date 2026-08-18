//! Tests for state that has to survive `load_binary`.
//!
//! `load_binary` builds a fresh `Cpu`, which used to drop every setting the
//! host had already applied. A device registers its syscalls before a binary
//! exists, so the custom-syscall flag has to cross the reset.

use riscv_rs::Simulator;

#[test]
fn the_custom_syscall_flag_survives_load_binary() {
    let mut sim = Simulator::new();
    sim.set_has_custom_syscalls(true);

    sim.load_binary_with_args(&[], &[]);

    assert!(
        sim.has_custom_syscalls(),
        "a syscall registered before the first run must still be active in it"
    );
}

#[test]
fn load_binary_does_not_invent_a_custom_syscall_flag() {
    let mut sim = Simulator::new();

    sim.load_binary_with_args(&[], &[]);

    assert!(!sim.has_custom_syscalls());
}
