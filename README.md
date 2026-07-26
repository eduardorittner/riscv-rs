# rust-whisper

`rust-whisper` is a high-performance RISC-V emulator written in Rust, compiled to WebAssembly (Wasm) as a direct drop-in replacement for `whisper.js` in the RISC-V ALE project.

## Target RISC-V ISA Subset

The emulator targets **RV32IMAFD / RV32G** with **RV32C** compressed instruction support, using the **`ilp32d`** ABI:

- **State**: 32 × 32-bit integer registers (`x0`–`x31`, `x0 = 0`) and 32 × 64-bit double-precision floating-point registers (`f0`–`f31`).
- **RV32I**: Base integer arithmetic, logic, shifts, loads, stores, jumps (`jal`, `jalr`), and branches (`beq`, `bne`, `blt`, `bge`, `bltu`, `bgeu`).
- **RV32M**: Hardware multiplication (`mul`, `mulh`, `mulhsu`, `mulhu`) and division/remainder (`div`, `divu`, `rem`, `remu`).
- **RV32A**: 32-bit atomic operations (`lr.w`, `sc.w`, `amoswap.w`, `amoadd.w`, `amoxor.w`, `amoand.w`, `amoor.w`, `amomin.w`/`max`).
- **RV32F & RV32D**: Single (`.s`) and double (`.d`) precision floating-point operations, conversions, sign injection (`fsgnj`), comparisons, and fused multiply-add (`fmadd`, `fmsub`, etc.).
- **RV32C**: 16-bit compressed instructions (`C.LI`, `C.MV`, `C.ADD`, `C.J`, `C.LW`, `C.SW`, `C.BEQZ`, `C.BNEZ`, `C.SLLI`, `C.LWSP`, `C.SWSP`).
- **CSRs & Privileges**: `fcsr`, `mstatus`, `mepc`, `mcause`, `mret` exception returns, and `ecall` system calls (trapping to host JS).

## WebAssembly Compilation

Compiled to WebAssembly via `wasm-bindgen` / `wasm-pack`:

```bash
wasm-pack build --target web --release
```

## Testing

Run unit, instruction, and property-based differential tests against the SweRV-ISS Whisper C++ oracle:

```bash
cargo test
```
