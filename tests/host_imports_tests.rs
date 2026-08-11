use rust_whisper::host_imports::{
    get_mock_stderr, get_mock_stdout, js_print, js_print_err, js_read_mmio, js_write_mmio,
    read_from_stdin, reset_mocks, set_mock_mmio_read, set_mock_mmio_write, set_mock_stdin,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

#[test]
fn test_js_print_captures() {
    reset_mocks();

    js_print("Hello stdout");
    js_print("Second line");
    js_print_err("Hello stderr");

    let stdout = get_mock_stdout();
    let stderr = get_mock_stderr();

    assert_eq!(stdout, vec!["Hello stdout", "Second line"]);
    assert_eq!(stderr, vec!["Hello stderr"]);
}

#[test]
fn test_read_from_stdin_js() {
    reset_mocks();

    let input_bytes = b"Hello stdin input stream\n";
    set_mock_stdin(input_bytes);

    let mut buf = vec![0u8; 32];
    let n1 = read_from_stdin(buf.as_mut_ptr(), 11);
    assert_eq!(n1, 11);
    assert_eq!(&buf[..11], b"Hello stdin");

    let n2 = read_from_stdin(buf.as_mut_ptr(), 100);
    assert_eq!(n2, 14); // Remaining bytes in stdin stream
    assert_eq!(&buf[..14], b" input stream\n");

    let n3 = read_from_stdin(buf.as_mut_ptr(), 10);
    assert_eq!(n3, 0); // Stdin EOF
}

#[test]
fn test_js_mmio_dispatch() {
    reset_mocks();

    let last_written_addr = Arc::new(AtomicU32::new(0));
    let last_written_val = Arc::new(AtomicU32::new(0));

    let addr_clone = Arc::clone(&last_written_addr);
    let val_clone = Arc::clone(&last_written_val);

    set_mock_mmio_write(move |addr, size, val| {
        addr_clone.store(addr | (size << 28), Ordering::SeqCst);
        val_clone.store(val, Ordering::SeqCst);
    });

    set_mock_mmio_read(|addr, size| match (addr, size) {
        (0xFFFF0000, 1) => 0xAB,
        (0xFFFF0004, 2) => 0x1234,
        (0xFFFF0008, 4) => 0xDEADBEEF,
        _ => 0,
    });

    // Verify MMIO Read Dispatch across sizes (1, 2, 4 bytes)
    assert_eq!(js_read_mmio(0xFFFF0000, 1), 0xAB);
    assert_eq!(js_read_mmio(0xFFFF0004, 2), 0x1234);
    assert_eq!(js_read_mmio(0xFFFF0008, 4), 0xDEADBEEF);

    // Verify MMIO Write Dispatch
    js_write_mmio(0xFFFF0010, 4, 0xCAFEBABE);
    assert_eq!(
        last_written_addr.load(Ordering::SeqCst),
        0xFFFF0010 | (4 << 28)
    );
    assert_eq!(last_written_val.load(Ordering::SeqCst), 0xCAFEBABE);
}
