use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use riscv_rs::memory::{Memory, MemoryOps};
use std::collections::HashMap;

/// An alternative contiguous implementer of `MemoryOps` backed by a flat `Vec<u8>`.
pub struct FlatMemory {
    data: Vec<u8>,
    brk_ptr: u32,
}

impl FlatMemory {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
            brk_ptr: 0x1000000,
        }
    }
}

impl MemoryOps for FlatMemory {
    fn read_u8(&self, addr: u32) -> u8 {
        self.data.get(addr as usize).copied().unwrap_or(0)
    }

    fn write_u8(&mut self, addr: u32, val: u8) {
        let idx = addr as usize;
        if idx < self.data.len() {
            self.data[idx] = val;
        }
    }

    fn get_brk(&self) -> u32 {
        self.brk_ptr
    }

    fn set_brk(&mut self, val: u32) {
        self.brk_ptr = val;
    }
}

/// An alternative sparse implementer of `MemoryOps` backed by a `HashMap`.
pub struct HashMapMemory {
    storage: HashMap<u32, u8>,
    brk_ptr: u32,
}

impl Default for HashMapMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl HashMapMemory {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
            brk_ptr: 0x1000000,
        }
    }
}

impl MemoryOps for HashMapMemory {
    fn read_u8(&self, addr: u32) -> u8 {
        self.storage.get(&addr).copied().unwrap_or(0)
    }

    fn write_u8(&mut self, addr: u32, val: u8) {
        self.storage.insert(addr, val);
    }

    fn get_brk(&self) -> u32 {
        self.brk_ptr
    }

    fn set_brk(&mut self, val: u32) {
        self.brk_ptr = val;
    }
}

/// Generic benchmark matrix for any implementer of `MemoryOps`.
/// Covers independent options:
/// 1. Access Type: Only Reads / Only Writes / Reads & Writes
/// 2. Access Pattern: Sequential Access / Random Access
fn bench_memory_matrix<M: MemoryOps, F: Fn() -> M>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    impl_name: &str,
    create_mem: F,
) {
    let payload_64k = vec![0xABu8; 65536];
    let payload_256b = vec![0xCDu8; 256];

    // Pre-generate 1024 deterministic pseudo-random addresses in range [0, 16MB - 64KB]
    let random_addrs: Vec<u32> = (0..1024u32)
        .map(|i| (i.wrapping_mul(2654435761) % (16 * 1024 * 1024 - 65536)) & !3)
        .collect();

    // -------------------------------------------------------------
    // Scenario 1: Sequential Access (Contiguous Blocks / Strides)
    // -------------------------------------------------------------

    // 1a. Sequential Read Only (bulk bytes)
    group.bench_function(
        BenchmarkId::new("sequential/read_only_bulk_64k", impl_name),
        |b| {
            let mut mem = create_mem();
            mem.write_bytes(0x10000, &payload_64k);
            b.iter(|| {
                let res = mem.read_bytes(0x10000, 65536);
                black_box(res)
            });
        },
    );

    // 1b. Sequential Write Only (bulk bytes)
    group.bench_function(
        BenchmarkId::new("sequential/write_only_bulk_64k", impl_name),
        |b| {
            let mut mem = create_mem();
            b.iter(|| {
                mem.write_bytes(0x10000, black_box(&payload_64k));
            });
        },
    );

    // 1c. Sequential Read-Write (bulk bytes)
    group.bench_function(
        BenchmarkId::new("sequential/read_write_bulk_64k", impl_name),
        |b| {
            let mut mem = create_mem();
            b.iter(|| {
                mem.write_bytes(0x10000, black_box(&payload_64k));
                let res = mem.read_bytes(0x10000, 65536);
                black_box(res)
            });
        },
    );

    // -------------------------------------------------------------
    // Scenario 2: Random Access (Scattered Addresses Across Memory)
    // -------------------------------------------------------------

    // 2a. Random Read Only (bulk 256B chunks)
    group.bench_function(
        BenchmarkId::new("random/read_only_bulk_256b", impl_name),
        |b| {
            let mut mem = create_mem();
            for &addr in &random_addrs {
                mem.write_bytes(addr, &payload_256b);
            }
            let mut idx = 0;
            b.iter(|| {
                let addr = random_addrs[idx % random_addrs.len()];
                idx += 1;
                let res = mem.read_bytes(addr, 256);
                black_box(res)
            });
        },
    );

    // 2b. Random Write Only (bulk 256B chunks)
    group.bench_function(
        BenchmarkId::new("random/write_only_bulk_256b", impl_name),
        |b| {
            let mut mem = create_mem();
            let mut idx = 0;
            b.iter(|| {
                let addr = random_addrs[idx % random_addrs.len()];
                idx += 1;
                mem.write_bytes(addr, black_box(&payload_256b));
            });
        },
    );

    // 2c. Random Read-Write (bulk 256B chunks)
    group.bench_function(
        BenchmarkId::new("random/read_write_bulk_256b", impl_name),
        |b| {
            let mut mem = create_mem();
            let mut idx = 0;
            b.iter(|| {
                let addr = random_addrs[idx % random_addrs.len()];
                idx += 1;
                mem.write_bytes(addr, black_box(&payload_256b));
                let res = mem.read_bytes(addr, 256);
                black_box(res)
            });
        },
    );

    // -------------------------------------------------------------
    // Word Accesses (Sequential vs Random)
    // -------------------------------------------------------------

    group.bench_function(
        BenchmarkId::new("sequential/read_write_u32", impl_name),
        |b| {
            let mut mem = create_mem();
            let mut addr = 0u32;
            b.iter(|| {
                mem.write_u32(addr, black_box(0x12345678));
                let val = mem.read_u32(addr);
                addr = (addr + 4) % 0x100000;
                black_box(val)
            });
        },
    );

    group.bench_function(BenchmarkId::new("random/read_write_u32", impl_name), |b| {
        let mut mem = create_mem();
        let mut idx = 0;
        b.iter(|| {
            let addr = random_addrs[idx % random_addrs.len()];
            idx += 1;
            mem.write_u32(addr, black_box(0x87654321));
            let val = mem.read_u32(addr);
            black_box(val)
        });
    });
}

fn bench_all_memory_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("MemoryOps Matrix");

    // Benchmark current implementer (Paged Memory)
    bench_memory_matrix(&mut group, "PagedMemory (Current)", Memory::new);

    // Benchmark alternative implementers
    bench_memory_matrix(&mut group, "FlatMemory", || FlatMemory::new(0x2000000));
    bench_memory_matrix(&mut group, "HashMapMemory", HashMapMemory::new);

    group.finish();
}

criterion_group!(benches, bench_all_memory_matrix);
criterion_main!(benches);
