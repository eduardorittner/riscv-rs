use crate::host_imports;

const PAGE_SIZE: usize = 65536; // 64 KB per page
const PAGE_MASK: u32 = 0xFFFF;
const PAGE_SHIFT: u32 = 16;
const MMIO_BASE: u32 = 0xFFFF0000;

/// Trait defining memory operations for RISC-V simulation.
pub trait MemoryOps {
    fn read_u8(&self, addr: u32) -> u8;
    fn write_u8(&mut self, addr: u32, val: u8);

    fn read_u16(&self, addr: u32) -> u16 {
        let b0 = self.read_u8(addr) as u16;
        let b1 = self.read_u8(addr.wrapping_add(1)) as u16;
        b0 | (b1 << 8)
    }

    fn write_u16(&mut self, addr: u32, val: u16) {
        let bytes = val.to_le_bytes();
        self.write_u8(addr, bytes[0]);
        self.write_u8(addr.wrapping_add(1), bytes[1]);
    }

    fn read_u32(&self, addr: u32) -> u32 {
        let b0 = self.read_u8(addr) as u32;
        let b1 = self.read_u8(addr.wrapping_add(1)) as u32;
        let b2 = self.read_u8(addr.wrapping_add(2)) as u32;
        let b3 = self.read_u8(addr.wrapping_add(3)) as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    fn write_u32(&mut self, addr: u32, val: u32) {
        let bytes = val.to_le_bytes();
        self.write_u8(addr, bytes[0]);
        self.write_u8(addr.wrapping_add(1), bytes[1]);
        self.write_u8(addr.wrapping_add(2), bytes[2]);
        self.write_u8(addr.wrapping_add(3), bytes[3]);
    }

    fn read_bytes(&self, addr: u32, len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; len];
        for (i, item) in buf.iter_mut().enumerate() {
            *item = self.read_u8(addr.wrapping_add(i as u32));
        }
        buf
    }

    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) {
        for (i, &b) in bytes.iter().enumerate() {
            self.write_u8(addr.wrapping_add(i as u32), b);
        }
    }

    /// Fetch the instruction window at `addr` with a single page lookup.
    ///
    /// The low 16 bits of the returned word are always the halfword at `addr`,
    /// which is what decides whether the instruction is compressed. The second
    /// element says whether the high 16 bits are the following halfword.
    ///
    /// It is false in two cases, and the caller must then read the upper half
    /// separately: at a page edge, where the four-byte window straddles two
    /// pages, and next to the MMIO window, where a blind four-byte read would
    /// perform a device read with side effects that the guest never asked for.
    ///
    /// Instruction fetch used to cost two reads for every 32-bit instruction —
    /// a `read_u16` to classify it and a `read_u32` to decode it — and each one
    /// repeated the MMIO bounds check and the page lookup.
    fn fetch_window(&self, addr: u32) -> (u32, bool) {
        (self.read_u32(addr), true)
    }

    fn get_brk(&self) -> u32 {
        0
    }

    fn set_brk(&mut self, _val: u32) {}
}

/// Lazy-initialized sequential list of memory pages.
///
/// Pages are only allocated on first access.
pub struct Memory {
    pages: Vec<Option<Box<[u8; PAGE_SIZE]>>>,
    pub brk_ptr: u32,
    pub initial_brk: u32,
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory {
    pub fn new() -> Self {
        let mut pages = Vec::with_capacity(65536);
        pages.resize_with(65536, || None);
        let initial_brk = 0x1000000; // 16 MB default heap start
        Self {
            pages,
            brk_ptr: initial_brk,
            initial_brk,
        }
    }

    #[inline(always)]
    fn get_or_create_page(&mut self, page_idx: usize) -> &mut [u8; PAGE_SIZE] {
        if self.pages[page_idx].is_none() {
            self.pages[page_idx] = Some(Box::new([0u8; PAGE_SIZE]));
        }
        match &mut self.pages[page_idx] {
            Some(page) => page.as_mut(),
            None => unreachable!(),
        }
    }

    #[inline(always)]
    fn get_page(&self, page_idx: usize) -> Option<&[u8; PAGE_SIZE]> {
        self.pages[page_idx].as_deref()
    }
}

impl MemoryOps for Memory {
    fn read_u8(&self, addr: u32) -> u8 {
        if addr >= MMIO_BASE {
            return (host_imports::js_read_mmio(addr, 1) & 0xFF) as u8;
        }
        match self.get_page(idx_of(addr)) {
            Some(page) => page[offset_of(addr)],
            // Memory is zero-initialized by default.
            None => 0,
        }
    }

    fn write_u8(&mut self, addr: u32, val: u8) {
        if addr >= MMIO_BASE {
            host_imports::js_write_mmio(addr, 1, val as u32);
            return;
        }
        let page = self.get_or_create_page(idx_of(addr));
        page[offset_of(addr)] = val;
    }

    fn read_u16(&self, addr: u32) -> u16 {
        if addr >= MMIO_BASE {
            return (host_imports::js_read_mmio(addr, 2) & 0xFFFF) as u16;
        }

        let offset = offset_of(addr);
        if addr & 1 == 0 || offset + 1 < PAGE_SIZE {
            let page_idx = idx_of(addr);
            if let Some(page) = self.get_page(page_idx) {
                return u16::from_le_bytes([page[offset], page[offset + 1]]);
            }
            return 0;
        }
        let b0 = self.read_u8(addr) as u16;
        let b1 = self.read_u8(addr.wrapping_add(1)) as u16;
        b0 | (b1 << 8)
    }

    fn write_u16(&mut self, addr: u32, val: u16) {
        if addr >= MMIO_BASE {
            host_imports::js_write_mmio(addr, 2, val as u32);
            return;
        }

        let offset = offset_of(addr);
        if addr & 1 == 0 || offset + 1 < PAGE_SIZE {
            let page_idx = (addr >> PAGE_SHIFT) as usize;
            let page = self.get_or_create_page(page_idx);
            let bytes = val.to_le_bytes();
            page[offset] = bytes[0];
            page[offset + 1] = bytes[1];
            return;
        }

        let bytes = val.to_le_bytes();
        self.write_u8(addr, bytes[0]);
        self.write_u8(addr.wrapping_add(1), bytes[1]);
    }

    fn read_u32(&self, addr: u32) -> u32 {
        if addr >= MMIO_BASE {
            return host_imports::js_read_mmio(addr, 4);
        }

        let offset = offset_of(addr);
        if addr & 3 == 0 || offset + 3 < PAGE_SIZE {
            let page_idx = idx_of(addr);
            if let Some(page) = self.get_page(page_idx) {
                return u32::from_le_bytes([
                    page[offset],
                    page[offset + 1],
                    page[offset + 2],
                    page[offset + 3],
                ]);
            }
            return 0;
        }

        let b0 = self.read_u8(addr) as u32;
        let b1 = self.read_u8(addr.wrapping_add(1)) as u32;
        let b2 = self.read_u8(addr.wrapping_add(2)) as u32;
        let b3 = self.read_u8(addr.wrapping_add(3)) as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    fn fetch_window(&self, addr: u32) -> (u32, bool) {
        let offset = offset_of(addr);
        // The whole four-byte window is inside one ordinary page: one bounds
        // check, one page lookup, one load. This is the path every instruction
        // that is not on a page edge takes.
        if addr < MMIO_BASE && offset + 3 < PAGE_SIZE {
            return match self.get_page(idx_of(addr)) {
                Some(page) => (
                    u32::from_le_bytes([
                        page[offset],
                        page[offset + 1],
                        page[offset + 2],
                        page[offset + 3],
                    ]),
                    true,
                ),
                // Unmapped pages read as zero, and the all-zero halfword is the
                // canonical illegal instruction, which is the right answer for
                // a fetch from nothing.
                None => (0, false),
            };
        }

        // Page edge, or the MMIO window. Read only the halfword that was asked
        // for: widening the read here could trigger a device read the guest
        // never performed.
        (self.read_u16(addr) as u32, false)
    }

    fn write_u32(&mut self, addr: u32, val: u32) {
        if addr >= MMIO_BASE {
            host_imports::js_write_mmio(addr, 4, val);
            return;
        }

        let offset = offset_of(addr);
        if addr & 3 == 0 || offset + 3 < PAGE_SIZE {
            let page_idx = idx_of(addr);
            let page = self.get_or_create_page(page_idx);
            let bytes = val.to_le_bytes();
            page[offset] = bytes[0];
            page[offset + 1] = bytes[1];
            page[offset + 2] = bytes[2];
            page[offset + 3] = bytes[3];
            return;
        }
        let bytes = val.to_le_bytes();
        self.write_u8(addr, bytes[0]);
        self.write_u8(addr.wrapping_add(1), bytes[1]);
        self.write_u8(addr.wrapping_add(2), bytes[2]);
        self.write_u8(addr.wrapping_add(3), bytes[3]);
    }

    fn read_bytes(&self, addr: u32, len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; len];
        if len == 0 {
            return buf;
        }

        let mut curr_addr = addr;
        let mut offset_in_buf = 0;

        // This pattern coalesces reads from the same page, instead of having to repeatedly call
        // `.read_u8` or whatnot
        while offset_in_buf < len {
            if curr_addr >= MMIO_BASE {
                for (i, item) in buf[offset_in_buf..].iter_mut().enumerate() {
                    *item = self.read_u8(curr_addr.wrapping_add(i as u32));
                }
                break;
            }

            let page_idx = idx_of(curr_addr);
            let page_offset = offset_of(curr_addr);
            let bytes_in_page = (PAGE_SIZE - page_offset).min(len - offset_in_buf);

            if let Some(page) = self.get_page(page_idx) {
                buf[offset_in_buf..offset_in_buf + bytes_in_page]
                    .copy_from_slice(&page[page_offset..page_offset + bytes_in_page]);
            } else {
                buf[offset_in_buf..offset_in_buf + bytes_in_page].fill(0);
            }

            curr_addr = curr_addr.wrapping_add(bytes_in_page as u32);
            offset_in_buf += bytes_in_page;
        }

        buf
    }

    fn write_bytes(&mut self, addr: u32, bytes: &[u8]) {
        let len = bytes.len();
        if len == 0 {
            return;
        }

        let mut curr_addr = addr;
        let mut offset_in_src = 0;

        while offset_in_src < len {
            if curr_addr >= MMIO_BASE {
                for (i, &b) in bytes[offset_in_src..].iter().enumerate() {
                    self.write_u8(curr_addr.wrapping_add(i as u32), b);
                }
                break;
            }

            let page_idx = idx_of(curr_addr);
            let page_offset = offset_of(curr_addr);
            let bytes_in_page = (PAGE_SIZE - page_offset).min(len - offset_in_src);

            let page = self.get_or_create_page(page_idx);
            page[page_offset..page_offset + bytes_in_page]
                .copy_from_slice(&bytes[offset_in_src..offset_in_src + bytes_in_page]);

            curr_addr = curr_addr.wrapping_add(bytes_in_page as u32);
            offset_in_src += bytes_in_page;
        }
    }

    fn get_brk(&self) -> u32 {
        self.brk_ptr
    }

    fn set_brk(&mut self, val: u32) {
        if val >= self.initial_brk && val < MMIO_BASE {
            self.brk_ptr = val;
        }
    }
}

#[inline(always)]
fn idx_of(addr: u32) -> usize {
    (addr >> PAGE_SHIFT) as usize
}

#[inline(always)]
fn offset_of(addr: u32) -> usize {
    (addr & PAGE_MASK) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_coalesced_bytes_operations() {
        let mut mem = Memory::new();

        // Write across page boundary (PAGE_SIZE = 65536)
        let addr = 65530; // 6 bytes before end of page 0
        let data: Vec<u8> = (0..20).map(|i| i as u8).collect();
        mem.write_bytes(addr, &data);

        // Read back entire slice across boundary
        let read_data = mem.read_bytes(addr, 20);
        assert_eq!(read_data, data);

        // Verify individual byte reads match
        for (i, &item) in data.iter().enumerate() {
            assert_eq!(mem.read_u8(addr + i as u32), item);
        }

        // Read from unallocated page
        let unalloc_read = mem.read_bytes(0x200000, 100);
        assert_eq!(unalloc_read, vec![0u8; 100]);
    }

    #[test]
    fn fetch_window_returns_a_whole_word_inside_a_page() {
        let mut mem = Memory::new();
        mem.write_u32(0x1000, 0xDEAD_BEEF);

        let (bits, wide) = mem.fetch_window(0x1000);
        assert_eq!(bits, 0xDEAD_BEEF);
        assert!(wide, "a window inside one page carries all 32 bits");
    }

    #[test]
    fn fetch_window_reports_a_partial_read_at_a_page_edge() {
        let mut mem = Memory::new();
        // The last halfword of page 0. The other half of a 32-bit instruction
        // here lives in page 1.
        let addr = PAGE_SIZE as u32 - 2;
        mem.write_u16(addr, 0xB0B5);
        mem.write_u16(addr + 2, 0xF00D);

        let (bits, wide) = mem.fetch_window(addr);
        assert_eq!(bits & 0xFFFF, 0xB0B5, "the asked-for halfword is present");
        assert!(
            !wide,
            "a window that straddles two pages must report itself as partial"
        );
    }

    #[test]
    fn fetch_window_does_not_widen_into_the_mmio_window() {
        let mem = Memory::new();
        // A widening read here would call out to the host MMIO handler for an
        // address the guest never touched.
        let (_, wide) = mem.fetch_window(MMIO_BASE - 2);
        assert!(!wide, "a fetch beside MMIO must not read across the border");
    }

    #[test]
    fn fetch_window_reads_an_unmapped_page_as_zero() {
        let mem = Memory::new();
        let (bits, _) = mem.fetch_window(0x0080_0000);
        assert_eq!(bits, 0, "an unmapped fetch is the illegal instruction");
    }
}
