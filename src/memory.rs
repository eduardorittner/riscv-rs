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
}

impl Memory {
    pub fn new() -> Self {
        let mut pages = Vec::with_capacity(65536);
        pages.resize_with(65536, || None);
        Self {
            pages,
            brk_ptr: 0x1000000, // 16 MB default heap start
        }
    }

    #[inline(always)]
    fn get_or_create_page(&mut self, page_idx: usize) -> &mut [u8; PAGE_SIZE] {
        if self.pages[page_idx].is_none() {
            self.pages[page_idx] = Some(Box::new([0u8; PAGE_SIZE]));
        }
        self.pages[page_idx].as_mut().unwrap()
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
        self.brk_ptr = val;
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
}
