/// Shift right by `shift` bits then bitwise AND with `mask`.
/// Equivalent to `(val >> shift) & mask`.
#[inline(always)]
pub const fn shift_then_mask(val: u32, shift: u32, mask: u32) -> u32 {
    (val >> shift) & mask
}

/// Helper trait enabling `val.shift_then_mask(shift, mask)` method syntax on integer types.
pub trait ShiftThenMask {
    fn shift_then_mask(self, shift: u32, mask: Self) -> Self;
}

impl ShiftThenMask for u32 {
    #[inline(always)]
    fn shift_then_mask(self, shift: u32, mask: u32) -> u32 {
        (self >> shift) & mask
    }
}

impl ShiftThenMask for u16 {
    #[inline(always)]
    fn shift_then_mask(self, shift: u32, mask: u16) -> u16 {
        (self >> shift) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shift_then_mask() {
        let val: u32 = 0xABCD_EF12;
        assert_eq!(shift_then_mask(val, 4, 0xF), 0x1);
        assert_eq!(val.shift_then_mask(8, 0xFF), 0xEF);

        let val16: u16 = 0xE6C0;
        assert_eq!(val16.shift_then_mask(13, 0x7), 0x7);
    }
}
