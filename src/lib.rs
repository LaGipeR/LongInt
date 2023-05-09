use std::cmp::{max, Ordering};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Rem, Shl, Shr, Sub};

#[derive(Clone)]
pub struct LongInt {
    blocks: Vec<u32>,
}

impl LongInt {
    pub fn new() -> LongInt {
        LongInt { blocks: vec![0u32] }
    }

    pub fn from_blocks_little_endian(blocks: Vec<u32>) -> LongInt {
        let mut num = LongInt { blocks };

        num.fix();

        num
    }

    pub fn from_blocks_big_endian(mut blocks: Vec<u32>) -> LongInt {
        blocks.reverse();
        Self::from_blocks_little_endian(blocks)
    }

    pub fn from_hex(hex: &str) -> LongInt {
        let mut number = LongInt::new();
        number.setHex(hex);
        number
    }

    fn fix(&mut self) {
        for i in (0..self.blocks.len()).rev() {
            if self.blocks[i] > 0 {
                break;
            }

            self.blocks.pop();
        }

        if self.blocks.len() == 0 {
            self.blocks = vec![0];
        }
    }

    pub fn setHex(&mut self, hex: &str) {
        let mut end_hex_block_pos = hex.len();

        self.blocks = Vec::new();
        while end_hex_block_pos > 0 {
            let start_hex_block_pos = if end_hex_block_pos < 8 {
                0
            } else {
                end_hex_block_pos - 8usize
            };
            self.blocks.push(Self::hex2block(
                &hex[start_hex_block_pos..end_hex_block_pos],
            ));
            end_hex_block_pos = start_hex_block_pos;
        }

        self.fix();
    }

    fn hex2block(hex: &str) -> u32 {
        if hex.len() > 8 {
            panic!("Not big hex block");
        }
        let mut block = 0u32;

        let mut shift = 4 * hex.len();
        for hex_digit in hex.chars() {
            shift -= 4;
            block |= Self::hex2digit(hex_digit) << shift;
        }

        block
    }

    fn hex2digit(hex: char) -> u32 {
        if !hex.is_ascii_hexdigit() {
            panic!("Not hex digit");
        }

        return if hex.is_digit(10) {
            hex as u32 - '0' as u32
        } else {
            hex.to_lowercase().next().unwrap() as u32 - 'a' as u32 + 10u32
        };
    }

    pub fn getHex(&self) -> String {
        let sz = self.blocks.len();
        if sz == 1 && self.blocks[0] == 0 {
            return String::from("0");
        }

        let mut hex = Self::block2hex(self.blocks[sz - 1usize], true);

        for i in (0..(sz - 1)).rev() {
            hex.push_str(&Self::block2hex(self.blocks[i], false));
        }

        hex
    }

    fn block2hex(block: u32, mut leading_zero: bool) -> String {
        let mut hex = String::new();
        let digit_bits: u32 = 0b1111;
        let mut digit_shift = 32 - 4;
        for _ in 0..8 {
            let digit: u32 = (block & (digit_bits << digit_shift)) >> digit_shift;
            digit_shift -= 4;

            if digit > 0 {
                leading_zero = false;
            }
            if leading_zero {
                continue;
            }

            hex.push(std::char::from_digit(digit, 16).unwrap());
        }

        hex
    }

    pub fn pow(base: &LongInt, degree: &LongInt, module: &LongInt) -> LongInt {
        if *degree == LongInt::new() {
            return LongInt::from_hex("1");
        }

        let res = Self::pow(&base, &((degree) >> 1), module);
        let res = (&res * &res) % (module);

        return if degree.blocks[0] & 1 == 1 {
            ((&res) * (base)) % (module)
        } else {
            res
        };
    }
}

impl Not for &LongInt {
    type Output = LongInt;

    fn not(self) -> Self::Output {
        let mut result_blocks: Vec<u32> = Vec::new();

        for block in self.blocks.iter() {
            result_blocks.push(!block);
        }

        LongInt::from_blocks_little_endian(result_blocks)
    }
}
impl Not for LongInt {
    type Output = LongInt;

    fn not(self) -> Self::Output {
        (&self).not()
    }
}

impl BitAnd<&LongInt> for &LongInt {
    type Output = LongInt;

    fn bitand(self, rhs: &LongInt) -> Self::Output {
        let sz = max(self.blocks.len(), rhs.blocks.len());
        let mut result_blocks = Vec::with_capacity(sz);
        for i in 0..sz {
            let lhs_block = if i < self.blocks.len() {
                self.blocks[i]
            } else {
                0u32
            };
            let rhs_block = if i < rhs.blocks.len() {
                rhs.blocks[i]
            } else {
                0u32
            };

            result_blocks.push(lhs_block & rhs_block);
        }

        let mut result = LongInt::from_blocks_little_endian(result_blocks);
        result.fix();
        result
    }
}
impl BitAnd<LongInt> for &LongInt {
    type Output = LongInt;

    fn bitand(self, rhs: LongInt) -> Self::Output {
        (self) & (&rhs)
    }
}
impl BitAnd<&LongInt> for LongInt {
    type Output = LongInt;

    fn bitand(self, rhs: &LongInt) -> Self::Output {
        (&self) & (rhs)
    }
}
impl BitAnd<LongInt> for LongInt {
    type Output = LongInt;

    fn bitand(self, rhs: LongInt) -> Self::Output {
        (&self) & (&rhs)
    }
}

impl BitXor<&LongInt> for &LongInt {
    type Output = LongInt;

    fn bitxor(self, rhs: &LongInt) -> Self::Output {
        let sz = max(self.blocks.len(), rhs.blocks.len());
        let mut result_blocks = Vec::with_capacity(sz);
        for i in 0..sz {
            let lhs_block = if i < self.blocks.len() {
                self.blocks[i]
            } else {
                0u32
            };
            let rhs_block = if i < rhs.blocks.len() {
                rhs.blocks[i]
            } else {
                0u32
            };

            result_blocks.push(lhs_block ^ rhs_block);
        }

        let mut result = LongInt::from_blocks_little_endian(result_blocks);
        result.fix();
        result
    }
}
impl BitXor<LongInt> for &LongInt {
    type Output = LongInt;

    fn bitxor(self, rhs: LongInt) -> Self::Output {
        (self) ^ (&rhs)
    }
}
impl BitXor<&LongInt> for LongInt {
    type Output = LongInt;

    fn bitxor(self, rhs: &LongInt) -> Self::Output {
        (&self) ^ (rhs)
    }
}
impl BitXor<LongInt> for LongInt {
    type Output = LongInt;

    fn bitxor(self, rhs: LongInt) -> Self::Output {
        (&self) ^ (&rhs)
    }
}

impl BitOr<&LongInt> for &LongInt {
    type Output = LongInt;

    fn bitor(self, rhs: &LongInt) -> Self::Output {
        let sz = max(self.blocks.len(), rhs.blocks.len());
        let mut result_blocks = Vec::with_capacity(sz);
        for i in 0..sz {
            let lhs_block = if i < self.blocks.len() {
                self.blocks[i]
            } else {
                0u32
            };
            let rhs_block = if i < rhs.blocks.len() {
                rhs.blocks[i]
            } else {
                0u32
            };

            result_blocks.push(lhs_block | rhs_block);
        }

        let mut result = LongInt::from_blocks_little_endian(result_blocks);
        result.fix();
        result
    }
}
impl BitOr<LongInt> for &LongInt {
    type Output = LongInt;

    fn bitor(self, rhs: LongInt) -> Self::Output {
        (self) | (&rhs)
    }
}
impl BitOr<&LongInt> for LongInt {
    type Output = LongInt;

    fn bitor(self, rhs: &LongInt) -> Self::Output {
        (&self) | (rhs)
    }
}
impl BitOr<LongInt> for LongInt {
    type Output = LongInt;

    fn bitor(self, rhs: LongInt) -> Self::Output {
        (&self) | (&rhs)
    }
}

impl Shl<u32> for &LongInt {
    type Output = LongInt;

    fn shl(self, rhs: u32) -> Self::Output {
        let blocks_shift = (rhs >> 5) as usize;
        let sz = self.blocks.len();

        let mut result_blocks = Vec::new();
        result_blocks.resize(sz + blocks_shift + 1usize, 0);

        for i in 0..sz {
            result_blocks[sz - 1 - i + blocks_shift] = self.blocks[sz - 1 - i];
        }

        let shift = rhs & 0b11111;

        if shift > 0 {
            let mut extra = 0u32;
            for i in blocks_shift..(blocks_shift + sz + 1usize) {
                let tmp_extra = result_blocks[i] >> (32 - shift);
                result_blocks[i] = (result_blocks[i] << shift) | extra;
                extra = tmp_extra;
            }
        }

        let mut result = LongInt::from_blocks_little_endian(result_blocks);
        result.fix();
        result
    }
}
impl Shl<u32> for LongInt {
    type Output = LongInt;

    fn shl(self, rhs: u32) -> Self::Output {
        (&self) << rhs
    }
}

impl Shr<u32> for &LongInt {
    type Output = LongInt;

    fn shr(self, rhs: u32) -> Self::Output {
        let blocks_shift = (rhs >> 5) as usize;
        let sz = self.blocks.len();

        if sz <= blocks_shift {
            return LongInt::new();
        }

        let mut result_blocks = Vec::new();
        result_blocks.resize(sz - blocks_shift, 0);

        for i in 0..(sz - blocks_shift) {
            result_blocks[i] = self.blocks[i + blocks_shift];
        }

        let shift = rhs & 0b11111;

        if shift > 0 {
            let mut extra = 0u32;
            for i in (0..(sz - blocks_shift)).rev() {
                let tmp_extra = result_blocks[i] & ((1 << shift) - 1);
                result_blocks[i] = (result_blocks[i] >> shift) | (extra << (32 - shift));
                extra = tmp_extra;
            }
        }

        let mut result = LongInt::from_blocks_little_endian(result_blocks);
        result.fix();
        result
    }
}
impl Shr<u32> for LongInt {
    type Output = LongInt;

    fn shr(self, rhs: u32) -> Self::Output {
        (&self) >> rhs
    }
}

impl Add<&LongInt> for &LongInt {
    type Output = LongInt;

    fn add(self, rhs: &LongInt) -> Self::Output {
        let sz = max(self.blocks.len(), rhs.blocks.len()) + 1;
        let mut result_blocks = Vec::with_capacity(sz);
        let mut extra = 0u64;
        let result_block_bits: u64 = (1 << 32) - 1;
        for i in 0..sz {
            let lhs_value = if i < self.blocks.len() {
                self.blocks[i] as u64
            } else {
                0u64
            };
            let rhs_value = if i < rhs.blocks.len() {
                rhs.blocks[i] as u64
            } else {
                0u64
            };

            let result_value = lhs_value + rhs_value + extra;
            result_blocks.push((result_value & result_block_bits) as u32);
            extra = result_value >> 32;
        }
        let mut result = LongInt::from_blocks_little_endian(result_blocks);
        result.fix();
        result
    }
}
impl Add<LongInt> for &LongInt {
    type Output = LongInt;

    fn add(self, rhs: LongInt) -> Self::Output {
        (self) + (&rhs)
    }
}
impl Add<&LongInt> for LongInt {
    type Output = LongInt;

    fn add(self, rhs: &LongInt) -> Self::Output {
        (&self) + (rhs)
    }
}
impl Add<LongInt> for LongInt {
    type Output = LongInt;

    fn add(self, rhs: LongInt) -> Self::Output {
        (&self) + (&rhs)
    }
}

impl PartialEq<LongInt> for LongInt {
    fn eq(&self, other: &LongInt) -> bool {
        if self.blocks.len() != other.blocks.len() {
            return false;
        }

        for i in 0..self.blocks.len() {
            if self.blocks[i] != other.blocks[i] {
                return false;
            }
        }

        true
    }

    fn ne(&self, other: &LongInt) -> bool {
        !((&self) == (&other))
    }
}

impl PartialOrd<LongInt> for LongInt {
    fn partial_cmp(&self, other: &LongInt) -> Option<Ordering> {
        return if (&self) < (&other) {
            Some(Ordering::Less)
        } else if (&self) == (&other) {
            Some(Ordering::Equal)
        } else if (&self) > (&other) {
            Some(Ordering::Greater)
        } else {
            None
        };
    }

    fn lt(&self, other: &LongInt) -> bool {
        if self.blocks.len() != other.blocks.len() {
            return self.blocks.len() < other.blocks.len();
        }

        for i in (0..self.blocks.len()).rev() {
            if self.blocks[i] != other.blocks[i] {
                return self.blocks[i] < other.blocks[i];
            }
        }

        false
    }

    fn le(&self, other: &LongInt) -> bool {
        ((&self) < (&other)) || ((&self) == (&other))
    }

    fn gt(&self, other: &LongInt) -> bool {
        (&other) < (&self)
    }

    fn ge(&self, other: &LongInt) -> bool {
        (&other) <= (&self)
    }
}

impl Sub<&LongInt> for &LongInt {
    type Output = LongInt;

    fn sub(self, rhs: &LongInt) -> Self::Output {
        let mut result_blocks = Vec::with_capacity(self.blocks.len());
        let mut extra = 0u64;
        for i in 0..self.blocks.len() {
            let lhs_value = self.blocks[i] as u64;
            let rhs_value = if i < rhs.blocks.len() {
                rhs.blocks[i]
            } else {
                0
            } as u64;
            if lhs_value < rhs_value + extra {
                result_blocks.push(((1 << 32) + lhs_value - rhs_value - extra) as u32);
                extra = 1;
            } else {
                result_blocks.push((lhs_value - rhs_value - extra) as u32);
                extra = 0;
            }
        }

        let mut result = LongInt::from_blocks_little_endian(result_blocks);
        result.fix();
        result
    }
}
impl Sub<LongInt> for &LongInt {
    type Output = LongInt;

    fn sub(self, rhs: LongInt) -> Self::Output {
        (self) - (&rhs)
    }
}
impl Sub<&LongInt> for LongInt {
    type Output = LongInt;

    fn sub(self, rhs: &LongInt) -> Self::Output {
        (&self) - (rhs)
    }
}
impl Sub<LongInt> for LongInt {
    type Output = LongInt;

    fn sub(self, rhs: LongInt) -> Self::Output {
        (&self) - (&rhs)
    }
}

impl Mul<u32> for &LongInt {
    type Output = LongInt;

    fn mul(self, rhs: u32) -> Self::Output {
        let mut result_blocks = Vec::new();
        let mut extra = 0u32;
        let block_bits = u32::MAX as u64;
        for i in 0..self.blocks.len() {
            let result_block = (self.blocks[i] as u64) * (rhs as u64) + (extra as u64);
            result_blocks.push((result_block & block_bits) as u32);
            extra = (result_block >> 32) as u32;
        }

        if extra > 0 {
            result_blocks.push(extra);
        }

        let mut result = LongInt::from_blocks_little_endian(result_blocks);
        result.fix();
        result
    }
}
impl Mul<u32> for LongInt {
    type Output = LongInt;

    fn mul(self, rhs: u32) -> Self::Output {
        (&self) * rhs
    }
}
impl Mul<&LongInt> for u32 {
    type Output = LongInt;

    fn mul(self, rhs: &LongInt) -> Self::Output {
        (rhs) * (self)
    }
}
impl Mul<LongInt> for u32 {
    type Output = LongInt;

    fn mul(self, rhs: LongInt) -> Self::Output {
        (&rhs) * (self)
    }
}

impl Mul<&LongInt> for &LongInt {
    type Output = LongInt;

    fn mul(self, rhs: &LongInt) -> Self::Output {
        if self.blocks.len() == 1 {
            return rhs * self.blocks[0];
        }
        if rhs.blocks.len() == 1 {
            return self * rhs.blocks[0];
        }

        let mid = (max(self.blocks.len(), rhs.blocks.len()) + 1) >> 1;

        let (x0, x1) = if mid > self.blocks.len() {
            (self.clone(), LongInt::new())
        } else {
            let (x0_blocks, x1_blocks) = self.blocks.split_at(mid);
            (
                LongInt::from_blocks_little_endian(x0_blocks.to_vec()),
                LongInt::from_blocks_little_endian(x1_blocks.to_vec()),
            )
        };

        let (y0, y1) = if mid > rhs.blocks.len() {
            (rhs.clone(), LongInt::new())
        } else {
            let (y0_blocks, y1_blocks) = rhs.blocks.split_at(mid);
            (
                LongInt::from_blocks_little_endian(y0_blocks.to_vec()),
                LongInt::from_blocks_little_endian(y1_blocks.to_vec()),
            )
        };

        let z2 = (&x1) * (&y1);
        let z0 = (&x0) * (&y0);
        let z1 = (&(&x1 + &x0)) * (&(&y1 + &y0)) - &z2 - &z0;

        (z2 << ((2 * 32 * mid) as u32)) + (z1 << ((32 * mid) as u32)) + z0
    }
}
impl Mul<LongInt> for &LongInt {
    type Output = LongInt;

    fn mul(self, rhs: LongInt) -> Self::Output {
        (self) * (&rhs)
    }
}
impl Mul<&LongInt> for LongInt {
    type Output = LongInt;

    fn mul(self, rhs: &LongInt) -> Self::Output {
        (&self) * (rhs)
    }
}
impl Mul<LongInt> for LongInt {
    type Output = LongInt;

    fn mul(self, rhs: LongInt) -> Self::Output {
        (&self) * (&rhs)
    }
}

impl Rem<&LongInt> for &LongInt {
    type Output = LongInt;

    fn rem(self, rhs: &LongInt) -> Self::Output {
        let mut result = LongInt::new();

        for block in self.blocks.iter().rev() {
            result = result << 32;
            result.blocks[0] = *block;

            let mut l = 0u32;
            let mut r = u32::MAX;

            let rhs_r = rhs * r;
            if rhs_r <= result {
                result = result - rhs;
                continue;
            }

            let mut rhs_l = LongInt::new();
            while l + 1 < r {
                let m = l + ((r - l) >> 1);

                let rhs_m = rhs * m;
                if rhs_m <= result {
                    l = m;
                    rhs_l = rhs_m;
                } else {
                    r = m;
                }
            }

            result = result - rhs_l;
        }

        result.fix();
        result
    }
}
impl Rem<LongInt> for &LongInt {
    type Output = LongInt;

    fn rem(self, rhs: LongInt) -> Self::Output {
        (self) % (&rhs)
    }
}
impl Rem<&LongInt> for LongInt {
    type Output = LongInt;

    fn rem(self, rhs: &LongInt) -> Self::Output {
        (&self) % (rhs)
    }
}
impl Rem<LongInt> for LongInt {
    type Output = LongInt;

    fn rem(self, rhs: LongInt) -> Self::Output {
        (&self) % (&rhs)
    }
}

impl Div<&LongInt> for &LongInt {
    type Output = LongInt;

    fn div(self, rhs: &LongInt) -> Self::Output {
        let mut rem = LongInt::new();
        let mut result = LongInt::new();
        for block in self.blocks.iter().rev() {
            rem = rem << 32;
            rem.blocks[0] = *block;

            let mut l = 0u32;
            let mut r = u32::MAX;

            let rhs_r = rhs * r;
            if rhs_r <= rem {
                rem = rem - rhs;
                result = result << 32;
                result.blocks[0] = r;
                continue;
            }

            let mut rhs_l = LongInt::new();
            while l + 1 < r {
                let m = l + ((r - l) >> 1);

                let rhs_m = rhs * m;
                if rhs_m <= rem {
                    l = m;
                    rhs_l = rhs_m;
                } else {
                    r = m;
                }
            }

            rem = rem - rhs_l;
            result = result << 32;
            result.blocks[0] = l;
        }

        result.fix();
        result
    }
}
impl Div<LongInt> for &LongInt {
    type Output = LongInt;

    fn div(self, rhs: LongInt) -> Self::Output {
        (self) / (&rhs)
    }
}
impl Div<&LongInt> for LongInt {
    type Output = LongInt;

    fn div(self, rhs: &LongInt) -> Self::Output {
        (&self) / (rhs)
    }
}
impl Div<LongInt> for LongInt {
    type Output = LongInt;

    fn div(self, rhs: LongInt) -> Self::Output {
        (&self) / (&rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::LongInt;

    #[test]
    fn test_new() {
        let a = LongInt::new();

        assert_eq!(a.blocks, vec![0])
    }

    #[test]
    fn test_from_hex() {
        let a =
            LongInt::from_hex("1182d8299c0ec40ca8bf3f49362e95e4ecedaf82bfd167988972412095b13db8");
        assert_eq!(
            a.getHex(),
            "1182d8299c0ec40ca8bf3f49362e95e4ecedaf82bfd167988972412095b13db8"
        );

        let a = LongInt::from_hex("");
        assert_eq!(a.getHex(), "0");

        let a = LongInt::from_hex("0");
        assert_eq!(a.getHex(), "0");

        let a = LongInt::from_hex("1234567890abcdefABCDEF");
        assert_eq!(a.getHex(), "1234567890abcdefabcdef");
    }

    #[test]
    fn test_from_blocks_little() {
        let a = LongInt::from_blocks_little_endian(vec![0b0]);
        assert_eq!(a.blocks, vec![0]);

        let a = LongInt::from_blocks_little_endian(vec![0b1000110]);
        assert_eq!(a.blocks, vec![0b1000110]);

        let a = LongInt::from_blocks_little_endian(vec![0b01011101110100101011010110011010]);
        assert_eq!(a.blocks, vec![0b1011101110100101011010110011010]);

        let a = LongInt::from_blocks_little_endian(vec![0b10101011001011000111101010010110, 0b0]);
        assert_eq!(a.blocks, vec![0b10101011001011000111101010010110]);

        let a = LongInt::from_blocks_little_endian(vec![
            0b10101011001011000111101010010110,
            0b010100101011110101,
        ]);
        assert_eq!(
            a.blocks,
            vec![0b10101011001011000111101010010110, 0b10100101011110101]
        );
    }

    #[test]
    fn test_from_blocks_big() {
        let a = LongInt::from_blocks_big_endian(vec![0b0]);
        assert_eq!(a.blocks, vec![0]);

        let a = LongInt::from_blocks_big_endian(vec![0b1000110]);
        assert_eq!(a.blocks, vec![0b1000110]);

        let a = LongInt::from_blocks_big_endian(vec![0b01011101110100101011010110011010]);
        assert_eq!(a.blocks, vec![0b1011101110100101011010110011010]);

        let a = LongInt::from_blocks_big_endian(vec![0b10101011001011000111101010010110, 0b0]);
        assert_eq!(a.blocks, vec![0, 0b10101011001011000111101010010110]);

        let a = LongInt::from_blocks_big_endian(vec![
            0b10101011001011000111101010010110,
            0b010100101011110101,
        ]);
        assert_eq!(
            a.blocks,
            vec![0b10100101011110101, 0b10101011001011000111101010010110]
        );
    }

    #[test]
    fn test_set() {
        let mut a = LongInt::new();

        a.setHex("a");
        assert_eq!(a.blocks, vec![10]);

        a.setHex("F");
        assert_eq!(a.blocks, vec![15]);

        a.setHex("1");
        assert_eq!(a.blocks, vec![1]);

        a.setHex("0");
        assert_eq!(a.blocks, vec![0]);

        a.setHex("000000000000000000000000000000000000000000000000000000000000000000000");
        assert_eq!(a.blocks, vec![0]);

        a.setHex("");
        assert_eq!(a.blocks, vec![0]);
    }
    #[test]
    fn test_set1() {
        let mut a = LongInt::new();

        a.setHex("ABCDEF");
        assert_eq!(a.blocks, vec![11259375]);

        a.setHex("ABcdEf");
        assert_eq!(a.blocks, vec![11259375]);

        a.setHex("123456789ABcdef");
        assert_eq!(a.blocks, vec![2309737967, 19088743]);

        a.setHex("000123456789ABcdef");
        assert_eq!(a.blocks, vec![2309737967, 19088743]);

        a.setHex("CED92B0423392AF");
        assert_eq!(
            a.blocks,
            vec![
                0b01000010001100111001001010101111,
                0b00001100111011011001001010110000
            ]
        );
    }

    #[test]
    #[should_panic]
    fn test_set_panic() {
        let mut a = LongInt::new();

        a.setHex("G");
    }
    #[test]
    #[should_panic]
    fn test_set_panic1() {
        let mut a = LongInt::new();

        a.setHex("/");
    }
    #[test]
    #[should_panic]
    fn test_set_panic2() {
        let mut a = LongInt::new();

        a.setHex("14628957bbcdeffefef019265152012/8512085601a124142bbc124bde124411fef");
    }

    #[test]
    fn test_get() {
        let mut a = LongInt::new();

        assert_eq!(a.getHex(), String::from("0"));

        a.setHex("0");
        assert_eq!(a.getHex(), String::from("0"));

        a.setHex("F");
        assert_eq!(a.getHex(), String::from("f"));

        a.setHex("a");
        assert_eq!(a.getHex(), String::from("a"));

        a.setHex("1");
        assert_eq!(a.getHex(), String::from("1"));

        a.setHex("5");
        assert_eq!(a.getHex(), String::from("5"));
    }
    #[test]
    fn test_get1() {
        let mut a = LongInt::new();

        a.setHex("CED92B0423392AF");
        assert_eq!(a.getHex(), String::from("ced92b0423392af"));

        a.setHex("51bf608414ad5726a3c1bec098f77b1b54ffb2787f8d528a74c1d7fde6470ea4");
        assert_eq!(
            a.getHex(),
            String::from("51bf608414ad5726a3c1bec098f77b1b54ffb2787f8d528a74c1d7fde6470ea4")
        );

        a.setHex("7d7deab2affa38154326e96d350deee1");
        assert_eq!(a.getHex(), String::from("7d7deab2affa38154326e96d350deee1"));

        a.setHex("a78865c13b14ae4e25e90771b54963ee2d68c0a64d4a8ba7c6f45ee0e9daa65b");
        assert_eq!(
            a.getHex(),
            String::from("a78865c13b14ae4e25e90771b54963ee2d68c0a64d4a8ba7c6f45ee0e9daa65b")
        );
    }

    #[test]
    fn test_xor() {
        let mut a = LongInt::new();
        let mut b = LongInt::new();

        a.setHex("c");
        b.setHex("a");
        let c = (&a) ^ (&b);
        assert_eq!(c.getHex(), "6");

        a.setHex("1234567812345678");
        b.setHex("12345678");
        let c = (&a) ^ (&b);
        assert_eq!(c.getHex(), "1234567800000000");

        a.setHex("12345678");
        b.setHex("123456781234567812345678");
        let c = (&a) ^ (&b);
        assert_eq!(c.getHex(), "123456781234567800000000");

        a.setHex("0");
        b.setHex("1234567812345678");
        let c = (&a) ^ (&b);
        assert_eq!(c.getHex(), "1234567812345678");

        a.setHex("ffffffffffffffff0000000000000000");
        b.setHex("ffffffff00000000ffffffff00000000");
        let c = (&a) ^ (&b);
        assert_eq!(c.getHex(), "ffffffffffffffff00000000");

        a.setHex("51bf608414ad5726a3c1bec098f77b1b54ffb2787f8d528a74c1d7fde6470ea4");
        b.setHex("403db8ad88a3932a0b7e8189aed9eeffb8121dfac05c3512fdb396dd73f6331c");
        let c = (&a) ^ (&b);
        assert_eq!(
            c.getHex(),
            "1182d8299c0ec40ca8bf3f49362e95e4ecedaf82bfd167988972412095b13db8"
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (&a) ^ (&b);
        assert_eq!(
            c.blocks,
            vec![
                0b00110010011111111001011100001110,
                0b10111000110100111001110100111111
            ]
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (a) ^ (&b);
        assert_eq!(
            c.blocks,
            vec![
                0b00110010011111111001011100001110,
                0b10111000110100111001110100111111
            ]
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (&a) ^ (b);
        assert_eq!(
            c.blocks,
            vec![
                0b00110010011111111001011100001110,
                0b10111000110100111001110100111111
            ]
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (a) ^ (b);
        assert_eq!(
            c.blocks,
            vec![
                0b00110010011111111001011100001110,
                0b10111000110100111001110100111111
            ]
        );
    }

    #[test]
    fn test_or() {
        let mut a = LongInt::new();
        let mut b = LongInt::new();

        a.setHex("c");
        b.setHex("a");
        let c = (&a) | (&b);
        assert_eq!(c.getHex(), "e");

        a.setHex("1234567812345678");
        b.setHex("12345678");
        let c = (&a) | (&b);
        assert_eq!(c.getHex(), "1234567812345678");

        a.setHex("12345678");
        b.setHex("123456781234567812345678");
        let c = (&a) | (&b);
        assert_eq!(c.getHex(), "123456781234567812345678");

        a.setHex("0");
        b.setHex("1234567812345678");
        let c = (&a) | (&b);
        assert_eq!(c.getHex(), "1234567812345678");

        a.setHex("ffffffffffffffff0000000000000000");
        b.setHex("ffffffff00000000ffffffff00000000");
        let c = (&a) | (&b);
        assert_eq!(c.getHex(), "ffffffffffffffffffffffff00000000");

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (&a) | (&b);
        assert_eq!(
            c.blocks,
            vec![
                0b10110111011111111011111110011110,
                0b10111011111100111001111101111111
            ]
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (a) | (&b);
        assert_eq!(
            c.blocks,
            vec![
                0b10110111011111111011111110011110,
                0b10111011111100111001111101111111
            ]
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (&a) | (b);
        assert_eq!(
            c.blocks,
            vec![
                0b10110111011111111011111110011110,
                0b10111011111100111001111101111111
            ]
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (a) | (b);
        assert_eq!(
            c.blocks,
            vec![
                0b10110111011111111011111110011110,
                0b10111011111100111001111101111111
            ]
        );
    }

    #[test]
    fn test_and() {
        let mut a = LongInt::new();
        let mut b = LongInt::new();

        a.setHex("c");
        b.setHex("a");
        let c = (&a) & (&b);
        assert_eq!(c.getHex(), "8");

        a.setHex("1234567812345678");
        b.setHex("12345678");
        let c = (&a) & (&b);
        assert_eq!(c.getHex(), "12345678");

        a.setHex("12345678");
        b.setHex("123456781234567812345678");
        let c = (&a) & (&b);
        assert_eq!(c.getHex(), "12345678");

        a.setHex("0");
        b.setHex("1234567812345678");
        let c = (&a) & (&b);
        assert_eq!(c.getHex(), "0");

        a.setHex("ffffffffffffffff0000000000000000");
        b.setHex("ffffffff00000000ffffffff00000000");
        let c = (&a) & (&b);
        assert_eq!(c.getHex(), "ffffffff000000000000000000000000");

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (&a) & (&b);
        assert_eq!(
            c.blocks,
            vec![
                0b10000101000000000010100010010000,
                0b00000011001000000000001001000000
            ]
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (a) & (&b);
        assert_eq!(
            c.blocks,
            vec![
                0b10000101000000000010100010010000,
                0b00000011001000000000001001000000
            ]
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (&a) & (b);
        assert_eq!(
            c.blocks,
            vec![
                0b10000101000000000010100010010000,
                0b00000011001000000000001001000000
            ]
        );

        let a = LongInt::from_blocks_little_endian(vec![
            0b10010101010101010010111010011010,
            0b00010011011000010000111101110101,
        ]);
        let b = LongInt::from_blocks_little_endian(vec![
            0b10100111001010101011100110010100,
            0b10101011101100101001001001001010,
        ]);
        let c = (a) & (b);
        assert_eq!(
            c.blocks,
            vec![
                0b10000101000000000010100010010000,
                0b00000011001000000000001001000000
            ]
        );
    }

    #[test]
    fn test_shl() {
        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) << 4;
        assert_eq!(b.blocks, vec![0b1111111111111000011110000]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) << 1;
        assert_eq!(b.blocks, vec![0b1111111111111000011110]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) << 0;
        assert_eq!(b.blocks, vec![0b111111111111100001111]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) << 32;
        assert_eq!(b.blocks, vec![0, 0b111111111111100001111]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) << 64;
        assert_eq!(b.blocks, vec![0, 0, 0b111111111111100001111]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) << 40;
        assert_eq!(b.blocks, vec![0, 0b11111111111110000111100000000]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) << 52;
        assert_eq!(
            b.blocks,
            vec![0, 0b11110000111100000000000000000000, 0b111111111]
        );

        let a = LongInt::from_blocks_big_endian(vec![
            0b101001011010101101010101010101,
            0b1111001000101010000011101,
            0b11111111111111111111111111111111,
        ]);
        let b = (&a) << 32;
        assert_eq!(
            b.blocks,
            vec![
                0,
                0b11111111111111111111111111111111,
                0b1111001000101010000011101,
                0b101001011010101101010101010101
            ]
        );

        let a = LongInt::from_blocks_big_endian(vec![
            0b00101001011010101101010101010101,
            0b00000001111001000101010000011101,
            0b11111111111111111111111111111111,
        ]);
        let b = (&a) << 20;
        assert_eq!(
            b.blocks,
            vec![
                0b11111111111100000000000000000000,
                0b01000001110111111111111111111111,
                0b01010101010100000001111001000101,
                0b00101001011010101101
            ]
        );

        let a = LongInt::from_blocks_big_endian(vec![
            0b00101001011010101101010101010101,
            0b00000001111001000101010000011101,
            0b11111111111111111111111111111111,
        ]);
        let b = (&a) << 55;
        let mut result_blocks = vec![
            0b00101001011010101101010,
            0b10101010100000001111001000101010,
            0b00001110111111111111111111111111,
            0b11111111100000000000000000000000,
            0b00000000000000000000000000000000,
        ];
        result_blocks.reverse();
        assert_eq!(b.blocks, result_blocks);

        let a = LongInt::from_blocks_big_endian(vec![
            0b00101001011010101101010101010101,
            0b00000001111001000101010000011101,
            0b11111111111111111111111111111111,
        ]);
        let b = (&a) << 17;
        assert_eq!(
            b.blocks,
            vec![
                0b11111111111111100000000000000000,
                0b10101000001110111111111111111111,
                0b10101010101010100000001111001000,
                0b00101001011010101
            ]
        );
    }

    #[test]
    fn test_shr() {
        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) >> 4;
        assert_eq!(b.blocks, vec![0b11111111111110000]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) >> 1;
        assert_eq!(b.blocks, vec![0b11111111111110000111]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) >> 0;
        assert_eq!(b.blocks, vec![0b111111111111100001111]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) >> 32;
        assert_eq!(b.blocks, vec![0]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) >> 64;
        assert_eq!(b.blocks, vec![0]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) >> 5;
        assert_eq!(b.blocks, vec![0b1111111111111000]);

        let a = LongInt::from_blocks_big_endian(vec![0b111111111111100001111]);
        let b = (&a) >> 8;
        assert_eq!(b.blocks, vec![0b1111111111111]);

        let a = LongInt::from_blocks_big_endian(vec![
            0b101001011010101101010101010101,
            0b1111001000101010000011101,
            0b11111111111111111111111111111111,
        ]);
        let b = (&a) >> 32;
        assert_eq!(
            b.blocks,
            vec![
                0b1111001000101010000011101,
                0b101001011010101101010101010101
            ]
        );

        let a = LongInt::from_blocks_big_endian(vec![
            0b00101001011010101101010101010101,
            0b00000001111001000101010000011101,
            0b11111111111111111111111111111111,
        ]);
        let b = (&a) >> 20;
        assert_eq!(
            b.blocks,
            vec![
                0b01000101010000011101111111111111,
                0b10101101010101010101000000011110,
                0b001010010110
            ]
        );

        let a = LongInt::from_blocks_big_endian(vec![
            0b00101001011010101101010101010101,
            0b00000001111001000101010000011101,
            0b11111111111111111111111111111111,
        ]);
        let b = (&a) >> 55;
        assert_eq!(
            b.blocks,
            vec![0b11010101101010101010101000000011, 0b001010010]
        );

        let a = LongInt::from_blocks_big_endian(vec![
            0b00101001011010101101010101010101,
            0b00000001111001000101010000011101,
            0b11111111111111111111111111111111,
        ]);
        let b = (&a) >> 17;
        assert_eq!(
            b.blocks,
            vec![
                0b00101010000011101111111111111111,
                0b01101010101010101000000011110010,
                0b001010010110101
            ]
        );
    }

    #[test]
    fn test_add() {
        let a = LongInt::from_hex("A");
        let b = LongInt::from_hex("12");
        let c = a + b;
        assert_eq!(c.getHex(), "1c");

        let a = LongInt::from_blocks_big_endian(vec![0b11111111111111111111111111111111]);
        let b = LongInt::from_blocks_big_endian(vec![0b11111111111111111111111111111111]);
        let c = a + b;
        assert_eq!(c.blocks, vec![0b11111111111111111111111111111110, 0b1]);

        let a = LongInt::from_blocks_big_endian(vec![
            0b11111111111111111111111111111111,
            0b11111111111111111111111111111111,
        ]);
        let b = LongInt::from_blocks_big_endian(vec![0b1]);
        let c = a + b;
        assert_eq!(c.blocks, vec![0, 0, 0b1]);

        let a =
            LongInt::from_hex("a78865c13b14ae4e25e90771b54963ee2d68c0a64d4a8ba7c6f45ee0e9daa65b");
        let b = LongInt::from_hex("0");
        let c = (&a) + (&b);
        assert_eq!(
            c.getHex(),
            "a78865c13b14ae4e25e90771b54963ee2d68c0a64d4a8ba7c6f45ee0e9daa65b"
        );

        let a =
            LongInt::from_hex("36f028580bb02cc8272a9a020f4200e346e276ae664e45ee80745574e2f5ab80");
        let b =
            LongInt::from_hex("70983d692f648185febe6d6fa607630ae68649f7e6fc45b94680096c06e4fadb");
        let c = (&a) + (&b);
        assert_eq!(
            c.getHex(),
            "a78865c13b14ae4e25e90771b54963ee2d68c0a64d4a8ba7c6f45ee0e9daa65b"
        );
        assert_eq!(
            a.getHex(),
            "36f028580bb02cc8272a9a020f4200e346e276ae664e45ee80745574e2f5ab80"
        );
        assert_eq!(
            b.getHex(),
            "70983d692f648185febe6d6fa607630ae68649f7e6fc45b94680096c06e4fadb"
        );

        let d = (&b) + (&a);
        assert_eq!(c.getHex(), d.getHex());

        let a =
            LongInt::from_hex("36f028580bb02cc8272a9a020f4200e346e276ae664e45ee80745574e2f5ab80");
        let b =
            LongInt::from_hex("70983d692f648185febe6d6fa607630ae68649f7e6fc45b94680096c06e4fadb");
        let c = (&a) + (b);
        assert_eq!(
            c.getHex(),
            "a78865c13b14ae4e25e90771b54963ee2d68c0a64d4a8ba7c6f45ee0e9daa65b"
        );
        assert_eq!(
            a.getHex(),
            "36f028580bb02cc8272a9a020f4200e346e276ae664e45ee80745574e2f5ab80"
        );

        let a =
            LongInt::from_hex("36f028580bb02cc8272a9a020f4200e346e276ae664e45ee80745574e2f5ab80");
        let b =
            LongInt::from_hex("70983d692f648185febe6d6fa607630ae68649f7e6fc45b94680096c06e4fadb");
        let c = (a) + (&b);
        assert_eq!(
            c.getHex(),
            "a78865c13b14ae4e25e90771b54963ee2d68c0a64d4a8ba7c6f45ee0e9daa65b"
        );
        assert_eq!(
            b.getHex(),
            "70983d692f648185febe6d6fa607630ae68649f7e6fc45b94680096c06e4fadb"
        );

        let a =
            LongInt::from_hex("36f028580bb02cc8272a9a020f4200e346e276ae664e45ee80745574e2f5ab80");
        let b =
            LongInt::from_hex("70983d692f648185febe6d6fa607630ae68649f7e6fc45b94680096c06e4fadb");
        let c = (a) + (b);
        assert_eq!(
            c.getHex(),
            "a78865c13b14ae4e25e90771b54963ee2d68c0a64d4a8ba7c6f45ee0e9daa65b"
        );
    }

    #[test]
    fn test_sub() {
        let a = LongInt::new();
        let b = LongInt::from_hex("0");
        let c = a - b;
        assert_eq!(c.getHex(), "0");

        let a = LongInt::from_hex("125def19a51ab15");
        let b = LongInt::from_hex("0");
        let c = a - b;
        assert_eq!(c.getHex(), "125def19a51ab15");

        let a = LongInt::from_hex("123FFFFFFFE");
        let b = LongInt::from_hex("FFFFFFFF");
        let c = a - b;
        assert_eq!(c.getHex(), "122ffffffff");

        let a = LongInt::from_blocks_big_endian(vec![1, 0]);
        let b = LongInt::from_hex("FFFFFFFF");
        let c = a - b;
        assert_eq!(c.getHex(), "1");

        let a =
            LongInt::from_hex("33ced2c76b26cae94e162c4c0d2c0ff7c13094b0185a3c122e732d5ba77efebc");
        let b =
            LongInt::from_hex("22e962951cb6cd2ce279ab0e2095825c141d48ef3ca9dabf253e38760b57fe03");
        let c = (&a) - (&b);
        assert_eq!(
            c.getHex(),
            "10e570324e6ffdbc6b9c813dec968d9bad134bc0dbb061530934f4e59c2700b9"
        );
        assert_eq!(
            a.getHex(),
            "33ced2c76b26cae94e162c4c0d2c0ff7c13094b0185a3c122e732d5ba77efebc"
        );
        assert_eq!(
            b.getHex(),
            "22e962951cb6cd2ce279ab0e2095825c141d48ef3ca9dabf253e38760b57fe03"
        );

        let a =
            LongInt::from_hex("33ced2c76b26cae94e162c4c0d2c0ff7c13094b0185a3c122e732d5ba77efebc");
        let b =
            LongInt::from_hex("22e962951cb6cd2ce279ab0e2095825c141d48ef3ca9dabf253e38760b57fe03");
        let c = (&a) - (b);
        assert_eq!(
            c.getHex(),
            "10e570324e6ffdbc6b9c813dec968d9bad134bc0dbb061530934f4e59c2700b9"
        );
        assert_eq!(
            a.getHex(),
            "33ced2c76b26cae94e162c4c0d2c0ff7c13094b0185a3c122e732d5ba77efebc"
        );

        let a =
            LongInt::from_hex("33ced2c76b26cae94e162c4c0d2c0ff7c13094b0185a3c122e732d5ba77efebc");
        let b =
            LongInt::from_hex("22e962951cb6cd2ce279ab0e2095825c141d48ef3ca9dabf253e38760b57fe03");
        let c = (a) - (&b);
        assert_eq!(
            c.getHex(),
            "10e570324e6ffdbc6b9c813dec968d9bad134bc0dbb061530934f4e59c2700b9"
        );
        assert_eq!(
            b.getHex(),
            "22e962951cb6cd2ce279ab0e2095825c141d48ef3ca9dabf253e38760b57fe03"
        );

        let a =
            LongInt::from_hex("33ced2c76b26cae94e162c4c0d2c0ff7c13094b0185a3c122e732d5ba77efebc");
        let b =
            LongInt::from_hex("22e962951cb6cd2ce279ab0e2095825c141d48ef3ca9dabf253e38760b57fe03");
        let c = (a) - (b);
        assert_eq!(
            c.getHex(),
            "10e570324e6ffdbc6b9c813dec968d9bad134bc0dbb061530934f4e59c2700b9"
        );
    }

    #[test]
    fn test_less() {
        let a = LongInt::new();
        let b = LongInt::from_hex("0");
        assert_eq!(a < b, false);

        let a = LongInt::new();
        let b = LongInt::from_hex("12341215781");
        assert_eq!(a < b, true);

        let a = LongInt::from_hex("1FFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a < b, false);

        let a = LongInt::from_hex("EFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a < b, true);

        let a = LongInt::from_blocks_big_endian(vec![
            0b11010100101010101010111101010111,
            0b11011111010111110101001010101011,
        ]);
        let b = LongInt::from_blocks_big_endian(vec![
            0b11010101010010101000010101011010,
            0b11010000010101011011101101011001,
        ]);
        assert_eq!(a < b, true);
    }

    #[test]
    fn test_less_equal() {
        let a = LongInt::new();
        let b = LongInt::from_hex("0");
        assert_eq!(a <= b, true);

        let a = LongInt::new();
        let b = LongInt::from_hex("12341215781");
        assert_eq!(a <= b, true);

        let a = LongInt::from_hex("1FFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a <= b, false);

        let a = LongInt::from_hex("EFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a <= b, true);

        let a = LongInt::from_blocks_big_endian(vec![
            0b11010100101010101010111101010111,
            0b11011111010111110101001010101011,
        ]);
        let b = LongInt::from_blocks_big_endian(vec![
            0b11010101010010101000010101011010,
            0b11010000010101011011101101011001,
        ]);
        assert_eq!(a <= b, true);
    }

    #[test]
    fn test_greater() {
        let a = LongInt::new();
        let b = LongInt::from_hex("0");
        assert_eq!(a > b, false);

        let a = LongInt::new();
        let b = LongInt::from_hex("12341215781");
        assert_eq!(a > b, false);

        let a = LongInt::from_hex("1FFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a > b, true);

        let a = LongInt::from_hex("EFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a > b, false);

        let a = LongInt::from_blocks_big_endian(vec![
            0b11010100101010101010111101010111,
            0b11011111010111110101001010101011,
        ]);
        let b = LongInt::from_blocks_big_endian(vec![
            0b11010101010010101000010101011010,
            0b11010000010101011011101101011001,
        ]);
        assert_eq!(a > b, false);
    }

    #[test]
    fn test_greater_equal() {
        let a = LongInt::new();
        let b = LongInt::from_hex("0");
        assert_eq!(a >= b, true);

        let a = LongInt::new();
        let b = LongInt::from_hex("12341215781");
        assert_eq!(a >= b, false);

        let a = LongInt::from_hex("1FFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a >= b, true);

        let a = LongInt::from_hex("EFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a >= b, false);

        let a = LongInt::from_blocks_big_endian(vec![
            0b11010100101010101010111101010111,
            0b11011111010111110101001010101011,
        ]);
        let b = LongInt::from_blocks_big_endian(vec![
            0b11010101010010101000010101011010,
            0b11010000010101011011101101011001,
        ]);
        assert_eq!(a >= b, false);
    }

    #[test]
    fn test_equal() {
        let a = LongInt::new();
        let b = LongInt::from_hex("0");
        assert_eq!(a == b, true);

        let a = LongInt::new();
        let b = LongInt::from_hex("12341215781");
        assert_eq!(a == b, false);

        let a = LongInt::from_hex("1FFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a == b, false);

        let a = LongInt::from_hex("FFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a == b, true);

        let a = LongInt::from_blocks_big_endian(vec![
            0b11010100101010101010111101010111,
            0b11011111010111110101001010101011,
        ]);
        let b = LongInt::from_blocks_big_endian(vec![
            0b11010101010010101000010101011010,
            0b11010000010101011011101101011001,
        ]);
        assert_eq!(a == b, false);
    }

    #[test]
    fn test_not_equal() {
        let a = LongInt::new();
        let b = LongInt::from_hex("0");
        assert_eq!(a != b, false);

        let a = LongInt::new();
        let b = LongInt::from_hex("12341215781");
        assert_eq!(a != b, true);

        let a = LongInt::from_hex("1FFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a != b, true);

        let a = LongInt::from_hex("FFFFFFFF");
        let b = LongInt::from_hex("FFFFFFFF");
        assert_eq!(a != b, false);

        let a = LongInt::from_blocks_big_endian(vec![
            0b11010100101010101010111101010111,
            0b11011111010111110101001010101011,
        ]);
        let b = LongInt::from_blocks_big_endian(vec![
            0b11010101010010101000010101011010,
            0b11010000010101011011101101011001,
        ]);
        assert_eq!(a != b, true);
    }

    #[test]
    fn test_mod() {
        let a = LongInt::from_blocks_big_endian(vec![66]);
        let b = LongInt::from_blocks_big_endian(vec![64]);
        let c = (&a) % (&b);
        assert_eq!(c.getHex(), "2");

        let a = LongInt::from_blocks_big_endian(vec![128]);
        let b = LongInt::from_blocks_big_endian(vec![64]);
        let c = (&a) % (&b);
        assert_eq!(c.getHex(), "0");

        let a = LongInt::from_blocks_big_endian(vec![32]);
        let b = LongInt::from_blocks_big_endian(vec![64]);
        let c = (&a) % (&b);
        assert_eq!(c.blocks, vec![32]);

        let a = LongInt::from_blocks_big_endian(vec![715815115u32]);
        let b = LongInt::from_blocks_big_endian(vec![168951u32]);
        let c = (&a) % (&b);
        assert_eq!(c.blocks, vec![138679]);

        let a = LongInt::from_hex("ba89421bfa1265981e");
        let b = LongInt::from_hex("142fd1246128c");
        let c = (&a) % (&b);
        assert_eq!(c.getHex(), "fc9acfccad2a");

        let a = LongInt::from_hex("21cfd124cddf1924df124");
        let b = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let c = (&a) % (&b);
        assert_eq!(c.getHex(), "21cfd124cddf1924df124");

        let a = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let b = LongInt::from_hex("21cfd124cddf1924df124");
        let c = (&a) % (&b);
        assert_eq!(c.getHex(), "1abfaf375d141b5f6d82d");
        assert_eq!(a.getHex(), "124bbc2d16f419581acfe1573a514bdfe17dff15");
        assert_eq!(b.getHex(), "21cfd124cddf1924df124");

        let a = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let b = LongInt::from_hex("21cfd124cddf1924df124");
        let c = (&a) % (b);
        assert_eq!(c.getHex(), "1abfaf375d141b5f6d82d");
        assert_eq!(a.getHex(), "124bbc2d16f419581acfe1573a514bdfe17dff15");

        let a = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let b = LongInt::from_hex("21cfd124cddf1924df124");
        let c = (a) % (&b);
        assert_eq!(c.getHex(), "1abfaf375d141b5f6d82d");
        assert_eq!(b.getHex(), "21cfd124cddf1924df124");

        let a = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let b = LongInt::from_hex("21cfd124cddf1924df124");
        let c = (a) % (b);
        assert_eq!(c.getHex(), "1abfaf375d141b5f6d82d");
    }

    #[test]
    fn test_mul_by_u32() {
        let a = LongInt::from_hex("2");
        let b = 4u32;
        let c = (&a) * b;
        assert_eq!(c.getHex(), "8");

        let a = LongInt::from_hex("1");
        let b = 215u32;
        let c = (&a) * b;
        assert_eq!(c.getHex(), "d7");

        let a = LongInt::from_hex("1586bd512409cdf12561afd");
        let b = 1u32;
        let c = (&a) * b;
        assert_eq!(c.getHex(), "1586bd512409cdf12561afd");

        let a = LongInt::from_hex("125617582");
        let b = 0u32;
        let c = (&a) * b;
        assert_eq!(c.getHex(), "0");

        let a = LongInt::from_hex("1000000000000000000000000");
        let b = 1 << 28;
        let c = (&a) * b;
        assert_eq!(c.getHex(), "10000000000000000000000000000000");

        let a = LongInt::from_hex("217541bdcfeda38215158316");
        let b = 215481257u32;
        let c = (&a) * b;
        assert_eq!(c.getHex(), "1adb9732a06456aae6e32c222341b86");
        assert_eq!(a.getHex(), "217541bdcfeda38215158316");
        assert_eq!(b, 215481257);

        let a = LongInt::from_hex("217541bdcfeda38215158316");
        let b = 215481257u32;
        let c = a * b;
        assert_eq!(c.getHex(), "1adb9732a06456aae6e32c222341b86");
        assert_eq!(b, 215481257);

        let a = LongInt::from_blocks_big_endian(vec![
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        ]);
        let b = u32::MAX;
        let c = (&a) * b;
        assert_eq!(
            c.getHex(),
            "fffffffeffffffffffffffffffffffffffffffffffffffff00000001"
        );
    }

    #[test]
    fn test_mul() {
        let a = LongInt::from_hex("2");
        let b = LongInt::from_hex("4");
        let c = (&a) * (&b);
        assert_eq!(c.getHex(), "8");

        let a = LongInt::from_hex("1");
        let b = LongInt::from_hex("215");
        let c = (&a) * (&b);
        assert_eq!(c.getHex(), "215");

        let a = LongInt::from_hex("125617582");
        let b = LongInt::from_hex("0");
        let c = (&a) * (&b);
        assert_eq!(c.getHex(), "0");

        let a = LongInt::from_hex("1000000000000000000000000");
        let b = LongInt::from_hex("100000000");
        let c = (&a) * (&b);
        assert_eq!(c.getHex(), "100000000000000000000000000000000");

        let a = LongInt::from_hex("217541bdcfeda38215158316");
        let b = LongInt::from_hex("12856abbe1534dfede581734");
        let c = (&a) * (&b);
        assert_eq!(
            c.getHex(),
            "26bae7d9b8e4bb218d18f1e9d5fec61d2dd7b375ab59a78"
        );

        let a = LongInt::from_hex("7d7deab2affa38154326e96d350deee1");
        let b = LongInt::from_hex("97f92a75b3faf8939e8e98b96476fd22");
        let c = (&a) * (&b);
        assert_eq!(
            c.getHex(),
            "4a7f69b908e167eb0dc9af7bbaa5456039c38359e4de4f169ca10c44d0a416e2"
        );
        assert_eq!(a.getHex(), "7d7deab2affa38154326e96d350deee1");
        assert_eq!(b.getHex(), "97f92a75b3faf8939e8e98b96476fd22");

        let a = LongInt::from_hex("7d7deab2affa38154326e96d350deee1");
        let b = LongInt::from_hex("97f92a75b3faf8939e8e98b96476fd22");
        let c = (&a) * (b);
        assert_eq!(
            c.getHex(),
            "4a7f69b908e167eb0dc9af7bbaa5456039c38359e4de4f169ca10c44d0a416e2"
        );
        assert_eq!(a.getHex(), "7d7deab2affa38154326e96d350deee1");

        let a = LongInt::from_hex("7d7deab2affa38154326e96d350deee1");
        let b = LongInt::from_hex("97f92a75b3faf8939e8e98b96476fd22");
        let c = (a) * (&b);
        assert_eq!(
            c.getHex(),
            "4a7f69b908e167eb0dc9af7bbaa5456039c38359e4de4f169ca10c44d0a416e2"
        );
        assert_eq!(b.getHex(), "97f92a75b3faf8939e8e98b96476fd22");

        let a = LongInt::from_hex("7d7deab2affa38154326e96d350deee1");
        let b = LongInt::from_hex("97f92a75b3faf8939e8e98b96476fd22");
        let c = (a) * (b);
        assert_eq!(
            c.getHex(),
            "4a7f69b908e167eb0dc9af7bbaa5456039c38359e4de4f169ca10c44d0a416e2"
        );

        let a = LongInt::from_blocks_big_endian(vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
        let b = LongInt::from_blocks_big_endian(vec![
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        ]);
        let c = (&a) * (&b);
        assert_eq!(
            c.getHex(),
            "fffffffffffffffffffffffffffffffeffffffffffffffff00000000000000000000000000000001"
        );
    }

    #[test]
    fn test_div() {
        let a = LongInt::from_blocks_big_endian(vec![66]);
        let b = LongInt::from_blocks_big_endian(vec![64]);
        let c = (&a) / (&b);
        assert_eq!(c.getHex(), "1");

        let a = LongInt::from_blocks_big_endian(vec![128]);
        let b = LongInt::from_blocks_big_endian(vec![64]);
        let c = (&a) / (&b);
        assert_eq!(c.getHex(), "2");

        let a = LongInt::from_blocks_big_endian(vec![32]);
        let b = LongInt::from_blocks_big_endian(vec![64]);
        let c = (&a) / (&b);
        assert_eq!(c.blocks, vec![0]);

        let a = LongInt::from_blocks_big_endian(vec![715815115u32]);
        let b = LongInt::from_blocks_big_endian(vec![168951u32]);
        let c = (&a) / (&b);
        assert_eq!(c.blocks, vec![4236u32]);

        let a = LongInt::from_hex("ba89421bfa1265981e");
        let b = LongInt::from_hex("142fd1246128c");
        let c = (&a) / (&b);
        assert_eq!(c.blocks, vec![9689375]);

        let a = LongInt::from_hex("21cfd124cddf1924df124");
        let b = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let c = (&a) / (&b);
        assert_eq!(c.getHex(), "0");

        let a = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let b = LongInt::from_hex("21cfd124cddf1924df124");
        let c = (&a) / (&b);
        assert_eq!(c.getHex(), "8a86141ace6f86ab7ea");
        assert_eq!(a.getHex(), "124bbc2d16f419581acfe1573a514bdfe17dff15");
        assert_eq!(b.getHex(), "21cfd124cddf1924df124");

        let a = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let b = LongInt::from_hex("21cfd124cddf1924df124");
        let c = (&a) / (b);
        assert_eq!(c.getHex(), "8a86141ace6f86ab7ea");
        assert_eq!(a.getHex(), "124bbc2d16f419581acfe1573a514bdfe17dff15");

        let a = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let b = LongInt::from_hex("21cfd124cddf1924df124");
        let c = (a) / (&b);
        assert_eq!(c.getHex(), "8a86141ace6f86ab7ea");
        assert_eq!(b.getHex(), "21cfd124cddf1924df124");

        let a = LongInt::from_hex("124bbc2d16f419581acfe1573a514bdfe17dff15");
        let b = LongInt::from_hex("21cfd124cddf1924df124");
        let c = (a) / (b);
        assert_eq!(c.getHex(), "8a86141ace6f86ab7ea");
    }

    #[test]
    fn test_pow() {
        let a = LongInt::from_hex("2");
        let b = LongInt::from_hex("20");
        let module = LongInt::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFF");
        let c = LongInt::pow(&a, &b, &module);
        assert_eq!(c.getHex(), "100000000");

        let a = LongInt::from_hex("14bdf371dcb14");
        let b = LongInt::from_hex("0");
        let module = LongInt::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
        let c = LongInt::pow(&a, &b, &module);
        assert_eq!(c.getHex(), "1");

        let a = LongInt::from_hex("0");
        let b = LongInt::from_hex("125bc2965dfeaa615cbde1");
        let module = LongInt::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
        let c = LongInt::pow(&a, &b, &module);
        assert_eq!(c.getHex(), "0");

        let a = LongInt::from_hex("3");
        let b = LongInt::from_hex("12");
        let module = LongInt::from_hex("4");
        let c = LongInt::pow(&a, &b, &module);
        assert_eq!(c.getHex(), "1");

        let a = LongInt::from_hex("3156715");
        let b = LongInt::from_hex("12125");
        let module = LongInt::from_hex("2516812");
        let c = LongInt::pow(&a, &b, &module);
        assert_eq!(c.getHex(), "1997863");

        let a = LongInt::from_hex("12ddd312cb124869deaf2512ade");
        let b = LongInt::from_hex("dcbf135869196bcd6510924");
        let module = LongInt::from_hex("bcd138956163cdea9411590671de");
        let c = LongInt::pow(&a, &b, &module);
        assert_eq!(c.getHex(), "553feeecaceb3c9f86e1349e945e");
    }
}
