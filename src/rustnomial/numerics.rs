use std::convert::TryFrom;

pub trait HasOne {
    fn one() -> Self;
}

impl HasOne for u8 {
    fn one() -> Self {
        1
    }
}

impl HasOne for u16 {
    fn one() -> Self {
        1
    }
}

impl HasOne for u32 {
    fn one() -> Self {
        1
    }
}

impl HasOne for u64 {
    fn one() -> Self {
        1
    }
}

impl HasOne for u128 {
    fn one() -> Self {
        1
    }
}

impl HasOne for usize {
    fn one() -> Self {
        1
    }
}

impl HasOne for i8 {
    fn one() -> Self {
        1
    }
}

impl HasOne for i16 {
    fn one() -> Self {
        1
    }
}

impl HasOne for i32 {
    fn one() -> Self {
        1
    }
}

impl HasOne for i64 {
    fn one() -> Self {
        1
    }
}

impl HasOne for i128 {
    fn one() -> Self {
        1
    }
}

impl HasOne for isize {
    fn one() -> Self {
        1
    }
}

impl HasOne for f32 {
    fn one() -> Self {
        1.0
    }
}

impl HasOne for f64 {
    fn one() -> Self {
        1.0
    }
}

pub trait HasZero {
    fn zero() -> Self;
}

impl HasZero for u8 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for u16 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for u32 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for u64 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for u128 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for usize {
    fn zero() -> Self {
        0
    }
}

impl HasZero for i8 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for i16 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for i32 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for i64 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for i128 {
    fn zero() -> Self {
        0
    }
}

impl HasZero for isize {
    fn zero() -> Self {
        0
    }
}

impl HasZero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl HasZero for f64 {
    fn zero() -> Self {
        0.0
    }
}

pub trait IsNegativeOne {
    fn is_negative_one(self) -> bool;
}


impl IsNegativeOne for u8 {
    fn is_negative_one(self) -> bool {
        false
    }
}

impl IsNegativeOne for u16 {
    fn is_negative_one(self) -> bool {
        false
    }
}

impl IsNegativeOne for u32 {
    fn is_negative_one(self) -> bool {
        false
    }
}

impl IsNegativeOne for u64 {
    fn is_negative_one(self) -> bool {
        false
    }
}

impl IsNegativeOne for u128 {
    fn is_negative_one(self) -> bool {
        false
    }
}

impl IsNegativeOne for usize {
    fn is_negative_one(self) -> bool {
        false
    }
}

impl IsNegativeOne for i8 {
    fn is_negative_one(self) -> bool {
        self == -1
    }
}

impl IsNegativeOne for i16 {
    fn is_negative_one(self) -> bool {
        self == -1
    }
}

impl IsNegativeOne for i32 {
    fn is_negative_one(self) -> bool {
        self == -1
    }
}

impl IsNegativeOne for i64 {
    fn is_negative_one(self) -> bool {
        self == -1
    }
}

impl IsNegativeOne for i128 {
    fn is_negative_one(self) -> bool {
        self == -1
    }
}

impl IsNegativeOne for isize {
    fn is_negative_one(self) -> bool {
        self == -1
    }
}

impl IsNegativeOne for f32 {
    fn is_negative_one(self) -> bool {
        self == -1.0
    }
}

impl IsNegativeOne for f64 {
    fn is_negative_one(self) -> bool {
        self == -1.0
    }
}

pub trait Abs {
    fn abs(self) -> Self;
}


impl Abs for u8 {
    fn abs(self) -> Self {
        self
    }
}

impl Abs for u16 {
    fn abs(self) -> Self {
        self
    }
}

impl Abs for u32 {
    fn abs(self) -> Self {
        self
    }
}

impl Abs for u64 {
    fn abs(self) -> Self {
        self
    }
}

impl Abs for u128 {
    fn abs(self) -> Self {
        self
    }
}

impl Abs for usize {
    fn abs(self) -> Self {
        self
    }
}

impl Abs for i8 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for i16 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for i32 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for i64 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for i128 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for f32 {
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for f64 {
    fn abs(self) -> Self {
        self.abs()
    }
}
