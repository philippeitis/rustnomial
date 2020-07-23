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

pub trait PowUsize {
    fn upow(self, exp: usize) -> Self;
}

impl PowUsize for u8 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for u16 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for u32 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for u64 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for u128 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for usize {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for i8 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for i16 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for i32 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for i64 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for i128 {
    fn upow(self, exp: usize) -> Self {
        self.pow(exp as u32)
    }
}

impl PowUsize for f32 {
    fn upow(self, exp: usize) -> Self {
        self.powi(exp as i32)
    }
}

impl PowUsize for f64 {
    fn upow(self, exp: usize) -> Self {
        self.powi(exp as i32)
    }
}

pub trait IsPositive {
    fn is_positive(&self) -> bool;
}

impl IsPositive for u8 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for u16 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for u32 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for u64 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for u128 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for usize {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for i8 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for i16 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for i32 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for i64 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for i128 {
    fn is_positive(&self) -> bool {
        self > &0
    }
}

impl IsPositive for f32 {
    fn is_positive(&self) -> bool {
        self > &0.0
    }
}

impl IsPositive for f64 {
    fn is_positive(&self) -> bool {
        self > &0.0
    }
}
