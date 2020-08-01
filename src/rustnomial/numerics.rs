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

pub trait AbsSqrt {
    fn abs_sqrt(self) -> Self;
}

impl AbsSqrt for u8 {
    fn abs_sqrt(self) -> Self {
        let x = self;
        if x < 2 {
            return x;
        }

        let shift = 8 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for u16 {
    fn abs_sqrt(self) -> Self {
        let x = self;
        if x < 2 {
            return x;
        }

        let shift = 16 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for u32 {
    fn abs_sqrt(self) -> Self {
        let x = self;
        if x < 2 {
            return x;
        }

        let shift = 32 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for u64 {
    fn abs_sqrt(self) -> Self {
        let x = self;
        if x < 2 {
            return x;
        }

        let shift = 64 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for u128 {
    fn abs_sqrt(self) -> Self {
        let x = self;
        if x < 2 {
            return x;
        }

        let shift = 128 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for usize {
    fn abs_sqrt(self) -> Self {
        let x = self;
        if x < 2 {
            return x;
        }

        let shift = 0usize.count_zeros() - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for i8 {
    fn abs_sqrt(self) -> Self {
        let x = self.abs();
        if x < 2 {
            return x;
        }

        let shift = 8 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for i16 {
    fn abs_sqrt(self) -> Self {
        let x = self.abs();
        if x < 2 {
            return x;
        }

        let shift = 16 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for i32 {
    fn abs_sqrt(self) -> Self {
        let x = self.abs();
        if x < 2 {
            return x;
        }

        let shift = 32 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for i64 {
    fn abs_sqrt(self) -> Self {
        let x = self.abs();
        if x < 2 {
            return x;
        }

        let shift = 64 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for i128 {
    fn abs_sqrt(self) -> Self {
        let x = self.abs();
        if x < 2 {
            return x;
        }

        let shift = 128 - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for isize {
    fn abs_sqrt(self) -> Self {
        let x = self.abs();
        if x < 2 {
            return x;
        }

        let shift = 0isize.count_zeros() - x.leading_zeros();

        let mut guess = (x >> (shift / 2)) + 1;
        let mut res = (guess + x / guess) / 2;
        loop {
            if res > guess {
                if res - guess <= 1 {
                    break;
                }
            } else {
                if guess - res <= 1 {
                    break;
                }
            }
            guess = res;
            res = (guess + x / guess) / 2;
        }

        while res * res > x {
            res -= 1
        }
        res
    }
}

impl AbsSqrt for f32 {
    fn abs_sqrt(self) -> Self {
        self.abs().sqrt()
    }
}

impl AbsSqrt for f64 {
    fn abs_sqrt(self) -> Self {
        self.abs().sqrt()
    }
}
