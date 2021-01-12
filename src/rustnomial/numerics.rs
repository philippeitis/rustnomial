use num::{One, Zero};

pub trait IsNegativeOne {
    /// Returns true is self is equal to negative one. Used for string formatting purposes,
    /// and does not respect tolerances.
    fn is_negative_one(&self) -> bool;
}

macro_rules! is_neg_one_u {
    ($U:ty) => {
        impl IsNegativeOne for $U {
            fn is_negative_one(&self) -> bool {
                false
            }
        }
    };
}

macro_rules! is_neg_one {
    ($U:ty) => {
        impl IsNegativeOne for $U {
            #[allow(clippy::float_cmp)]
            fn is_negative_one(&self) -> bool {
                *self == -<$U>::one()
            }
        }
    };
}

is_neg_one_u!(u8);
is_neg_one_u!(u16);
is_neg_one_u!(u32);
is_neg_one_u!(u64);
is_neg_one_u!(u128);
is_neg_one_u!(usize);
is_neg_one!(i8);
is_neg_one!(i16);
is_neg_one!(i32);
is_neg_one!(i64);
is_neg_one!(i128);
is_neg_one!(isize);
is_neg_one!(f32);
is_neg_one!(f64);

pub trait Abs {
    fn abs(self) -> Self;
}

macro_rules! abs_u {
    ($U:ty) => {
        impl Abs for $U {
            fn abs(self) -> Self {
                self
            }
        }
    };
}

macro_rules! abs {
    ($U:ty) => {
        impl Abs for $U {
            fn abs(self) -> Self {
                self.abs()
            }
        }
    };
}

abs_u!(u8);
abs_u!(u16);
abs_u!(u32);
abs_u!(u64);
abs_u!(u128);
abs_u!(usize);
abs!(i8);
abs!(i16);
abs!(i32);
abs!(i64);
abs!(i128);
abs!(isize);
abs!(f32);
abs!(f64);

pub trait PowUsize {
    fn upow(self, exp: usize) -> Self;
}

macro_rules! pow_u {
    ($T:ty) => {
        impl PowUsize for $T {
            fn upow(self, exp: usize) -> Self {
                self.pow(exp as u32)
            }
        }
    };
}

pow_u!(u8);
pow_u!(u16);
pow_u!(u32);
pow_u!(u64);
pow_u!(u128);
pow_u!(usize);
pow_u!(i8);
pow_u!(i16);
pow_u!(i32);
pow_u!(i64);
pow_u!(i128);
pow_u!(isize);

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

macro_rules! is_positive {
    ($T:ty) => {
        impl IsPositive for $T {
            fn is_positive(&self) -> bool {
                self > &<$T>::zero()
            }
        }
    };
}

is_positive!(u8);
is_positive!(u16);
is_positive!(u32);
is_positive!(u64);
is_positive!(u128);
is_positive!(usize);
is_positive!(i8);
is_positive!(i16);
is_positive!(i32);
is_positive!(i64);
is_positive!(i128);
is_positive!(isize);
is_positive!(f32);
is_positive!(f64);

pub trait AbsSqrt {
    fn abs_sqrt(self) -> Self;
}

macro_rules! abs_sqrt_u {
    ($U:ty) => {
        impl AbsSqrt for $U {
            fn abs_sqrt(self) -> Self {
                if self < 2 {
                    return self;
                }

                let shift = <$U>::zero().count_zeros() - self.leading_zeros();

                let mut guess = (self >> (shift / 2)) + 1;
                let mut res = (guess + self / guess) / 2;
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
                    res = (guess + self / guess) / 2;
                }

                while res * res > self {
                    res -= 1
                }
                res
            }
        }
    };
}

macro_rules! abs_sqrt_i {
    ($T:ty, $U:ty) => {
        impl AbsSqrt for $T {
            fn abs_sqrt(self) -> Self {
                (self.abs() as $U).abs_sqrt() as $T
            }
        }
    };
}

abs_sqrt_u!(u8);
abs_sqrt_u!(u16);
abs_sqrt_u!(u32);
abs_sqrt_u!(u64);
abs_sqrt_u!(u128);
abs_sqrt_u!(usize);
abs_sqrt_i!(i8, u8);
abs_sqrt_i!(i16, u16);
abs_sqrt_i!(i32, u32);
abs_sqrt_i!(i64, u64);
abs_sqrt_i!(i128, u128);
abs_sqrt_i!(isize, usize);

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

pub trait Cbrt {
    fn cbrt(self) -> Self;
}

impl Cbrt for f32 {
    fn cbrt(self) -> Self {
        self.cbrt()
    }
}

impl Cbrt for f64 {
    fn cbrt(self) -> Self {
        self.cbrt()
    }
}
