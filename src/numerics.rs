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

pub trait CanNegate {
    fn can_negate() -> bool;
}

macro_rules! can_negate {
    ($T:ty, $C:expr) => {
        impl CanNegate for $T {
            fn can_negate() -> bool {
                $C
            }
        }
    };
}

can_negate!(u8, false);
can_negate!(u16, false);
can_negate!(u32, false);
can_negate!(u64, false);
can_negate!(u128, false);
can_negate!(usize, false);
can_negate!(i8, true);
can_negate!(i16, true);
can_negate!(i32, true);
can_negate!(i64, true);
can_negate!(i128, true);
can_negate!(isize, true);
can_negate!(f32, true);
can_negate!(f64, true);

// /// Specifies a conversion from N to self such that any number
// /// below N also has an exact representation.
// pub trait TryFromExact<N>: Sized {
//     type Error;
//     fn try_from_exact(num: N) -> Result<Self, Self::Error>;
// }
//
// pub enum ConversionError {
//     NotRepresentable,
//     Overflow,
// }
//
// macro_rules! try_from_exact_big_from_small {
//     ($B:ty, $S:ty) => {
//         impl TryFromExact<$S> for $B {
//             type Error = ConversionError;
//             fn try_from_exact(num: $S) -> Result<Self, Self::Error> {
//                 Ok(num as $B)
//             }
//         }
//     };
// }
//
// upcast!(try_from_exact_big_from_small);
//
// macro_rules! try_from_exact_target_to_float {
//     ($N:ty, $F:ty) => {
//         impl TryFromExact<$N> for $F {
//             type Error = ConversionError;
//             fn try_from_exact(num: $N) -> Result<Self, Self::Error> {
//                 let conv = num as $F;
//                 if (conv as $N) != num {
//                     Err(ConversionError::NotRepresentable)
//                 } else {
//                     Ok(conv)
//                 }
//             }
//         }
//     };
// }
//
// try_from_exact_target_to_float!(u32, f32);
// try_from_exact_target_to_float!(u64, f32);
// try_from_exact_target_to_float!(u128, f32);
// try_from_exact_target_to_float!(i32, f32);
// try_from_exact_target_to_float!(i64, f32);
// try_from_exact_target_to_float!(i128, f32);
//
// try_from_exact_target_to_float!(u64, f64);
// try_from_exact_target_to_float!(u128, f64);
// try_from_exact_target_to_float!(i64, f64);
// try_from_exact_target_to_float!(i128, f64);
//
// macro_rules! try_from_exact_big_to_small {
//     ($B:ty, $S:ty) => {
//         impl TryFromExact<$B> for $S {
//             type Error = ConversionError;
//             fn try_from_exact(num: $B) -> Result<Self, Self::Error> {
//                 if num > <$S>::MAX as $B {
//                     Err(ConversionError::Overflow)
//                 } else if num < <$S>::MIN as $B {
//                     Err(ConversionError::Overflow)
//                 } else {
//                     Ok(num as $S)
//                 }
//             }
//         }
//     };
// }
//
// try_from_exact_big_to_small!(usize, u8);
// try_from_exact_big_to_small!(usize, u16);
// try_from_exact_big_to_small!(usize, u32);
// try_from_exact_big_to_small!(usize, u64);
// try_from_exact_big_to_small!(usize, i8);
// try_from_exact_big_to_small!(usize, i16);
// try_from_exact_big_to_small!(usize, i32);
// try_from_exact_big_to_small!(usize, i64);

/// Specifies a conversion from usize to self such that subtracting by one will yield
/// an exact representation in usize, down to 0.
pub trait TryFromUsizeContinuous: Sized {
    fn try_from_usize_cont(num: usize) -> Result<Self, ConversionError>;
}

#[derive(Debug)]
pub enum ConversionError {
    Overflow,
}

impl TryFromUsizeContinuous for f32 {
    fn try_from_usize_cont(num: usize) -> Result<Self, ConversionError> {
        if num > 2 << 23 {
            Err(ConversionError::Overflow)
        } else {
            Ok(num as f32)
        }
    }
}

impl TryFromUsizeContinuous for f64 {
    fn try_from_usize_cont(num: usize) -> Result<Self, ConversionError> {
        if num > 2 << 51 {
            Err(ConversionError::Overflow)
        } else {
            Ok(num as f64)
        }
    }
}

macro_rules! try_from_continuous_unsigned {
    ($S:ty) => {
        impl TryFromUsizeContinuous for $S {
            fn try_from_usize_cont(num: usize) -> Result<Self, ConversionError> {
                if core::mem::size_of::<Self>() >= core::mem::size_of::<usize>() {
                    Ok(num as $S)
                } else if num > <$S>::MAX as usize {
                    Err(ConversionError::Overflow)
                } else {
                    Ok(num as $S)
                }
            }
        }
    };
}

macro_rules! try_from_continuous_signed {
    ($S:ty) => {
        impl TryFromUsizeContinuous for $S {
            fn try_from_usize_cont(num: usize) -> Result<Self, ConversionError> {
                if core::mem::size_of::<Self>() > core::mem::size_of::<usize>() {
                    Ok(num as $S)
                } else if num > <$S>::MAX as usize {
                    Err(ConversionError::Overflow)
                } else {
                    Ok(num as $S)
                }
            }
        }
    };
}

try_from_continuous_unsigned!(u8);
try_from_continuous_unsigned!(u16);
try_from_continuous_unsigned!(u32);
try_from_continuous_unsigned!(u64);
try_from_continuous_unsigned!(u128);
try_from_continuous_signed!(i8);
try_from_continuous_signed!(i16);
try_from_continuous_signed!(i32);
try_from_continuous_signed!(i64);
try_from_continuous_signed!(i128);

/// Specifies a conversion from usize to self such that an exact representation of the usize
/// is returned.
pub trait TryFromUsizeExact: Sized {
    fn try_from_usize_exact(num: usize) -> Result<Self, ExactConversionError>;
}

#[derive(Debug)]
pub enum ExactConversionError {
    Overflow,
    Unrepresentable,
}

impl TryFromUsizeExact for f32 {
    fn try_from_usize_exact(num: usize) -> Result<Self, ExactConversionError> {
        let conv = num as f32;
        if conv as usize == num {
            Ok(conv)
        } else {
            Err(ExactConversionError::Unrepresentable)
        }
    }
}

impl TryFromUsizeExact for f64 {
    fn try_from_usize_exact(num: usize) -> Result<Self, ExactConversionError> {
        let conv = num as f64;
        if conv as usize == num {
            Ok(conv)
        } else {
            Err(ExactConversionError::Unrepresentable)
        }
    }
}

macro_rules! try_from_exact_unsigned {
    ($S:ty) => {
        impl TryFromUsizeExact for $S {
            fn try_from_usize_exact(num: usize) -> Result<Self, ExactConversionError> {
                if core::mem::size_of::<Self>() >= core::mem::size_of::<usize>() {
                    Ok(num as $S)
                } else if num > <$S>::MAX as usize {
                    Err(ExactConversionError::Overflow)
                } else {
                    Ok(num as $S)
                }
            }
        }
    };
}

macro_rules! try_from_exact_signed {
    ($S:ty) => {
        impl TryFromUsizeExact for $S {
            fn try_from_usize_exact(num: usize) -> Result<Self, ExactConversionError> {
                if core::mem::size_of::<Self>() > core::mem::size_of::<usize>() {
                    Ok(num as $S)
                } else if num > <$S>::MAX as usize {
                    Err(ExactConversionError::Overflow)
                } else {
                    Ok(num as $S)
                }
            }
        }
    };
}

try_from_exact_unsigned!(u8);
try_from_exact_unsigned!(u16);
try_from_exact_unsigned!(u32);
try_from_exact_unsigned!(u64);
try_from_exact_unsigned!(u128);
try_from_exact_signed!(i8);
try_from_exact_signed!(i16);
try_from_exact_signed!(i32);
try_from_exact_signed!(i64);
try_from_exact_signed!(i128);
