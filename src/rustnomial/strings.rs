use std::fmt;
use std::fmt::Display;
use rustnomial::numerics::{IsNegativeOne, Abs, IsPositive};
use num::One;

pub(crate) fn write_leading_term<N>(f: &mut fmt::Formatter, coeff: N, degree: usize) -> fmt::Result
    where N: Display + One + IsNegativeOne + PartialEq + Copy {
        if degree == 0 {
            return write!(f, "{}", coeff);
        }

        if coeff.is_negative_one() {
            write!(f, "-")?;
        } else if !coeff.is_one() {
            write!(f, "{}", coeff)?;
        }

        if degree > 1 {
            write!(f, "x^{}", degree)
        } else {
            write!(f, "x")
        }
}

pub(crate) fn write_trailing_term<N>(f: &mut fmt::Formatter, coeff: N, degree: usize) -> fmt::Result
    where N: Display + One + IsPositive + PartialEq + Copy + Abs {
        if coeff.is_positive() {
            write!(f, " + ")?;
        } else {
            write!(f, " - ")?;
        }

        let coeff = coeff.abs();

        if degree == 0 {
            write!(f, "{}", coeff)
        } else {
            if !coeff.is_one() {
                write!(f, "{}", coeff)?;
            }

            if degree > 1 {
                write!(f, "x^{}", degree)
            } else {
                write!(f, "x")
            }
        }
}

