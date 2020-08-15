use std::fmt::{Debug, Display};
use std::ops::{AddAssign, Mul, MulAssign, Sub};
use std::{fmt, iter};

use num::{One, Zero};

use rustnomial::numerics::{Abs, IsNegativeOne, IsPositive};
use {Evaluable, FreeSizePolynomial, Polynomial, SizedPolynomial};

#[macro_export]
macro_rules! integral {
    ( $( $x:expr ),* ) => {
        {
            use $crate::{Polynomial, Integrable};
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            Polynomial::new(temp_vec).integral()
        }
    };
}

#[derive(Debug, Clone)]
pub struct Integral<N> {
    pub polynomial: Polynomial<N>,
}

pub trait Integrable<N> {
    fn integral(&self) -> Integral<N>;
}

impl<N> Display for Integral<N>
where
    N: IsPositive + Zero + One + Copy + IsNegativeOne + PartialEq + Display + Abs,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.polynomial.is_zero() {
            write!(f, "C")
        } else {
            write!(f, "{} + C", self.polynomial)
        }
    }
}

impl<N: Zero + Copy + AddAssign> Integral<N> {
    pub fn replace_c(&self, c: N) -> Polynomial<N> {
        Polynomial::from_terms(
            self.polynomial
                .term_iter()
                .chain(iter::once((c, 0)))
                .collect(),
        )
    }
}

impl<N> Integral<N>
where
    N: Zero + One + Copy + AddAssign + MulAssign + Mul<Output = N> + Sub<Output = N>,
{
    /// Returns the area of the underlying `Polynomial` from the first point to the second point.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, Integrable};
    /// let polynomial = Polynomial::new(vec![2.0, 1.0]);
    /// let integral = polynomial.integral();
    /// assert_eq!(2.0, integral.eval(0.0, 1.0));
    /// assert_eq!(6.0, integral.eval(0.0, 2.0));
    /// assert_eq!(4.0, integral.eval(1.0, 2.0));
    /// ```
    pub fn eval(&self, start: N, end: N) -> N {
        self.polynomial.eval(end) - self.polynomial.eval(start)
    }
}

#[cfg(test)]
mod test {
    use integral;

    #[test]
    fn test_integral_empty_polynomial() {
        let integral = integral!(0);
        assert_eq!("C", integral.to_string());
    }

    #[test]
    fn test_integral_str() {
        let integral = integral!(6, 4, 2);
        assert_eq!("2x^3 + 2x^2 + 2x + C", integral.to_string());
    }

    #[test]
    fn test_integral_str_negatives() {
        let a = integral![-3, -2, 1];
        assert_eq!("-x^3 - x^2 + x + C", a.to_string());
    }
}
