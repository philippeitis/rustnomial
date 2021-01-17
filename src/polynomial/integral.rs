use core::fmt;
use core::fmt::{Debug, Display};
use core::marker::PhantomData;
use core::ops::Sub;

use num::{One, Zero};

use crate::numerics::{Abs, IsNegativeOne, IsPositive};
use crate::{Evaluable, FreeSizePolynomial, SizedPolynomial};

#[macro_export]
macro_rules! integral {
    ( $( $x:expr ),* ) => {
        {
            use $crate::{Polynomial, Integrable};
            Polynomial::new(vec![$($x,)*]).integral()
        }
    };
}

#[derive(Debug, Clone)]
/// A type which intreprets a polynomial as an integral, and ensures
/// consistent use of the underlying polynomial as an integral.
pub struct Integral<N, P: FreeSizePolynomial<N> + Evaluable<N>> {
    polynomial: P,
    phantom: PhantomData<N>,
}

impl<N, P: FreeSizePolynomial<N> + Evaluable<N>> Integral<N, P> {
    pub fn new(polynomial: P) -> Self {
        Integral {
            polynomial,
            phantom: PhantomData,
        }
    }

    pub fn inner(&self) -> &P {
        &self.polynomial
    }
}

pub trait Integrable<N, P: FreeSizePolynomial<N> + Evaluable<N>> {
    fn integral(&self) -> Integral<N, P>;
}

impl<N, P: FreeSizePolynomial<N> + Evaluable<N> + SizedPolynomial<N> + Display> Display
    for Integral<N, P>
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

impl<N, P: FreeSizePolynomial<N> + Evaluable<N> + Clone> Integral<N, P> {
    pub fn replace_c(&self, c: N) -> P {
        let mut p = self.polynomial.clone();
        p.add_term(c, 0);
        p
    }
}

impl<N: Sub<Output = N>, P: FreeSizePolynomial<N> + Evaluable<N>> Integral<N, P> {
    /// Returns the area of the integral from the first point to the second point.
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
    use crate::integral;
    use alloc::string::ToString;

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
