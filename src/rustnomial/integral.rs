use std::fmt;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, Mul, MulAssign, Sub};

use num::{One, Zero};

use rustnomial::numerics::{Abs, IsNegativeOne, IsPositive};
use {Evaluable, GenericPolynomial, Polynomial};

#[macro_export]
macro_rules! integral {
    ( $( $x:expr ),* ) => {
        {
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
    N: IsPositive + Zero + One + Copy + IsNegativeOne + PartialEq + PartialOrd + Display + Abs,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.polynomial.len() == 0 {
            return write!(f, "C");
        }

        match self.polynomial.fmt(f) {
            Ok(_) => write!(f, " + C"),
            Err(e) => Err(e),
        }
    }
}

impl<N: Zero + Copy + AddAssign> Integral<N> {
    pub fn replace_c(&self, c: N) -> Polynomial<N> {
        let mut terms: Vec<(N, usize)> = self.polynomial.term_iter().collect();
        terms.push((c, 0));
        Polynomial::from_terms(terms)
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
