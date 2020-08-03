use std::fmt;
use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shr, ShrAssign, Sub, SubAssign,
};
use std::str::FromStr;

use num::{Complex, One, Zero};

use rustnomial::binomial::LinearBinomial;
use rustnomial::degree::TermTokenizer;
use rustnomial::numerics::{Abs, AbsSqrt, IsNegativeOne, IsPositive};
use rustnomial::strings::{write_leading_term, write_trailing_term};
use rustnomial::traits::TermIterator;
use {Degree, Derivable, Evaluable, GenericPolynomial, Term};

#[macro_use] use ::fmt_poly;

#[derive(Debug, Clone)]
pub struct QuadraticTrinomial<N> {
    pub coefficients: [N; 3],
}

impl<N: Sized> QuadraticTrinomial<N> {
    /// Create a `Monomial` with coefficient and degree.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, Degree};
    /// let monomial = Monomial::new(3.0, 2);
    /// assert_eq!(3.0, monomial.coefficient);
    /// assert_eq!(Degree::Num(2), monomial.degree());
    /// ```
    pub fn new(coefficients: [N; 3]) -> QuadraticTrinomial<N> {
        QuadraticTrinomial { coefficients }
    }
}

impl<N: Copy + Zero> QuadraticTrinomial<N> {
    pub fn zero() -> Self {
        QuadraticTrinomial::new([N::zero(); 3])
    }
    /// Returns the degree of the `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, Degree};
    /// let monomial = Monomial::new(3.0, 2);
    /// assert_eq!(Degree::Num(2), monomial.degree());
    /// let zero_with_nonzero_deg = Monomial::new(0.0, 2);
    /// assert_eq!(Degree::NegInf, zero_with_nonzero_deg.degree());
    /// let nonzero_with_zero_degree = Monomial::new(1.0, 0);
    /// assert_eq!(Degree::Num(0), nonzero_with_zero_degree.degree());
    /// ```
    pub fn degree(&self) -> Degree {
        if !self.coefficients[0].is_zero() {
            Degree::Num(2)
        } else if !self.coefficients[1].is_zero() {
            Degree::Num(1)
        } else if !self.coefficients[2].is_zero() {
            Degree::Num(0)
        } else {
            Degree::NegInf
        }
    }

    /// Returns true if all terms are zero, and false if a non-zero term exists.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, Monomial};
    /// let zero = Monomial::new(0, 1);
    /// assert!(zero.is_zero());
    /// let non_zero = Monomial::new(1, 0);
    /// assert!(!non_zero.is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.degree() == Degree::NegInf
    }
}

impl<N> QuadraticTrinomial<N>
where
    N: Copy
        + Zero
        + Mul<Output = N>
        + Neg<Output = N>
        + Sub<Output = N>
        + From<u8>
        + Div<Output = N>
        + AbsSqrt
        + IsPositive
        + One,
{
    pub fn discriminant(&self) -> N {
        let [a, b, c] = self.coefficients;
        b * b - a * c * N::from(4)
    }

    /// Return the complex roots of `QuadraticTrinomial` with largest
    /// first, smallest second.
    pub fn complex_roots(&self) -> (Complex<N>, Complex<N>) {
        let discriminant = self.discriminant();
        let a = self.coefficients[0] * N::from(2);
        let b = -self.coefficients[1] / a;
        let sqrt = discriminant.abs_sqrt() / a;
        if discriminant.is_positive() {
            (
                Complex::new(b + sqrt, N::zero()),
                Complex::new(b - sqrt, N::zero()),
            )
        } else {
            (Complex::new(b, sqrt), Complex::new(b, -sqrt))
        }
    }

    pub fn real_roots(&self) -> Option<(N, N)> {
        let (root_a, root_b) = self.complex_roots();
        if root_a.im.is_zero() {
            Some((root_a.re, root_b.re))
        } else {
            None
        }
    }

    pub fn complex_factors(&self) -> (N, LinearBinomial<Complex<N>>, LinearBinomial<Complex<N>>) {
        let (root_a, root_b) = self.complex_roots();
        (
            self.coefficients[0],
            LinearBinomial::new([Complex::new(N::one(), N::zero()), root_a]),
            LinearBinomial::new([Complex::new(N::one(), N::zero()), root_b]),
        )
    }

    pub fn real_factors(&self) -> Option<(N, LinearBinomial<N>, LinearBinomial<N>)> {
        if let Some((root_a, root_b)) = self.real_roots() {
            Some((
                self.coefficients[0],
                LinearBinomial::new([N::one(), -root_a]),
                LinearBinomial::new([N::one(), -root_b]),
            ))
        } else {
            None
        }
    }
}

impl<N: Copy + Zero> GenericPolynomial<N> for QuadraticTrinomial<N> {
    /// Return the number of terms in `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, GenericPolynomial};
    /// let monomial = Monomial::new(3.0, 2);
    /// assert_eq!(1, monomial.len());
    /// assert_eq!(0, Monomial::<i32>::zero().len());
    /// ```
    fn len(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            3
        }
    }

    /// Returns the nth term of the `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, GenericPolynomial, Term};
    /// let monomial = Monomial::new(5, 2);
    /// assert_eq!(Term::Term(5, 2), monomial.nth_term(0));
    /// assert_eq!(Term::ZeroTerm, monomial.nth_term(1));
    /// ```
    fn nth_term(&self, index: usize) -> Term<N> {
        Term::new(self.coefficients[index], 2 - index)
    }

    /// Returns an iterator for the `Monomial`, yielding the term constant and degree. Terms are
    /// iterated over in descending degree order, excluding zero terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, GenericPolynomial};
    /// let monomial = Monomial::new(5, 2);
    /// let mut iter = monomial.term_iter();
    /// assert_eq!(Some((5, 2)), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn term_iter(&self) -> TermIterator<N> {
        TermIterator::new(self)
    }
}

impl<N> Evaluable<N> for QuadraticTrinomial<N>
where
    N: Add<Output = N> + Mul<Output = N> + Copy,
{
    /// Returns the value of the `Monomial` at the given point.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, Evaluable};
    /// let monomial = Monomial::new(5, 2);
    /// assert_eq!(125, monomial.eval(5));
    /// assert_eq!(1, Monomial::new(1, 0).eval(0));
    /// ```
    fn eval(&self, point: N) -> N {
        point * (self.coefficients[0] * point + self.coefficients[1]) + self.coefficients[2]
    }
}

impl<N> Derivable<N> for QuadraticTrinomial<N>
where
    N: Zero + One + Copy + Mul<Output = N> + From<u8>,
{
    /// Returns the derivative of the `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, Polynomial, Derivable};
    /// let monomial = Monomial::new(3.0, 2);
    /// assert_eq!(Monomial::new(6.0, 1), monomial.derivative());
    /// ```
    fn derivative(&self) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            N::zero(),
            self.coefficients[0] * N::from(2),
            self.coefficients[1],
        ])
    }
}

// impl<N> Integrable<N> for QuadraticTrinomial<N>
// where
//     N: Zero + Copy + AddAssign + Div<Output = N> + From<u8>,
// {
//     /// Returns the integral of the `Monomial`.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Monomial, Polynomial, Integrable};
//     /// let monomial = Monomial::new(3.0, 2);
//     /// let integral = monomial.integral();
//     /// assert_eq!(Polynomial::new(vec![1.0, 0.0, 0.0, 0.0]), integral.polynomial);
//     /// ```
//     fn integral(&self) -> Integral<N> {
//         match self.degree() {
//             Degree::NegInf => Integral {
//                 polynomial: Polynomial::new(vec![N::zero()]),
//             },
//             Degree::Num(x) => Integral {
//                 polynomial: Polynomial::from_terms(vec![(
//                     self.coefficient / N::from((x + 1) as u8),
//                     x + 1,
//                 )]),
//             },
//         }
//     }
// }

// impl<N: PowUsize + Copy> QuadraticTrinomial<N> {
//     /// Raises the `Monomial` to the power of exp.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::Monomial;
//     /// let monomial = Monomial::new(2, 1);
//     /// let monomial_sqr = monomial.pow(2);
//     /// let monomial_cub = monomial.pow(3);
//     /// assert_eq!(monomial.clone() * monomial.clone(), monomial_sqr);
//     /// assert_eq!(monomial_sqr.clone() * monomial.clone(), monomial_cub);
//     /// ```
//     pub fn pow(&self, exp: usize) -> QuadraticTrinomial<N> {
//         QuadraticTrinomial::new(self.coefficient.upow(exp), self.deg * exp)
//     }
// }

// TODO: Divmod implementation.

impl<N> PartialEq for QuadraticTrinomial<N>
where
    N: Zero + PartialEq + Copy,
{
    /// Returns true if this `Monomial` is equal to other.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Monomial;
    /// let a = Monomial::new(2, 2);
    /// let b = Monomial::new(2, 2);
    /// let c = Monomial::new(1, 2);
    /// assert_eq!(a, b);
    /// assert_ne!(a, c);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }
}

impl<N> FromStr for QuadraticTrinomial<N>
where
    N: Zero + One + Copy + AddAssign + FromStr,
{
    type Err = String;

    /// Returns a `Polynomial` with the corresponding terms,
    /// in order of ax^n + bx^(n-1) + ... + cx + d
    ///
    /// # Arguments
    ///
    /// * ` terms ` - A vector of constants, in decreasing order of degree.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Monomial;
    /// use std::str::FromStr;
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let monomial = Monomial::from_str("5x^2").unwrap();
    /// assert_eq!(Monomial::new(5, 2), monomial);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut polynomial = QuadraticTrinomial::zero();
        let mut has_iterated = false;
        for term in TermTokenizer::new(s).map(|s| Term::from_str(s.as_str())) {
            has_iterated = true;
            match term {
                Err(msg) => return Err(msg),
                Ok(Term::ZeroTerm) => {}
                Ok(Term::Term(coeff, deg)) => {
                    if deg <= 2 {
                        polynomial.coefficients[2 - deg] += coeff;
                    } else {
                        return Err("degree out of bounds".to_string());
                    }
                }
            }
        }

        if has_iterated {
            Ok(polynomial)
        } else {
            Err("Given string did not have any terms.".to_string())
        }
    }
}

impl<N> fmt::Display for QuadraticTrinomial<N>
where
    N: Zero + One + IsPositive + PartialEq + Abs + Copy + IsNegativeOne + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt_poly!(f, self)
    }
}

impl<N: Copy + Neg<Output = N>> Neg for QuadraticTrinomial<N> {
    type Output = QuadraticTrinomial<N>;

    fn neg(self) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            -self.coefficients[0],
            -self.coefficients[1],
            -self.coefficients[2],
        ])
    }
}

impl<N> Sub<QuadraticTrinomial<N>> for QuadraticTrinomial<N>
where
    N: Copy + Sub<Output = N>,
{
    type Output = QuadraticTrinomial<N>;

    fn sub(self, _rhs: QuadraticTrinomial<N>) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            self.coefficients[0] - _rhs.coefficients[0],
            self.coefficients[1] - _rhs.coefficients[1],
            self.coefficients[2] - _rhs.coefficients[2],
        ])
    }
}

impl<N> SubAssign<QuadraticTrinomial<N>> for QuadraticTrinomial<N>
where
    N: SubAssign + Copy,
{
    fn sub_assign(&mut self, _rhs: QuadraticTrinomial<N>) {
        self.coefficients[0] -= _rhs.coefficients[0];
        self.coefficients[1] -= _rhs.coefficients[1];
        self.coefficients[2] -= _rhs.coefficients[2];
    }
}

impl<N> Add<QuadraticTrinomial<N>> for QuadraticTrinomial<N>
where
    N: Add<Output = N> + Copy,
{
    type Output = QuadraticTrinomial<N>;

    fn add(self, _rhs: QuadraticTrinomial<N>) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            self.coefficients[0] + _rhs.coefficients[0],
            self.coefficients[1] + _rhs.coefficients[1],
            self.coefficients[2] + _rhs.coefficients[2],
        ])
    }
}

impl<N: Copy + AddAssign> AddAssign<QuadraticTrinomial<N>> for QuadraticTrinomial<N> {
    fn add_assign(&mut self, _rhs: QuadraticTrinomial<N>) {
        self.coefficients[0] += _rhs.coefficients[0];
        self.coefficients[1] += _rhs.coefficients[1];
        self.coefficients[2] += _rhs.coefficients[2];
    }
}

// impl<N: Copy + Mul<Output = N>> Mul<Monomial<N>> for Monomial<N> {
//     type Output = Monomial<N>;
//
//     fn mul(self, _rhs: Monomial<N>) -> Monomial<N> {
//         Monomial::new(self.coefficient * _rhs.coefficient, self.deg + _rhs.deg)
//     }
// }
//
// impl<N: MulAssign + AddAssign> MulAssign<Monomial<N>> for Monomial<N> {
//     fn mul_assign(&mut self, _rhs: Monomial<N>) {
//         self.coefficient *= _rhs.coefficient;
//         self.deg += _rhs.deg;
//     }
// }
//
// impl<N: Copy + Mul<Output = N>> Mul<&Monomial<N>> for Monomial<N> {
//     type Output = Monomial<N>;
//
//     fn mul(self, _rhs: &Monomial<N>) -> Monomial<N> {
//         Monomial::new(self.coefficient * _rhs.coefficient, self.deg + _rhs.deg)
//     }
// }
//
// impl<N> MulAssign<&Monomial<N>> for Monomial<N>
// where
//     N: MulAssign + AddAssign + Copy,
// {
//     fn mul_assign(&mut self, _rhs: &Monomial<N>) {
//         self.coefficient *= _rhs.coefficient;
//         self.deg += _rhs.deg;
//     }
// }
//
impl<N: Mul<Output = N> + Copy> Mul<N> for QuadraticTrinomial<N> {
    type Output = QuadraticTrinomial<N>;

    fn mul(self, _rhs: N) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            self.coefficients[0] * _rhs,
            self.coefficients[1] * _rhs,
            self.coefficients[2] * _rhs,
        ])
    }
}

impl<N: MulAssign + Copy> MulAssign<N> for QuadraticTrinomial<N> {
    fn mul_assign(&mut self, _rhs: N) {
        self.coefficients[0] *= _rhs;
        self.coefficients[1] *= _rhs;
        self.coefficients[2] *= _rhs;
    }
}

impl<N: Div<Output = N> + Copy> Div<N> for QuadraticTrinomial<N> {
    type Output = QuadraticTrinomial<N>;

    fn div(self, _rhs: N) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            self.coefficients[0] / _rhs,
            self.coefficients[1] / _rhs,
            self.coefficients[2] / _rhs,
        ])
    }
}

impl<N: DivAssign + Copy> DivAssign<N> for QuadraticTrinomial<N> {
    fn div_assign(&mut self, _rhs: N) {
        self.coefficients[0] /= _rhs;
        self.coefficients[1] /= _rhs;
        self.coefficients[2] /= _rhs;
    }
}

// impl<N: Zero + Copy> Shl<i32> for QuadraticTrinomial<N> {
//     type Output = QuadraticTrinomial<N>;
//
//     fn shl(self, _rhs: i32) -> QuadraticTrinomial<N> {
//         if _rhs < 0 {
//             self >> -_rhs
//         } else {
//             match _rhs {
//                 0 => {
//                     QuadraticTrinomial::new(self.coefficients.clone())
//                 }
//                 1 => {
//                     QuadraticTrinomial::new([self.coefficients[1], self.coefficients[2], N::zero()])
//                 }
//                 2 => {
//                     QuadraticTrinomial::new([self.coefficients[2], N::zero(), N::zero()])
//                 }
//                 _ => {QuadraticTrinomial::zero()}
//             }
//         }
//     }
// }
//
// impl<N: Zero + Copy> ShlAssign<i32> for QuadraticTrinomial<N> {
//     fn shl_assign(&mut self, _rhs: i32) {
//         if _rhs < 0 {
//             *self >>= -_rhs;
//         } else {
//             match _rhs {
//                 0 => {}
//                 1 => {
//                     self.coefficients[0] = self.coefficients[1];
//                     self.coefficients[1] = self.coefficients[2];
//                     self.coefficients[2] = N::zero();
//                 }
//                 2 => {
//                     self.coefficients[0] = self.coefficients[2];
//                     self.coefficients[1] = N::zero();
//                     self.coefficients[2] = N::zero();
//                 }
//                 _ => {
//                     self.coefficients[0] = N::zero();
//                     self.coefficients[1] = N::zero();
//                     self.coefficients[2] = N::zero();
//                 }
//
//             }
//         }
//     }
// }

impl<N: Zero + Copy> Shr<u32> for QuadraticTrinomial<N> {
    type Output = QuadraticTrinomial<N>;

    fn shr(self, _rhs: u32) -> QuadraticTrinomial<N> {
        match _rhs {
            0 => QuadraticTrinomial::new(self.coefficients.clone()),
            1 => QuadraticTrinomial::new([N::zero(), self.coefficients[0], self.coefficients[1]]),
            2 => QuadraticTrinomial::new([N::zero(), N::zero(), self.coefficients[0]]),
            _ => QuadraticTrinomial::zero(),
        }
    }
}

impl<N: Zero + Copy> ShrAssign<u32> for QuadraticTrinomial<N> {
    fn shr_assign(&mut self, _rhs: u32) {
        match _rhs {
            0 => {}
            1 => {
                self.coefficients[2] = self.coefficients[1];
                self.coefficients[1] = self.coefficients[0];
                self.coefficients[0] = N::zero();
            }
            2 => {
                self.coefficients[2] = self.coefficients[0];
                self.coefficients[1] = N::zero();
                self.coefficients[0] = N::zero();
            }
            _ => {
                self.coefficients[0] = N::zero();
                self.coefficients[1] = N::zero();
                self.coefficients[2] = N::zero();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use num::Complex;
    use rustnomial::trinomial::QuadraticTrinomial;
    use {Derivable, Evaluable};

    #[test]
    fn test_eval() {
        let a = QuadraticTrinomial::new([5, 0, 0]);
        assert_eq!(a.eval(5), 125);
    }
    //
    // #[test]
    // fn test_shl_pos() {
    //     let a = Monomial::new(1, 2);
    //     let c = Monomial::new(1, 7);
    //     assert_eq!(a << 5, c);
    // }
    //
    // #[test]
    // fn test_shl_assign_pos() {
    //     let mut a = Monomial::new(1, 2);
    //     let c = Monomial::new(1, 7);
    //     a <<= 5;
    //     assert_eq!(a, c);
    // }
    //
    // #[test]
    // fn test_shl_neg() {
    //     let a = Monomial::new(1, 7);
    //     let c = Monomial::new(1, 2);
    //     assert_eq!(a << -5, c);
    // }
    //
    // #[test]
    // fn test_shl_assign_neg() {
    //     let mut a = Monomial::new(1, 7);
    //     let c = Monomial::new(1, 2);
    //     a <<= -5;
    //     assert_eq!(a, c);
    // }

    #[test]
    fn test_shr_pos() {
        let a = QuadraticTrinomial::new([1, 0, 0]);
        let c = QuadraticTrinomial::new([0, 0, 1]);
        assert_eq!(a >> 2, c);
    }

    #[test]
    fn test_shr_assign_pos() {
        let mut a = QuadraticTrinomial::new([1, 0, 0]);
        let c = QuadraticTrinomial::new([0, 0, 1]);
        a >>= 2;
        assert_eq!(a, c);
    }

    // #[test]
    // fn test_shr_neg() {
    //     let a = Monomial::new(1, 2);
    //     let c = Monomial::new(1, 7);
    //     assert_eq!(a >> -5, c);
    // }
    //
    // #[test]
    // fn test_shr_assign_neg() {
    //     let mut a = Monomial::new(1, 2);
    //     let c = Monomial::new(1, 7);
    //     a >>= -5;
    //     assert_eq!(a, c);
    // }

    #[test]
    fn test_shr_to_zero() {
        let a = QuadraticTrinomial::new([1, 2, 3]);
        assert_eq!(a >> 5, QuadraticTrinomial::zero());
    }

    #[test]
    fn test_shr_assign_to_zero() {
        let mut a = QuadraticTrinomial::new([1, 2, 3]);
        a >>= 5;
        assert_eq!(a, QuadraticTrinomial::zero());
    }

    #[test]
    fn test_derivative_of_zero() {
        let a: QuadraticTrinomial<i32> = QuadraticTrinomial::zero();
        assert_eq!(a.derivative(), QuadraticTrinomial::zero());
    }

    #[test]
    fn test_derivative_of_degree_zero() {
        let a = QuadraticTrinomial::new([0, 0, 1]);
        assert_eq!(a.derivative(), QuadraticTrinomial::zero());
    }

    #[test]
    fn test_derivative() {
        let a = QuadraticTrinomial::new([1, 2, 3]);
        assert_eq!(a.derivative(), QuadraticTrinomial::new([0, 2, 2]));
    }

    #[test]
    fn test_complex_roots_pos() {
        let a = QuadraticTrinomial::new([1, 4, 4]);
        let c = (Complex::new(-2i16, 0), Complex::new(-2i16, 0));
        assert_eq!(a.complex_roots(), c);
    }

    #[test]
    fn test_complex_roots_neg() {
        let a = QuadraticTrinomial::new([1, 0, 4]);
        let c = (Complex::new(0, 2i16), Complex::new(0, -2i16));
        assert_eq!(a.complex_roots(), c);
    }
}
