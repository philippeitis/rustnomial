use std::fmt;
use std::fmt::Display;
use std::ops;
use std::ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg};
use std::str::FromStr;

use num::{One, Zero};

use rustnomial::numerics::{Abs, IsNegativeOne, PowUsize};
use rustnomial::traits::TermIterator;
use {Degree, Derivable, Evaluable, GenericPolynomial, Integrable, Integral, Polynomial, Term};

#[derive(Debug, Clone)]
pub struct Monomial<N> {
    pub coefficient: N,
    pub deg: usize,
}

impl<N> Monomial<N> {
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
    pub fn new(coefficient: N, degree: usize) -> Monomial<N> {
        Monomial {
            coefficient,
            deg: degree,
        }
    }
}

impl<N: Copy + Zero> Monomial<N> {
    pub fn zero() -> Self {
        Monomial::new(N::zero(), 0)
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
        if self.coefficient.is_zero() {
            Degree::NegInf
        } else {
            Degree::Num(self.deg)
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

impl<N: Copy + Zero> GenericPolynomial<N> for Monomial<N> {
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
            1
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
        if index != 0 {
            Term::ZeroTerm
        } else {
            Term::new(self.coefficient, self.deg)
        }
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

impl<N> Evaluable<N> for Monomial<N>
where
    N: PowUsize + Mul<Output = N> + Copy,
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
        self.coefficient * point.upow(self.deg)
    }
}

impl<N> Derivable<N> for Monomial<N>
where
    N: Zero + Copy + Mul<Output = N> + From<u8>,
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
    fn derivative(&self) -> Monomial<N> {
        match self.degree() {
            Degree::NegInf | Degree::Num(0) => Monomial::zero(),
            Degree::Num(x) => Monomial::new(self.coefficient * N::from(x as u8), x - 1),
        }
    }
}

impl<N> Integrable<N> for Monomial<N>
where
    N: Zero + Copy + AddAssign + Div<Output = N> + From<u8>,
{
    /// Returns the integral of the `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, Polynomial, Integrable};
    /// let monomial = Monomial::new(3.0, 2);
    /// let integral = monomial.integral();
    /// assert_eq!(Polynomial::new(vec![1.0, 0.0, 0.0, 0.0]), integral.polynomial);
    /// ```
    fn integral(&self) -> Integral<N> {
        match self.degree() {
            Degree::NegInf => Integral {
                polynomial: Polynomial::new(vec![N::zero()]),
            },
            Degree::Num(x) => Integral {
                polynomial: Polynomial::from_terms(vec![(
                    self.coefficient / N::from((x + 1) as u8),
                    x + 1,
                )]),
            },
        }
    }
}

impl<N: PowUsize + Copy> Monomial<N> {
    /// Raises the `Monomial` to the power of exp.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Monomial;
    /// let monomial = Monomial::new(2, 1);
    /// let monomial_sqr = monomial.pow(2);
    /// let monomial_cub = monomial.pow(3);
    /// assert_eq!(monomial.clone() * monomial.clone(), monomial_sqr);
    /// assert_eq!(monomial_sqr.clone() * monomial.clone(), monomial_cub);
    /// ```
    pub fn pow(&self, exp: usize) -> Monomial<N> {
        Monomial::new(self.coefficient.upow(exp), self.deg * exp)
    }
}

// TODO: Divmod implementation.

impl<N> PartialEq for Monomial<N>
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
        self.nth_term(0) == other.nth_term(0)
    }
}

impl<N> FromStr for Monomial<N>
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
        match Term::from_str(s) {
            Ok(Term::ZeroTerm) => Ok(Monomial::zero()),
            Ok(Term::Term(coeff, deg)) => Ok(Monomial::new(coeff, deg)),
            Err(e) => Err(e),
        }
    }
}

impl<N> fmt::Display for Monomial<N>
where
    N: Zero + One + Copy + IsNegativeOne + PartialEq + PartialOrd + Display + Abs,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut iter = self.term_iter();
        return match iter.next() {
            None => write!(f, "0"),
            Some((coeff, degree)) => {
                if coeff.is_negative_one() {
                    write!(f, "-")?;
                } else if (!coeff.is_one()) || (degree == 0) {
                    write!(f, "{}", coeff)?;
                }

                match degree {
                    0 => write!(f, ""),
                    1 => write!(f, "x"),
                    _ => write!(f, "x^{}", degree),
                }
            }
        };
    }
}

impl<N: Copy + Neg<Output = N>> ops::Neg for Monomial<N> {
    type Output = Monomial<N>;

    fn neg(self) -> Monomial<N> {
        Monomial::new(-self.coefficient, self.deg)
    }
}

// impl<N> ops::Sub<Polynomial<N>> for Polynomial<N>
//     where N: PartialEq + Zero + Copy + Sub<Output=N> + SubAssign + Neg<Output=N>{
//     type Output = Polynomial<N>;
//
//     fn sub(self, _rhs: Polynomial<N>) -> Polynomial<N> {
//         if _rhs.len() > self.len() {
//             let mut terms = _rhs.terms.clone();
//             let offset = _rhs.len() - self.len();
//
//             for index in terms[..offset].iter_mut() {
//                 *index = -*index;
//             }
//
//             for (index, val) in terms[offset..].iter_mut().zip(self.terms) {
//                 *index = val - *index;
//             }
//             Polynomial::new(terms)
//         } else {
//             let mut terms = self.terms.clone();
//             let offset = terms.len() - _rhs.len();
//             for (index, val) in terms[offset..].iter_mut().zip(_rhs.terms) {
//                 *index -= val;
//             }
//             Polynomial::new(terms)
//         }
//     }
// }

// impl<N> ops::SubAssign<Polynomial<N>> for Polynomial<N>
//     where N: Neg<Output=N> + Sub<Output=N> + SubAssign + Copy + Zero + PartialEq {
//     fn sub_assign(&mut self, _rhs: Polynomial<N>) {
//         if _rhs.len() > self.len() {
//             let mut terms = _rhs.terms.clone();
//             let offset = _rhs.len() - self.len();
//
//             for index in terms[..offset].iter_mut() {
//                 *index = -*index;
//             }
//
//             for (index, &val) in terms[offset..].iter_mut().zip(&self.terms) {
//                 *index = val - *index;
//             }
//             self.terms = terms;
//         } else {
//             let offset = self.len() - _rhs.len();
//             for (index, val) in self.terms[offset..].iter_mut().zip(_rhs.terms) {
//                 *index -= val;
//             }
//         }
//     }
// }

// impl<N> ops::Add<Polynomial<N>> for Polynomial<N>
//     where N: PartialEq + Zero + Copy + AddAssign {
//     type Output = Polynomial<N>;
//
//     fn add(self, _rhs: Polynomial<N>) -> Polynomial<N> {
//         let (mut terms, small) = if _rhs.len() > self.len() {
//             (_rhs.terms.clone(), &self.terms)
//         } else {
//             (self.terms.clone(), &_rhs.terms)
//         };
//
//         let offset = terms.len() - small.len();
//
//         for (index, &val) in terms[offset..].iter_mut().zip(small) {
//             *index += val;
//         }
//
//         Polynomial::new(terms)
//     }
// }

// impl<N: Copy + Zero + PartialEq + AddAssign> ops::AddAssign<Polynomial<N>> for Polynomial<N> {
//     fn add_assign(&mut self, _rhs: Polynomial<N>) {
//         if _rhs.len() > self.len() {
//             let offset = _rhs.len() - self.len();
//             let mut terms = _rhs.terms.clone();
//             for (index, &val) in terms[offset..].iter_mut().zip(&self.terms) {
//                 *index += val;
//             }
//             self.terms = terms;
//         } else {
//             let offset = self.len() - _rhs.len();
//             for (index, val) in self.terms[offset..].iter_mut().zip(_rhs.terms) {
//                 *index += val;
//             }
//         }
//     }
// }

impl<N: Copy + Mul<Output = N>> ops::Mul<Monomial<N>> for Monomial<N> {
    type Output = Monomial<N>;

    fn mul(self, _rhs: Monomial<N>) -> Monomial<N> {
        Monomial::new(self.coefficient * _rhs.coefficient, self.deg + _rhs.deg)
    }
}

impl<N: MulAssign + AddAssign> ops::MulAssign<Monomial<N>> for Monomial<N> {
    fn mul_assign(&mut self, _rhs: Monomial<N>) {
        self.coefficient *= _rhs.coefficient;
        self.deg += _rhs.deg;
    }
}

impl<N: Copy + Mul<Output = N>> ops::Mul<&Monomial<N>> for Monomial<N> {
    type Output = Monomial<N>;

    fn mul(self, _rhs: &Monomial<N>) -> Monomial<N> {
        Monomial::new(self.coefficient * _rhs.coefficient, self.deg + _rhs.deg)
    }
}

impl<N> ops::MulAssign<&Monomial<N>> for Monomial<N>
where
    N: MulAssign + AddAssign + Copy,
{
    fn mul_assign(&mut self, _rhs: &Monomial<N>) {
        self.coefficient *= _rhs.coefficient;
        self.deg += _rhs.deg;
    }
}

impl<N: Mul<Output = N>> ops::Mul<N> for Monomial<N> {
    type Output = Monomial<N>;

    fn mul(self, _rhs: N) -> Monomial<N> {
        Monomial::new(self.coefficient * _rhs, self.deg)
    }
}

impl<N: MulAssign> ops::MulAssign<N> for Monomial<N> {
    fn mul_assign(&mut self, _rhs: N) {
        self.coefficient *= _rhs;
    }
}

impl<N: Div<Output = N>> ops::Div<N> for Monomial<N> {
    type Output = Monomial<N>;

    fn div(self, _rhs: N) -> Monomial<N> {
        Monomial::new(self.coefficient / _rhs, self.deg)
    }
}

impl<N: DivAssign> ops::DivAssign<N> for Monomial<N> {
    fn div_assign(&mut self, _rhs: N) {
        self.coefficient /= _rhs;
    }
}

impl<N> ops::Shl<i32> for Monomial<N> {
    type Output = Monomial<N>;

    fn shl(self, _rhs: i32) -> Monomial<N> {
        if _rhs < 0 {
            self >> -_rhs
        } else {
            Monomial::new(self.coefficient, self.deg + (_rhs as usize))
        }
    }
}

impl<N> ops::ShlAssign<i32> for Monomial<N> {
    fn shl_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self >>= -_rhs;
        } else {
            self.deg += _rhs as usize;
        }
    }
}

impl<N> ops::Shr<i32> for Monomial<N> {
    type Output = Monomial<N>;

    fn shr(self, _rhs: i32) -> Monomial<N> {
        if _rhs < 0 {
            self << -_rhs
        } else {
            Monomial::new(self.coefficient, self.deg - (_rhs as usize))
        }
    }
}

impl<N> ops::ShrAssign<i32> for Monomial<N> {
    fn shr_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self <<= -_rhs;
        } else {
            self.deg -= _rhs as usize;
        }
    }
}

mod tests {
    use std::fmt::Write;
    use {Derivable, Evaluable, Integrable, Monomial, Polynomial};

    #[test]
    fn test_eval() {
        let a = Monomial::new(5, 2);
        assert_eq!(a.eval(5), 125);
    }

    #[test]
    fn test_derivative_of_zero() {
        let a: Monomial<i32> = Monomial::zero();
        assert_eq!(Monomial::zero(), a.derivative());
    }

    #[test]
    fn test_derivative_of_monomial_degree_zero() {
        let a = Monomial::new(5, 0);
        assert_eq!(Monomial::zero(), a.derivative());
    }

    #[test]
    fn test_derivative() {
        let a = Monomial::new(5, 3);
        assert_eq!(Monomial::new(15, 2), a.derivative());
    }

    #[test]
    fn test_integral() {
        let a = Monomial::new(5, 2);
        let integral = a.integral();
        assert_eq!(
            Polynomial::from_terms(vec![(5 / 3, 3)]),
            integral.polynomial
        );
    }

    #[test]
    fn test_str() {
        let a = Monomial::new(5, 2);
        assert_eq!(a.to_string(), "5x^2");
    }
}
