use core::fmt;
use core::fmt::Display;
use core::ops::{
    AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign, SubAssign,
};

use num::{One, Zero};

use crate::numerics::{IsNegativeOne, PowUsize, TryFromUsizeExact};
use crate::strings::write_leading_term;
use crate::{
    Degree, Derivable, Evaluable, FreeSizePolynomial, Integrable, Integral, MutablePolynomial,
    Roots, SizedPolynomial, SparsePolynomial, Term, TryAddError,
};

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
    /// use rustnomial::{Monomial, Degree, SizedPolynomial};
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
    fn as_term(&self) -> Term<N> {
        Term::new(self.coefficient, self.deg)
    }
}

impl<N: Copy + Zero> SizedPolynomial<N> for Monomial<N> {
    /// Return the number of terms in `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, SizedPolynomial};
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

    /// Returns the term with the given `degree` of the `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, SizedPolynomial, Term};
    /// let monomial = Monomial::new(5, 2);
    /// assert_eq!(Term::Term(5, 2), monomial.term_with_degree(2));
    /// assert_eq!(Term::ZeroTerm, monomial.term_with_degree(1));
    /// ```
    fn term_with_degree(&self, degree: usize) -> Term<N> {
        if degree != self.deg {
            Term::ZeroTerm
        } else {
            Term::new(self.coefficient, self.deg)
        }
    }

    /// Returns the degree of the `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, Monomial, Degree};
    /// let monomial = Monomial::new(3.0, 2);
    /// assert_eq!(Degree::Num(2), monomial.degree());
    /// let zero_with_nonzero_deg = Monomial::new(0.0, 2);
    /// assert_eq!(Degree::NegInf, zero_with_nonzero_deg.degree());
    /// let nonzero_with_zero_degree = Monomial::new(1.0, 0);
    /// assert_eq!(Degree::Num(0), nonzero_with_zero_degree.degree());
    /// ```
    fn degree(&self) -> Degree {
        if self.coefficient.is_zero() {
            Degree::NegInf
        } else {
            Degree::Num(self.deg)
        }
    }

    /// Return a `Monomial` which is equal to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, Monomial};
    /// assert!(Monomial::<i32>::zero().is_zero());
    /// ```
    fn zero() -> Self {
        Monomial::new(N::zero(), 0)
    }

    /// Sets self to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, Monomial};
    /// let mut non_zero = Monomial::new(1, 1);
    /// assert!(!non_zero.is_zero());
    /// non_zero.set_to_zero();
    /// assert!(non_zero.is_zero());
    /// ```
    fn set_to_zero(&mut self) {
        self.coefficient = N::zero();
    }
}

impl<N> MutablePolynomial<N> for Monomial<N>
where
    N: SubAssign + AddAssign + Copy + Zero,
{
    fn try_add_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        if self.is_zero() {
            self.coefficient += coeff;
            self.deg = degree;
            Ok(())
        } else if degree != self.deg {
            Err(TryAddError::TooManyTerms)
        } else {
            self.coefficient += coeff;
            Ok(())
        }
    }

    fn try_sub_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        if self.is_zero() {
            self.coefficient -= coeff;
            self.deg = degree;
            Ok(())
        } else if degree != self.deg {
            Err(TryAddError::TooManyTerms)
        } else {
            self.coefficient -= coeff;
            Ok(())
        }
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

impl<N: Copy + Zero> Monomial<N> {
    /// Return the root of `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, Roots, SizedPolynomial};
    /// let monomial = Monomial::new(1, 2);
    /// assert_eq!(Roots::OneRealRoot(0), monomial.root());
    /// let zero = Monomial::<i32>::zero();
    /// assert_eq!(Roots::InfiniteRoots, zero.root());
    /// let constant = Monomial::new(1, 0);
    /// assert_eq!(Roots::NoRoots, constant.root());
    /// ```
    pub fn root(&self) -> Roots<N> {
        match self.degree() {
            Degree::NegInf => Roots::InfiniteRoots,
            Degree::Num(0) => Roots::NoRoots,
            Degree::Num(_) => Roots::OneRealRoot(N::zero()),
        }
    }
}

impl<N> Derivable<N> for Monomial<N>
where
    N: Zero + Copy + Mul<Output = N> + TryFromUsizeExact,
{
    /// Returns the derivative of the `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, Derivable};
    /// let monomial = Monomial::new(3.0, 2);
    /// assert_eq!(Monomial::new(6.0, 1), monomial.derivative());
    /// ```
    ///
    /// # Errors
    /// Will panic if `N` can not losslessly represent the degree of `self`.
    fn derivative(&self) -> Monomial<N> {
        match self.degree() {
            Degree::NegInf | Degree::Num(0) => Monomial::zero(),
            Degree::Num(x) => Monomial::new(
                self.coefficient
                    * N::try_from_usize_exact(x)
                        .expect("Degree has no lossless representation in N."),
                x - 1,
            ),
        }
    }
}

impl<N> Integrable<N, SparsePolynomial<N>> for Monomial<N>
where
    N: Zero
        + Copy
        + Mul<Output = N>
        + AddAssign
        + PowUsize
        + Mul<Output = N>
        + Div<Output = N>
        + TryFromUsizeExact,
{
    /// Returns the integral of the `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, SparsePolynomial, Integrable, FreeSizePolynomial};
    /// let monomial = Monomial::new(3.0, 2);
    /// let integral = monomial.integral();
    /// assert_eq!(&SparsePolynomial::from_terms(&[(1.0, 3)]), integral.inner());
    /// assert_eq!(1., integral.eval(0., 1.));
    /// ```
    fn integral(&self) -> Integral<N, SparsePolynomial<N>> {
        match self.degree() {
            Degree::NegInf => Integral::new(SparsePolynomial::zero()),
            Degree::Num(x) => Integral::new(SparsePolynomial::from_terms(&[(
                self.coefficient / N::try_from_usize_exact(x + 1).unwrap(),
                x + 1,
            )])),
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
        self.as_term() == other.as_term()
    }
}

impl<N> From<N> for Monomial<N> {
    fn from(item: N) -> Self {
        Monomial::new(item, 0)
    }
}

macro_rules! from_monomial_a_to_b {
    ($A:ty, $B:ty) => {
        impl From<Monomial<$A>> for Monomial<$B> {
            fn from(item: Monomial<$A>) -> Self {
                Monomial::new(item.coefficient as $B, item.deg)
            }
        }
    };
}

upcast!(from_monomial_a_to_b);
poly_from_str!(Monomial);

impl<N> fmt::Display for Monomial<N>
where
    N: Zero + One + PartialEq + Copy + IsNegativeOne + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some((coeff, degree)) = self.term_iter().next() {
            write_leading_term(f, coeff, degree)
        } else {
            write!(f, "0")
        }
    }
}

impl<N: Copy + Neg<Output = N>> Neg for Monomial<N> {
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

impl<N: Copy + Mul<Output = N>> Mul<Monomial<N>> for Monomial<N> {
    type Output = Monomial<N>;

    fn mul(self, _rhs: Monomial<N>) -> Monomial<N> {
        Monomial::new(self.coefficient * _rhs.coefficient, self.deg + _rhs.deg)
    }
}

impl<N: MulAssign> MulAssign<Monomial<N>> for Monomial<N> {
    fn mul_assign(&mut self, _rhs: Monomial<N>) {
        self.coefficient *= _rhs.coefficient;
        self.deg += _rhs.deg;
    }
}

impl<N: Copy + Mul<Output = N>> Mul<&Monomial<N>> for Monomial<N> {
    type Output = Monomial<N>;

    fn mul(self, _rhs: &Monomial<N>) -> Monomial<N> {
        Monomial::new(self.coefficient * _rhs.coefficient, self.deg + _rhs.deg)
    }
}

impl<N> MulAssign<&Monomial<N>> for Monomial<N>
where
    N: MulAssign + AddAssign + Copy,
{
    fn mul_assign(&mut self, _rhs: &Monomial<N>) {
        self.coefficient *= _rhs.coefficient;
        self.deg += _rhs.deg;
    }
}

impl<N: Mul<Output = N>> Mul<N> for Monomial<N> {
    type Output = Monomial<N>;

    fn mul(self, _rhs: N) -> Monomial<N> {
        Monomial::new(self.coefficient * _rhs, self.deg)
    }
}

impl<N: MulAssign> MulAssign<N> for Monomial<N> {
    fn mul_assign(&mut self, _rhs: N) {
        self.coefficient *= _rhs;
    }
}

impl<N: Div<Output = N>> Div<N> for Monomial<N> {
    type Output = Monomial<N>;

    fn div(self, _rhs: N) -> Monomial<N> {
        Monomial::new(self.coefficient / _rhs, self.deg)
    }
}

impl<N: DivAssign> DivAssign<N> for Monomial<N> {
    fn div_assign(&mut self, _rhs: N) {
        self.coefficient /= _rhs;
    }
}

impl<N: Zero + Copy> Shl<i32> for Monomial<N> {
    type Output = Monomial<N>;

    fn shl(self, _rhs: i32) -> Monomial<N> {
        if _rhs < 0 {
            self >> -_rhs
        } else {
            Monomial::new(self.coefficient, self.deg + (_rhs as usize))
        }
    }
}

impl<N: Zero + Copy> ShlAssign<i32> for Monomial<N> {
    fn shl_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self >>= -_rhs;
        } else {
            self.deg += _rhs as usize;
        }
    }
}

impl<N: Zero + Copy> Shr<i32> for Monomial<N> {
    type Output = Monomial<N>;

    fn shr(self, _rhs: i32) -> Monomial<N> {
        if _rhs < 0 {
            self << -_rhs
        } else {
            let _rhs = _rhs as usize;
            if _rhs > self.deg {
                Monomial::zero()
            } else {
                Monomial::new(self.coefficient, self.deg - _rhs)
            }
        }
    }
}

impl<N: Zero + Copy> ShrAssign<i32> for Monomial<N> {
    fn shr_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self <<= -_rhs;
        } else {
            let _rhs = _rhs as usize;
            if _rhs > self.deg {
                self.coefficient = N::zero();
                self.deg = 0;
            } else {
                self.deg -= _rhs;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        Derivable, Evaluable, FreeSizePolynomial, Integrable, Monomial, Roots, SizedPolynomial,
        SparsePolynomial,
    };

    #[test]
    fn test_root_zero() {
        let a = Monomial::<i32>::zero();
        assert_eq!(Roots::InfiniteRoots, a.root());
    }

    #[test]
    fn test_root_constant() {
        let a = Monomial::new(1, 0);
        assert_eq!(Roots::NoRoots, a.root());
    }

    #[test]
    fn test_root_not_constant() {
        let a = Monomial::new(1, 1);
        assert_eq!(Roots::OneRealRoot(0), a.root());
    }

    #[test]
    fn test_eval() {
        let a = Monomial::new(5, 2);
        assert_eq!(125, a.eval(5));
    }

    #[test]
    fn test_shl_pos() {
        let a = Monomial::new(1, 2);
        let c = Monomial::new(1, 7);
        assert_eq!(c, a << 5);
    }

    #[test]
    fn test_shl_assign_pos() {
        let mut a = Monomial::new(1, 2);
        let c = Monomial::new(1, 7);
        a <<= 5;
        assert_eq!(c, a);
    }

    #[test]
    fn test_shl_neg() {
        let a = Monomial::new(1, 7);
        let c = Monomial::new(1, 2);
        assert_eq!(c, a << -5);
    }

    #[test]
    fn test_shl_assign_neg() {
        let mut a = Monomial::new(1, 7);
        let c = Monomial::new(1, 2);
        a <<= -5;
        assert_eq!(c, a);
    }

    #[test]
    fn test_shr_pos() {
        let a = Monomial::new(1, 7);
        let c = Monomial::new(1, 2);
        assert_eq!(c, a >> 5);
    }

    #[test]
    fn test_shr_assign_pos() {
        let mut a = Monomial::new(1, 7);
        let c = Monomial::new(1, 2);
        a >>= 5;
        assert_eq!(c, a);
    }

    #[test]
    fn test_shr_neg() {
        let a = Monomial::new(1, 2);
        let c = Monomial::new(1, 7);
        assert_eq!(c, a >> -5);
    }

    #[test]
    fn test_shr_assign_neg() {
        let mut a = Monomial::new(1, 2);
        let c = Monomial::new(1, 7);
        a >>= -5;
        assert_eq!(c, a);
    }

    #[test]
    fn test_shr_to_zero() {
        let a = Monomial::new(5, 1);
        assert_eq!(Monomial::zero(), a >> 5);
    }

    #[test]
    fn test_shr_assign_to_zero() {
        let mut a = Monomial::new(5, 1);
        a >>= 5;
        assert_eq!(Monomial::zero(), a);
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
            &SparsePolynomial::from_terms(&[(5 / 3, 3)]),
            integral.inner()
        );
    }
}
