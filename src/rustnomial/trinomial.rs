use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shr, ShrAssign, Sub, SubAssign,
};

use num::{Complex, One, Zero};

use rustnomial::binomial::LinearBinomial;
use rustnomial::err::TryAddError;
use rustnomial::find_roots::{discriminant_trinomial, trinomial_roots};
use rustnomial::numerics::{Abs, AbsSqrt, IsNegativeOne, IsPositive};
use rustnomial::traits::{MutablePolynomial, TermIterator};
use {Degree, Derivable, Evaluable, GenericPolynomial, Roots, Term};

#[derive(Debug, Clone)]
pub struct QuadraticTrinomial<N> {
    pub coefficients: [N; 3],
}

impl<N: Sized> QuadraticTrinomial<N> {
    /// Create a `QuadraticTrinomial` with the given coefficients.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{GenericPolynomial, QuadraticTrinomial, Degree};
    /// let trinomial = QuadraticTrinomial::new([3.0, 1.0, 0.5]);
    /// assert_eq!([3.0, 1.0, 0.5], trinomial.coefficients);
    /// assert_eq!(Degree::Num(2), trinomial.degree());
    /// ```
    pub fn new(coefficients: [N; 3]) -> QuadraticTrinomial<N> {
        QuadraticTrinomial { coefficients }
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
        discriminant_trinomial(a, b, c)
    }

    /// Return the roots of `QuadraticTrinomial` with largest
    /// first, smallest second.
    pub fn roots(&self) -> Roots<N> {
        let [a, b, c] = self.coefficients;
        trinomial_roots(a, b, c)
    }

    pub fn complex_factors(&self) -> (N, LinearBinomial<Complex<N>>, LinearBinomial<Complex<N>>) {
        match self.roots() {
            Roots::TwoComplexRoots(root_a, root_b) => (
                self.coefficients[0],
                LinearBinomial::new([Complex::new(N::one(), -N::zero()), root_a]),
                LinearBinomial::new([Complex::new(N::one(), -N::zero()), root_b]),
            ),
            Roots::TwoRealRoots(a, b) => (
                self.coefficients[0],
                LinearBinomial::new([
                    Complex::new(N::one(), N::zero()),
                    Complex::new(-a, N::zero()),
                ]),
                LinearBinomial::new([
                    Complex::new(N::one(), N::zero()),
                    Complex::new(-b, N::zero()),
                ]),
            ),
            _ => unreachable!(),
        }
    }

    pub fn real_factors(&self) -> Option<(N, LinearBinomial<N>, LinearBinomial<N>)> {
        if let Roots::TwoRealRoots(root_a, root_b) = self.roots() {
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
    /// Return a `QuadraticTrinomial` which is equal to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, GenericPolynomial};
    /// assert!(QuadraticTrinomial::<i32>::zero().is_zero());
    /// ```
    fn zero() -> Self {
        QuadraticTrinomial::new([N::zero(); 3])
    }

    /// Return the number of terms in `QuadraticTrinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, GenericPolynomial};
    /// let trinomial = QuadraticTrinomial::new([1, 2, 3]);
    /// assert_eq!(3, trinomial.len());
    /// assert_eq!(0, QuadraticTrinomial::<i32>::zero().len());
    /// ```
    fn len(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            3
        }
    }

    /// Returns the nth term of the `QuadraticTrinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, GenericPolynomial, Term};
    /// let trinomial = QuadraticTrinomial::new([1, 0, 3]);
    /// assert_eq!(Term::Term(1, 2), trinomial.nth_term(0));
    /// assert_eq!(Term::ZeroTerm, trinomial.nth_term(1));
    /// assert_eq!(Term::Term(3, 0), trinomial.nth_term(2));
    /// ```
    fn nth_term(&self, index: usize) -> Term<N> {
        Term::new(self.coefficients[index], 2 - index)
    }

    /// Returns an iterator for the `QuadraticTrinomial`, yielding the term constant and degree.
    /// Terms are iterated over in descending degree order, excluding zero terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, GenericPolynomial};
    /// let trinomial = QuadraticTrinomial::new([1, 0, 3]);
    /// let mut iter = trinomial.term_iter();
    /// assert_eq!(Some((1, 2)), iter.next());
    /// assert_eq!(Some((3, 0)), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn term_iter(&self) -> TermIterator<N> {
        TermIterator::new(self)
    }

    /// Returns the degree of the `QuadraticTrinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{GenericPolynomial, QuadraticTrinomial, Degree};
    /// let trinomial = QuadraticTrinomial::new([1, 2, 3]);
    /// assert_eq!(Degree::Num(2), trinomial.degree());
    /// let binomial = QuadraticTrinomial::new([0, 2, 3]);
    /// assert_eq!(Degree::Num(1), binomial.degree());
    /// let monomial = QuadraticTrinomial::new([0, 0, 3]);
    /// assert_eq!(Degree::Num(0), monomial.degree());
    /// let zero = QuadraticTrinomial::new([0, 0, 0]);
    /// assert_eq!(Degree::NegInf, zero.degree());
    /// ```
    fn degree(&self) -> Degree {
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
    /// use rustnomial::{GenericPolynomial, QuadraticTrinomial, Degree};
    /// let trinomial = QuadraticTrinomial::new([1, 2, 3]);
    /// assert!(!trinomial.is_zero());
    /// let zero = QuadraticTrinomial::new([0, 0, 0]);
    /// assert!(zero.is_zero());
    /// let zero = QuadraticTrinomial::<i32>::zero();
    /// assert!(zero.is_zero());
    /// ```
    fn is_zero(&self) -> bool {
        self.degree() == Degree::NegInf
    }
}

impl<N> MutablePolynomial<N> for QuadraticTrinomial<N>
where
    N: Zero + AddAssign + Copy,
{
    fn try_add_term(&mut self, term: N, degree: usize) -> Result<(), TryAddError> {
        if degree <= 2 {
            self.coefficients[2 - degree] += term;
            Ok(())
        } else {
            Err(TryAddError::DegreeOutOfBounds)
        }
    }

    fn set_to_zero(&mut self) {
        self.coefficients = [N::zero(); 3];
    }
}

impl<N> Evaluable<N> for QuadraticTrinomial<N>
where
    N: Add<Output = N> + Mul<Output = N> + Copy,
{
    /// Returns the value of the `QuadraticTrinomial` at the given point.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, Evaluable};
    /// let trinomial = QuadraticTrinomial::new([1, 2, 3]);
    /// assert_eq!(6, trinomial.eval(1));
    /// assert_eq!(3, trinomial.eval(0));
    /// ```
    fn eval(&self, point: N) -> N {
        point * (self.coefficients[0] * point + self.coefficients[1]) + self.coefficients[2]
    }
}

impl<N> Derivable<N> for QuadraticTrinomial<N>
where
    N: Zero + One + Copy + Mul<Output = N> + From<u8>,
{
    /// Returns the derivative of the `QuadraticTrinomial`.
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
    /// Returns true if this `QuadraticTrinomial` is equal to other.
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }
}

macro_rules! from_trinomial_a_to_b {
    ($A:ty, $B:ty) => {
        impl From<QuadraticTrinomial<$A>> for QuadraticTrinomial<$B> {
            fn from(item: QuadraticTrinomial<$A>) -> Self {
                QuadraticTrinomial::new([
                    item.coefficients[0] as $B,
                    item.coefficients[1] as $B,
                    item.coefficients[2] as $B,
                ])
            }
        }
    };
}

upcast!(from_trinomial_a_to_b);
poly_from_str!(QuadraticTrinomial);
fmt_poly!(QuadraticTrinomial);

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
    use {GenericPolynomial, Roots};

    #[test]
    fn test_eval() {
        let a = QuadraticTrinomial::new([5, 0, 0]);
        assert_eq!(125, a.eval(5));
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
        assert_eq!(c, a >> 2);
    }

    #[test]
    fn test_shr_assign_pos() {
        let mut a = QuadraticTrinomial::new([1, 0, 0]);
        let c = QuadraticTrinomial::new([0, 0, 1]);
        a >>= 2;
        assert_eq!(c, a);
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
        assert_eq!(QuadraticTrinomial::zero(), a >> 5);
    }

    #[test]
    fn test_shr_assign_to_zero() {
        let mut a = QuadraticTrinomial::new([1, 2, 3]);
        a >>= 5;
        assert_eq!(QuadraticTrinomial::zero(), a);
    }

    #[test]
    fn test_derivative_of_zero() {
        let a: QuadraticTrinomial<i32> = QuadraticTrinomial::zero();
        assert_eq!(QuadraticTrinomial::zero(), a.derivative());
    }

    #[test]
    fn test_derivative_of_degree_zero() {
        let a = QuadraticTrinomial::new([0, 0, 1]);
        assert_eq!(QuadraticTrinomial::zero(), a.derivative());
    }

    #[test]
    fn test_derivative() {
        let a = QuadraticTrinomial::new([1, 2, 3]);
        assert_eq!(QuadraticTrinomial::new([0, 2, 2]), a.derivative());
    }

    #[test]
    fn test_roots_pos() {
        let a = QuadraticTrinomial::new([1, 4, 4]);
        let c = Roots::TwoRealRoots(-2i16, -2i16);
        assert_eq!(c, a.roots());
    }

    #[test]
    fn test_complex_roots_neg() {
        let a = QuadraticTrinomial::new([1, 0, 4]);
        let c = Roots::TwoComplexRoots(Complex::new(0, 2i16), Complex::new(0, -2i16));
        assert_eq!(c, a.roots());
    }
}
