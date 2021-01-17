use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shr, ShrAssign, Sub, SubAssign,
};

use num::{Complex, One, Zero};

use crate::numerics::{Abs, AbsSqrt, IsNegativeOne, IsPositive, TryFromUsizeExact};
use crate::polynomial::find_roots::{discriminant_trinomial, trinomial_roots};
use crate::polynomial::polynomial::term_with_deg;
use crate::{
    Degree, Derivable, Evaluable, Integrable, Integral, LinearBinomial, MutablePolynomial,
    Polynomial, Roots, SizedPolynomial, Term, TryAddError,
};

#[derive(Debug, Clone)]
/// A type that stores terms of a quadratic trinomial in a static array. Operations are
/// much faster than on Polynomial for the same size polynomial, but terms can not
/// be added freely.
pub struct QuadraticTrinomial<N> {
    pub coefficients: [N; 3],
}

impl<N: Sized> QuadraticTrinomial<N> {
    /// Create a `QuadraticTrinomial` with the given coefficients.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, QuadraticTrinomial, Degree};
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

impl<N: Copy + Zero> SizedPolynomial<N> for QuadraticTrinomial<N> {
    /// Returns the term with the given `degree` of the `QuadraticTrinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, SizedPolynomial, Term};
    /// let trinomial = QuadraticTrinomial::new([1, 0, 3]);
    /// assert_eq!(Term::Term(1, 2), trinomial.term_with_degree(2));
    /// assert_eq!(Term::ZeroTerm, trinomial.term_with_degree(1));
    /// assert_eq!(Term::Term(3, 0), trinomial.term_with_degree(0));
    /// ```
    fn term_with_degree(&self, degree: usize) -> Term<N> {
        term_with_deg(&self.coefficients, degree)
    }

    /// Returns the degree of the `QuadraticTrinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, QuadraticTrinomial, Degree};
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

    /// Return a `QuadraticTrinomial` which is equal to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, SizedPolynomial};
    /// assert!(QuadraticTrinomial::<i32>::zero().is_zero());
    /// ```
    fn zero() -> Self {
        QuadraticTrinomial::new([N::zero(); 3])
    }

    /// Sets self to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, SizedPolynomial};
    /// let mut non_zero = QuadraticTrinomial::new([1, 1, 1]);
    /// assert!(!non_zero.is_zero());
    /// non_zero.set_to_zero();
    /// assert!(non_zero.is_zero());
    /// ```
    fn set_to_zero(&mut self) {
        self.coefficients = [N::zero(); 3];
    }
}

impl<N> MutablePolynomial<N> for QuadraticTrinomial<N>
where
    N: Zero + SubAssign + AddAssign + Copy,
{
    fn try_add_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        if degree <= 2 {
            self.coefficients[2 - degree] += coeff;
            Ok(())
        } else {
            Err(TryAddError::DegreeOutOfBounds)
        }
    }

    fn try_sub_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        if degree <= 2 {
            self.coefficients[2 - degree] -= coeff;
            Ok(())
        } else {
            Err(TryAddError::DegreeOutOfBounds)
        }
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
    N: Zero + One + Copy + Mul<Output = N> + TryFromUsizeExact,
{
    /// Returns the derivative of the `QuadraticTrinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, Derivable};
    /// let binomial = QuadraticTrinomial::new([3.0, 2.0, 1.0]);
    /// assert_eq!(QuadraticTrinomial::new([0.0, 6.0, 2.0]), binomial.derivative());
    /// ```
    fn derivative(&self) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            N::zero(),
            self.coefficients[0]
                * N::try_from_usize_exact(2).expect("Failed to convert 2usize to N."),
            self.coefficients[1],
        ])
    }
}

impl<N> Integrable<N, Polynomial<N>> for QuadraticTrinomial<N>
where
    N: Zero
        + TryFromUsizeExact
        + Copy
        + DivAssign
        + Mul<Output = N>
        + MulAssign
        + AddAssign
        + Div<Output = N>,
{
    /// Returns the integral of the `Monomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{QuadraticTrinomial, Integrable, Polynomial};
    /// let trinomial = QuadraticTrinomial::new([3.0, 0., 0.]);
    /// let integral = trinomial.integral();
    /// assert_eq!(&Polynomial::new(vec![1.0, 0.0, 0.0, 0.0]), integral.inner());
    /// ```
    ///
    /// # Errors
    /// Will panic if `N` can not losslessly represent `2usize` or `3usize`.
    fn integral(&self) -> Integral<N, Polynomial<N>> {
        Integral::new(Polynomial::new(vec![
            self.coefficients[0]
                / N::try_from_usize_exact(3).expect("Failed to convert 3usize to N."),
            self.coefficients[1]
                / N::try_from_usize_exact(2).expect("Failed to convert 2usize to N."),
            self.coefficients[2],
            N::zero(),
        ]))
    }
}

// impl<N: PowUsize + Copy> QuadraticTrinomial<N> {
//     /// Raises the `Monomial` to the power of exp.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use polynomial::Monomial;
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

    fn sub(self, rhs: QuadraticTrinomial<N>) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            self.coefficients[0] - rhs.coefficients[0],
            self.coefficients[1] - rhs.coefficients[1],
            self.coefficients[2] - rhs.coefficients[2],
        ])
    }
}

impl<N> SubAssign<QuadraticTrinomial<N>> for QuadraticTrinomial<N>
where
    N: SubAssign + Copy,
{
    fn sub_assign(&mut self, rhs: QuadraticTrinomial<N>) {
        self.coefficients[0] -= rhs.coefficients[0];
        self.coefficients[1] -= rhs.coefficients[1];
        self.coefficients[2] -= rhs.coefficients[2];
    }
}

impl<N> Add<QuadraticTrinomial<N>> for QuadraticTrinomial<N>
where
    N: Add<Output = N> + Copy,
{
    type Output = QuadraticTrinomial<N>;

    fn add(self, rhs: QuadraticTrinomial<N>) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            self.coefficients[0] + rhs.coefficients[0],
            self.coefficients[1] + rhs.coefficients[1],
            self.coefficients[2] + rhs.coefficients[2],
        ])
    }
}

impl<N: Copy + AddAssign> AddAssign<QuadraticTrinomial<N>> for QuadraticTrinomial<N> {
    fn add_assign(&mut self, rhs: QuadraticTrinomial<N>) {
        self.coefficients[0] += rhs.coefficients[0];
        self.coefficients[1] += rhs.coefficients[1];
        self.coefficients[2] += rhs.coefficients[2];
    }
}

// impl<N: Copy + Mul<Output = N>> Mul<Monomial<N>> for Monomial<N> {
//     type Output = Monomial<N>;
//
//     fn mul(self, rhs: Monomial<N>) -> Monomial<N> {
//         Monomial::new(self.coefficient * rhs.coefficient, self.deg + rhs.deg)
//     }
// }
//
// impl<N: MulAssign + AddAssign> MulAssign<Monomial<N>> for Monomial<N> {
//     fn mul_assign(&mut self, rhs: Monomial<N>) {
//         self.coefficient *= rhs.coefficient;
//         self.deg += rhs.deg;
//     }
// }
//
// impl<N: Copy + Mul<Output = N>> Mul<&Monomial<N>> for Monomial<N> {
//     type Output = Monomial<N>;
//
//     fn mul(self, rhs: &Monomial<N>) -> Monomial<N> {
//         Monomial::new(self.coefficient * rhs.coefficient, self.deg + rhs.deg)
//     }
// }
//
// impl<N> MulAssign<&Monomial<N>> for Monomial<N>
// where
//     N: MulAssign + AddAssign + Copy,
// {
//     fn mul_assign(&mut self, rhs: &Monomial<N>) {
//         self.coefficient *= rhs.coefficient;
//         self.deg += rhs.deg;
//     }
// }
//
impl<N: Mul<Output = N> + Copy> Mul<N> for QuadraticTrinomial<N> {
    type Output = QuadraticTrinomial<N>;

    fn mul(self, rhs: N) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            self.coefficients[0] * rhs,
            self.coefficients[1] * rhs,
            self.coefficients[2] * rhs,
        ])
    }
}

impl<N: MulAssign + Copy> MulAssign<N> for QuadraticTrinomial<N> {
    fn mul_assign(&mut self, rhs: N) {
        self.coefficients[0] *= rhs;
        self.coefficients[1] *= rhs;
        self.coefficients[2] *= rhs;
    }
}

impl<N: Div<Output = N> + Copy> Div<N> for QuadraticTrinomial<N> {
    type Output = QuadraticTrinomial<N>;

    fn div(self, rhs: N) -> QuadraticTrinomial<N> {
        QuadraticTrinomial::new([
            self.coefficients[0] / rhs,
            self.coefficients[1] / rhs,
            self.coefficients[2] / rhs,
        ])
    }
}

impl<N: DivAssign + Copy> DivAssign<N> for QuadraticTrinomial<N> {
    fn div_assign(&mut self, rhs: N) {
        self.coefficients[0] /= rhs;
        self.coefficients[1] /= rhs;
        self.coefficients[2] /= rhs;
    }
}

// impl<N: Zero + Copy> Shl<i32> for QuadraticTrinomial<N> {
//     type Output = QuadraticTrinomial<N>;
//
//     fn shl(self, rhs: i32) -> QuadraticTrinomial<N> {
//         if rhs < 0 {
//             self >> -rhs
//         } else {
//             match rhs {
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
//     fn shl_assign(&mut self, rhs: i32) {
//         if rhs < 0 {
//             *self >>= -rhs;
//         } else {
//             match rhs {
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

    fn shr(self, rhs: u32) -> QuadraticTrinomial<N> {
        match rhs {
            0 => QuadraticTrinomial::new(self.coefficients),
            1 => QuadraticTrinomial::new([N::zero(), self.coefficients[0], self.coefficients[1]]),
            2 => QuadraticTrinomial::new([N::zero(), N::zero(), self.coefficients[0]]),
            _ => QuadraticTrinomial::zero(),
        }
    }
}

impl<N: Zero + Copy> ShrAssign<u32> for QuadraticTrinomial<N> {
    fn shr_assign(&mut self, rhs: u32) {
        match rhs {
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
mod test {
    use crate::{Derivable, Evaluable, QuadraticTrinomial, Roots, SizedPolynomial};
    use num::Complex;

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
