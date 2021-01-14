use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shr, ShrAssign, Sub, SubAssign,
};

use num::{One, Zero};

use rustnomial::numerics::{Abs, IsNegativeOne, IsPositive};
use rustnomial::polynomial::term_with_deg;
use {Degree, Derivable, Evaluable, MutablePolynomial, Roots, SizedPolynomial, Term, TryAddError};

#[derive(Debug, Clone)]
pub struct LinearBinomial<N> {
    pub coefficients: [N; 2],
}

impl<N: Sized> LinearBinomial<N> {
    /// Create a `LinearBinomial` with the given terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, LinearBinomial, Degree};
    /// let binomial = LinearBinomial::new([3, 2]);
    /// assert_eq!(Degree::Num(1), binomial.degree());
    /// ```
    pub fn new(coefficients: [N; 2]) -> LinearBinomial<N> {
        LinearBinomial { coefficients }
    }
}

impl<N> LinearBinomial<N>
where
    N: Copy + Neg<Output = N> + Div<Output = N> + Zero,
{
    /// Return the root of `LinearBinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{LinearBinomial, Roots, SizedPolynomial};
    /// let binomial = LinearBinomial::new([1.0, 2.0]);
    /// assert_eq!(Roots::OneRealRoot(-2.0), binomial.root());
    /// let zero = LinearBinomial::<i32>::zero();
    /// assert_eq!(Roots::InfiniteRoots, zero.root());
    /// let constant = LinearBinomial::new([0, 1]);
    /// assert_eq!(Roots::NoRoots, constant.root());
    /// ```
    pub fn root(&self) -> Roots<N> {
        let [a, b] = self.coefficients;
        if a.is_zero() {
            if b.is_zero() {
                Roots::InfiniteRoots
            } else {
                Roots::NoRoots
            }
        } else {
            Roots::OneRealRoot(-b / a)
        }
    }
}

impl<N: Copy + Zero> SizedPolynomial<N> for LinearBinomial<N> {
    /// Return the length of `LinearBinomial`. Not equal to the number of terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{LinearBinomial, SizedPolynomial};
    /// let binomial = LinearBinomial::new([3.0, 2.0]);
    /// assert_eq!(2, binomial.len());
    /// assert_eq!(0, LinearBinomial::<i32>::zero().len());
    /// ```
    fn len(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            2
        }
    }

    /// Returns the term with the given `degree` of the `LinearBinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{LinearBinomial, SizedPolynomial, Term};
    /// let binomial = LinearBinomial::new([5, 0]);
    /// assert_eq!(Term::Term(5, 1), binomial.term_with_degree(1));
    /// assert_eq!(Term::ZeroTerm, binomial.term_with_degree(0));
    /// ```
    fn term_with_degree(&self, degree: usize) -> Term<N> {
        term_with_deg(&self.coefficients, degree)
    }

    /// Returns the degree of the `LinearBinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, LinearBinomial, Degree};
    /// let binomial = LinearBinomial::new([3.0, 2.0]);
    /// assert_eq!(Degree::Num(1), binomial.degree());
    /// let monomial = LinearBinomial::new([0.0, 1.0]);
    /// assert_eq!(Degree::Num(0), monomial.degree());
    /// let zero = LinearBinomial::<i32>::zero();
    /// assert_eq!(Degree::NegInf, zero.degree());
    /// ```
    fn degree(&self) -> Degree {
        if !self.coefficients[0].is_zero() {
            Degree::Num(1)
        } else if !self.coefficients[1].is_zero() {
            Degree::Num(0)
        } else {
            Degree::NegInf
        }
    }

    /// Returns a `LinearBinomial` with no terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, LinearBinomial};
    /// let zero = LinearBinomial::<i32>::zero();
    /// assert!(zero.is_zero());
    /// assert!(zero.term_iter().next().is_none());
    /// ```
    fn zero() -> Self {
        LinearBinomial::new([N::zero(); 2])
    }

    /// Sets self to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, LinearBinomial};
    /// let mut non_zero = LinearBinomial::new([1, 1]);
    /// assert!(!non_zero.is_zero());
    /// non_zero.set_to_zero();
    /// assert!(non_zero.is_zero());
    /// ```
    fn set_to_zero(&mut self) {
        self.coefficients = [N::zero(); 2];
    }
}

impl<N> MutablePolynomial<N> for LinearBinomial<N>
where
    N: Zero + AddAssign + Copy,
{
    fn try_add_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        if degree <= 1 {
            self.coefficients[1 - degree] += coeff;
            Ok(())
        } else {
            Err(TryAddError::DegreeOutOfBounds)
        }
    }
}

impl<N> Evaluable<N> for LinearBinomial<N>
where
    N: Add<Output = N> + Mul<Output = N> + Copy,
{
    /// Returns the value of the `LinearBinomial` at the given point.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{LinearBinomial, Evaluable};
    /// let binomial = LinearBinomial::new([1, 2]);
    /// assert_eq!(7, binomial.eval(5));
    /// assert_eq!(2, binomial.eval(0));
    /// ```
    fn eval(&self, point: N) -> N {
        point * self.coefficients[0] + self.coefficients[1]
    }
}

impl<N> Derivable<N> for LinearBinomial<N>
where
    N: Zero + One + Copy + Mul<Output = N> + From<u8>,
{
    /// Returns the derivative of the `LinearBinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{LinearBinomial, Derivable};
    /// let binomial = LinearBinomial::new([3.0, 1.0]);
    /// assert_eq!(LinearBinomial::new([0., 3.0]), binomial.derivative());
    /// ```
    fn derivative(&self) -> LinearBinomial<N> {
        LinearBinomial::new([N::zero(), self.coefficients[0]])
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
impl<N> PartialEq for LinearBinomial<N>
where
    N: Zero + PartialEq + Copy,
{
    /// Returns true if this `LinearBinomial` and other are equal.
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }
}

macro_rules! from_binomial_a_to_b {
    ($A:ty, $B:ty) => {
        impl From<LinearBinomial<$A>> for LinearBinomial<$B> {
            fn from(item: LinearBinomial<$A>) -> Self {
                LinearBinomial::new([item.coefficients[0] as $B, item.coefficients[1] as $B])
            }
        }
    };
}

upcast!(from_binomial_a_to_b);
poly_from_str!(LinearBinomial);
fmt_poly!(LinearBinomial);

impl<N: Copy + Neg<Output = N>> Neg for LinearBinomial<N> {
    type Output = LinearBinomial<N>;

    fn neg(self) -> LinearBinomial<N> {
        LinearBinomial::new([-self.coefficients[0], -self.coefficients[1]])
    }
}

impl<N> Sub<LinearBinomial<N>> for LinearBinomial<N>
where
    N: Copy + Sub<Output = N>,
{
    type Output = LinearBinomial<N>;

    fn sub(self, _rhs: LinearBinomial<N>) -> LinearBinomial<N> {
        LinearBinomial::new([
            self.coefficients[0] - _rhs.coefficients[0],
            self.coefficients[1] - _rhs.coefficients[1],
        ])
    }
}

impl<N> SubAssign<LinearBinomial<N>> for LinearBinomial<N>
where
    N: SubAssign + Copy,
{
    fn sub_assign(&mut self, _rhs: LinearBinomial<N>) {
        self.coefficients[0] -= _rhs.coefficients[0];
        self.coefficients[1] -= _rhs.coefficients[1];
    }
}

impl<N> Add<LinearBinomial<N>> for LinearBinomial<N>
where
    N: Add<Output = N> + Copy,
{
    type Output = LinearBinomial<N>;

    fn add(self, _rhs: LinearBinomial<N>) -> LinearBinomial<N> {
        LinearBinomial::new([
            self.coefficients[0] + _rhs.coefficients[0],
            self.coefficients[1] + _rhs.coefficients[1],
        ])
    }
}

impl<N: Copy + AddAssign> AddAssign<LinearBinomial<N>> for LinearBinomial<N> {
    fn add_assign(&mut self, _rhs: LinearBinomial<N>) {
        self.coefficients[0] += _rhs.coefficients[0];
        self.coefficients[1] += _rhs.coefficients[1];
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
impl<N: Mul<Output = N> + Copy> Mul<N> for LinearBinomial<N> {
    type Output = LinearBinomial<N>;

    fn mul(self, _rhs: N) -> LinearBinomial<N> {
        LinearBinomial::new([self.coefficients[0] * _rhs, self.coefficients[1] * _rhs])
    }
}

impl<N: MulAssign + Copy> MulAssign<N> for LinearBinomial<N> {
    fn mul_assign(&mut self, _rhs: N) {
        self.coefficients[0] *= _rhs;
        self.coefficients[1] *= _rhs;
    }
}

impl<N: Div<Output = N> + Copy> Div<N> for LinearBinomial<N> {
    type Output = LinearBinomial<N>;

    fn div(self, _rhs: N) -> LinearBinomial<N> {
        LinearBinomial::new([self.coefficients[0] / _rhs, self.coefficients[1] / _rhs])
    }
}

impl<N: DivAssign + Copy> DivAssign<N> for LinearBinomial<N> {
    fn div_assign(&mut self, _rhs: N) {
        self.coefficients[0] /= _rhs;
        self.coefficients[1] /= _rhs;
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

impl<N: Zero + Copy> Shr<u32> for LinearBinomial<N> {
    type Output = LinearBinomial<N>;

    fn shr(self, _rhs: u32) -> LinearBinomial<N> {
        match _rhs {
            0 => LinearBinomial::new(self.coefficients),
            1 => LinearBinomial::new([N::zero(), self.coefficients[0]]),
            _ => LinearBinomial::zero(),
        }
    }
}

impl<N: Zero + Copy> ShrAssign<u32> for LinearBinomial<N> {
    fn shr_assign(&mut self, _rhs: u32) {
        match _rhs {
            0 => {}
            1 => {
                self.coefficients[1] = self.coefficients[0];
                self.coefficients[0] = N::zero();
            }
            _ => {
                self.coefficients[0] = N::zero();
                self.coefficients[1] = N::zero();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use {Degree, Derivable, Evaluable, LinearBinomial, Roots, SizedPolynomial};

    #[test]
    fn test_root_both_zero() {
        let a = LinearBinomial::new([0, 0]);
        assert_eq!(Roots::InfiniteRoots, a.root());
    }

    #[test]
    fn test_root_constant() {
        let a = LinearBinomial::new([0, 1]);
        assert_eq!(Roots::NoRoots, a.root());
    }

    #[test]
    fn test_root_both() {
        let a = LinearBinomial::new([1, 2]);
        assert_eq!(Roots::OneRealRoot(-2), a.root());
    }

    #[test]
    fn test_degree_both_zero() {
        let a = LinearBinomial::<i32>::zero();
        assert_eq!(Degree::NegInf, a.degree());
    }

    #[test]
    fn test_degree_second_non_zero() {
        let a = LinearBinomial::new([0, 1u8]);
        assert_eq!(Degree::Num(0), a.degree());
    }

    #[test]
    fn test_degree_first_non_zero() {
        let a = LinearBinomial::new([1u8, 0]);
        assert_eq!(Degree::Num(1), a.degree());
    }

    #[test]
    fn test_degree_both_non_zero() {
        let a = LinearBinomial::new([2u8, 1u8]);
        assert_eq!(Degree::Num(1), a.degree());
    }

    #[test]
    fn test_eval() {
        let a = LinearBinomial::new([5, 0]);
        assert_eq!(25, a.eval(5));
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
        let a = LinearBinomial::new([1, 0]);
        let c = LinearBinomial::new([0, 1]);
        assert_eq!(a >> 1, c);
    }

    #[test]
    fn test_shr_assign_pos() {
        let mut a = LinearBinomial::new([1, 0]);
        let c = LinearBinomial::new([0, 1]);
        a >>= 1;
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
        let a = LinearBinomial::new([1, 2]);
        assert_eq!(a >> 5, LinearBinomial::zero());
    }

    #[test]
    fn test_shr_assign_to_zero() {
        let mut a = LinearBinomial::new([1, 2]);
        a >>= 5;
        assert_eq!(a, LinearBinomial::zero());
    }

    #[test]
    fn test_derivative_of_zero() {
        let a: LinearBinomial<i32> = LinearBinomial::zero();
        assert_eq!(a.derivative(), LinearBinomial::zero());
    }

    #[test]
    fn test_derivative_of_degree_zero() {
        let a = LinearBinomial::new([0, 1]);
        assert_eq!(a.derivative(), LinearBinomial::zero());
    }

    #[test]
    fn test_derivative() {
        let a = LinearBinomial::new([1, 3]);
        assert_eq!(a.derivative(), LinearBinomial::new([0, 1]));
    }
}
