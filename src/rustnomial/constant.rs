// use std::fmt;
// use std::fmt::Display;
// use std::ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shl, ShlAssign, Shr, ShrAssign};
// use std::str::FromStr;
//
// use num::{One, Zero};
//
// use rustnomial::numerics::{IsNegativeOne, PowUsize};
// use rustnomial::strings::write_leading_term;
// use rustnomial::traits::TermIterator;
// use {Degree, Derivable, Evaluable, GenericPolynomial, Integrable, Integral, Polynomial, Term};
// use FreeSizePolynomial;
// use std::string::ToString;
//
// #[derive(Debug, Clone)]
// pub struct Constant<N> {
//     pub coefficient: N,
// }
//
// impl<N> Constant<N> {
//     /// Create a `Monomial` with coefficient and degree.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Monomial, Degree};
//     /// let monomial = Monomial::new(3.0, 2);
//     /// assert_eq!(3.0, monomial.coefficient);
//     /// assert_eq!(Degree::Num(2), monomial.degree());
//     /// ```
//     pub fn new(coefficient: N) -> Monomial<N> {
//         Constant {
//             coefficient,
//         }
//     }
// }
//
// impl<N: Copy + Zero> Constant<N> {
//     pub fn zero() -> Self {
//         Constant::new(N::zero())
//     }
//     /// Returns the degree of the `Monomial`.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Monomial, Degree};
//     /// let monomial = Monomial::new(3.0, 2);
//     /// assert_eq!(Degree::Num(2), monomial.degree());
//     /// let zero_with_nonzero_deg = Monomial::new(0.0, 2);
//     /// assert_eq!(Degree::NegInf, zero_with_nonzero_deg.degree());
//     /// let nonzero_with_zero_degree = Monomial::new(1.0, 0);
//     /// assert_eq!(Degree::Num(0), nonzero_with_zero_degree.degree());
//     /// ```
//     pub fn degree(&self) -> Degree {
//         if self.is_zero() {
//             Degree::NegInf
//         } else {
//             Degree::Num(0)
//         }
//     }
//
//     /// Returns true if all terms are zero, and false if a non-zero term exists.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Polynomial, Monomial};
//     /// let zero = Monomial::new(0, 1);
//     /// assert!(zero.is_zero());
//     /// let non_zero = Monomial::new(1, 0);
//     /// assert!(!non_zero.is_zero());
//     /// ```
//     pub fn is_zero(&self) -> bool {
//         self.coefficient.is_zero()
//     }
// }
//
// impl<N: Copy + Zero> GenericPolynomial<N> for Constant<N> {
//     /// Return the number of terms in `Monomial`.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Monomial, GenericPolynomial};
//     /// let monomial = Monomial::new(3.0, 2);
//     /// assert_eq!(1, monomial.len());
//     /// assert_eq!(0, Monomial::<i32>::zero().len());
//     /// ```
//     fn len(&self) -> usize {
//         if self.is_zero() {
//             0
//         } else {
//             1
//         }
//     }
//
//     /// Returns the nth term of the `Monomial`.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Monomial, GenericPolynomial, Term};
//     /// let monomial = Monomial::new(5, 2);
//     /// assert_eq!(Term::Term(5, 2), monomial.nth_term(0));
//     /// assert_eq!(Term::ZeroTerm, monomial.nth_term(1));
//     /// ```
//     fn nth_term(&self, index: usize) -> Term<N> {
//         if index != 0 {
//             Term::ZeroTerm
//         } else {
//             Term::new(self.coefficient, 0)
//         }
//     }
//
//     /// Returns an iterator for the `Monomial`, yielding the term constant and degree. Terms are
//     /// iterated over in descending degree order, excluding zero terms.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Monomial, GenericPolynomial};
//     /// let monomial = Monomial::new(5, 2);
//     /// let mut iter = monomial.term_iter();
//     /// assert_eq!(Some((5, 2)), iter.next());
//     /// assert_eq!(None, iter.next());
//     /// ```
//     fn term_iter(&self) -> TermIterator<N> {
//         TermIterator::new(self)
//     }
// }
//
// impl<N> Evaluable<N> for Constant<N>
// where N: Copy,
// {
//     /// Returns the value of the `Monomial` at the given point.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Monomial, Evaluable};
//     /// let monomial = Monomial::new(5, 2);
//     /// assert_eq!(125, monomial.eval(5));
//     /// assert_eq!(1, Monomial::new(1, 0).eval(0));
//     /// ```
//     fn eval(&self, point: N) -> N {
//         self.coefficient
//     }
// }
//
// impl<N> Derivable<N> for Constant<N>
// where
//     N: Zero,
// {
//     /// Returns the derivative of the `Monomial`.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Monomial, Polynomial, Derivable};
//     /// let monomial = Monomial::new(3.0, 2);
//     /// assert_eq!(Monomial::new(6.0, 1), monomial.derivative());
//     /// ```
//     fn derivative(&self) -> Constant<N> {
//         Constant::zero()
//     }
// }
//
// impl<N> Integrable<N> for Constant<N>
// where
//     N: Zero + Copy + AddAssign,
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
//         Integral {
//             polynomial: Polynomial::new(
//             if self.is_zero() {
//                     vec![N::zero()]
//                 } else {
//                     vec![self.coefficient, N::zero()]
//                 }
//             )
//             }
//     }
//
// }
//
// impl<N: PowUsize + Copy> Constant<N> {
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
//     pub fn pow(&self, exp: usize) -> Constant<N> {
//         Constant::new(self.coefficient.upow(exp))
//     }
// }
//
// // TODO: Divmod implementation.
//
// impl<N> PartialEq for Constant<N>
// where
//     N: Zero + PartialEq + Copy,
// {
//     /// Returns true if this `Monomial` is equal to other.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::Monomial;
//     /// let a = Monomial::new(2, 2);
//     /// let b = Monomial::new(2, 2);
//     /// let c = Monomial::new(1, 2);
//     /// assert_eq!(a, b);
//     /// assert_ne!(a, c);
//     /// ```
//     fn eq(&self, other: &Self) -> bool {
//         self.coefficient == other.coefficient
//     }
// }
//
// impl<N> FromStr for Constant<N>
// where
//     N: Zero + One + Copy + AddAssign + FromStr,
// {
//     type Err = String;
//
//     /// Returns a `Polynomial` with the corresponding terms,
//     /// in order of ax^n + bx^(n-1) + ... + cx + d
//     ///
//     /// # Arguments
//     ///
//     /// * ` terms ` - A vector of constants, in decreasing order of degree.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::Monomial;
//     /// use std::str::FromStr;
//     /// // Corresponds to 1.0x^2 + 4.0x + 4.0
//     /// let monomial = Monomial::from_str("5x^2").unwrap();
//     /// assert_eq!(Monomial::new(5, 2), monomial);
//     /// ```
//     fn from_str(s: &str) -> Result<Self, Self::Err> {
//         match Term::from_str(s)? {
//             Ok(Term::ZeroTerm) => Ok(Constant::zero()),
//             Ok(Term::Term(coeff, 0)) => Ok(Constant::new(coeff)),
//             Ok(Term::Term(_, _)) => Err("Unexpected degree.".to_string()),
//         }
//     }
// }
//
// impl<N> fmt::Display for Constant<N>
// where
//     N: Zero + One + PartialEq + Copy + IsNegativeOne + Display,
// {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "{}", self.coefficient)
//     }
// }
//
// impl<N: Copy + Neg<Output = N>> Neg for Constant<N> {
//     type Output = Constant<N>;
//
//     fn neg(self) -> Constant<N> {
//         Constant::new(-self.coefficient)
//     }
// }
//
// // impl<N> ops::Sub<Polynomial<N>> for Polynomial<N>
// //     where N: PartialEq + Zero + Copy + Sub<Output=N> + SubAssign + Neg<Output=N>{
// //     type Output = Polynomial<N>;
// //
// //     fn sub(self, _rhs: Polynomial<N>) -> Polynomial<N> {
// //         if _rhs.len() > self.len() {
// //             let mut terms = _rhs.terms.clone();
// //             let offset = _rhs.len() - self.len();
// //
// //             for index in terms[..offset].iter_mut() {
// //                 *index = -*index;
// //             }
// //
// //             for (index, val) in terms[offset..].iter_mut().zip(self.terms) {
// //                 *index = val - *index;
// //             }
// //             Polynomial::new(terms)
// //         } else {
// //             let mut terms = self.terms.clone();
// //             let offset = terms.len() - _rhs.len();
// //             for (index, val) in terms[offset..].iter_mut().zip(_rhs.terms) {
// //                 *index -= val;
// //             }
// //             Polynomial::new(terms)
// //         }
// //     }
// // }
//
// // impl<N> ops::SubAssign<Polynomial<N>> for Polynomial<N>
// //     where N: Neg<Output=N> + Sub<Output=N> + SubAssign + Copy + Zero + PartialEq {
// //     fn sub_assign(&mut self, _rhs: Polynomial<N>) {
// //         if _rhs.len() > self.len() {
// //             let mut terms = _rhs.terms.clone();
// //             let offset = _rhs.len() - self.len();
// //
// //             for index in terms[..offset].iter_mut() {
// //                 *index = -*index;
// //             }
// //
// //             for (index, &val) in terms[offset..].iter_mut().zip(&self.terms) {
// //                 *index = val - *index;
// //             }
// //             self.terms = terms;
// //         } else {
// //             let offset = self.len() - _rhs.len();
// //             for (index, val) in self.terms[offset..].iter_mut().zip(_rhs.terms) {
// //                 *index -= val;
// //             }
// //         }
// //     }
// // }
//
// // impl<N> ops::Add<Polynomial<N>> for Polynomial<N>
// //     where N: PartialEq + Zero + Copy + AddAssign {
// //     type Output = Polynomial<N>;
// //
// //     fn add(self, _rhs: Polynomial<N>) -> Polynomial<N> {
// //         let (mut terms, small) = if _rhs.len() > self.len() {
// //             (_rhs.terms.clone(), &self.terms)
// //         } else {
// //             (self.terms.clone(), &_rhs.terms)
// //         };
// //
// //         let offset = terms.len() - small.len();
// //
// //         for (index, &val) in terms[offset..].iter_mut().zip(small) {
// //             *index += val;
// //         }
// //
// //         Polynomial::new(terms)
// //     }
// // }
//
// // impl<N: Copy + Zero + PartialEq + AddAssign> ops::AddAssign<Polynomial<N>> for Polynomial<N> {
// //     fn add_assign(&mut self, _rhs: Polynomial<N>) {
// //         if _rhs.len() > self.len() {
// //             let offset = _rhs.len() - self.len();
// //             let mut terms = _rhs.terms.clone();
// //             for (index, &val) in terms[offset..].iter_mut().zip(&self.terms) {
// //                 *index += val;
// //             }
// //             self.terms = terms;
// //         } else {
// //             let offset = self.len() - _rhs.len();
// //             for (index, val) in self.terms[offset..].iter_mut().zip(_rhs.terms) {
// //                 *index += val;
// //             }
// //         }
// //     }
// // }
//
// impl<N: Copy + Mul<Output = N>> Mul<Constant<N>> for Constant<N> {
//     type Output = Constant<N>;
//
//     fn mul(self, _rhs: Constant<N>) -> Constant<N> {
//         Constant::new(self.coefficient * _rhs.coefficient)
//     }
// }
//
// impl<N: MulAssign> MulAssign<Constant<N>> for Constant<N> {
//     fn mul_assign(&mut self, _rhs: Constant<N>) {
//         self.coefficient *= _rhs.coefficient;
//     }
// }
//
// impl<N: Copy + Mul<Output = N>> Mul<&Constant<N>> for Constant<N> {
//     type Output = Constant<N>;
//
//     fn mul(self, _rhs: &Constant<N>) -> Constant<N> {
//         Constant::new(self.coefficient * _rhs.coefficient)
//     }
// }
//
// impl<N> MulAssign<&Constant<N>> for Constant<N>
// where
//     N: MulAssign + AddAssign + Copy,
// {
//     fn mul_assign(&mut self, _rhs: &Constant<N>) {
//         self.coefficient *= _rhs.coefficient;
//     }
// }
//
// impl<N: Mul<Output = N>> Mul<N> for Constant<N> {
//     type Output = Constant<N>;
//
//     fn mul(self, _rhs: N) -> Constant<N> {
//         Constant::new(self.coefficient * _rhs)
//     }
// }
//
// impl<N: MulAssign> MulAssign<N> for Constant<N> {
//     fn mul_assign(&mut self, _rhs: N) {
//         self.coefficient *= _rhs;
//     }
// }
//
// impl<N: Div<Output = N>> Div<N> for Constant<N> {
//     type Output = Constant<N>;
//
//     fn div(self, _rhs: N) -> Constant<N> {
//         Constant::new(self.coefficient / _rhs)
//     }
// }
//
// impl<N: DivAssign> DivAssign<N> for Constant<N> {
//     fn div_assign(&mut self, _rhs: N) {
//         self.coefficient /= _rhs;
//     }
// }
//
// // #[cfg(test)]
// // mod tests {
// //     use FreeSizePolynomial;
// //     use {Derivable, Evaluable, Integrable, Monomial, Polynomial};
// //
// //     #[test]
// //     fn test_eval() {
// //         let a = Monomial::new(5, 2);
// //         assert_eq!(a.eval(5), 125);
// //     }
// //
// //     #[test]
// //     fn test_shl_pos() {
// //         let a = Monomial::new(1, 2);
// //         let c = Monomial::new(1, 7);
// //         assert_eq!(a << 5, c);
// //     }
// //
// //     #[test]
// //     fn test_shl_assign_pos() {
// //         let mut a = Monomial::new(1, 2);
// //         let c = Monomial::new(1, 7);
// //         a <<= 5;
// //         assert_eq!(a, c);
// //     }
// //
// //     #[test]
// //     fn test_shl_neg() {
// //         let a = Monomial::new(1, 7);
// //         let c = Monomial::new(1, 2);
// //         assert_eq!(a << -5, c);
// //     }
// //
// //     #[test]
// //     fn test_shl_assign_neg() {
// //         let mut a = Monomial::new(1, 7);
// //         let c = Monomial::new(1, 2);
// //         a <<= -5;
// //         assert_eq!(a, c);
// //     }
// //
// //     #[test]
// //     fn test_shr_pos() {
// //         let a = Monomial::new(1, 7);
// //         let c = Monomial::new(1, 2);
// //         assert_eq!(a >> 5, c);
// //     }
// //
// //     #[test]
// //     fn test_shr_assign_pos() {
// //         let mut a = Monomial::new(1, 7);
// //         let c = Monomial::new(1, 2);
// //         a >>= 5;
// //         assert_eq!(a, c);
// //     }
// //
// //     #[test]
// //     fn test_shr_neg() {
// //         let a = Monomial::new(1, 2);
// //         let c = Monomial::new(1, 7);
// //         assert_eq!(a >> -5, c);
// //     }
// //
// //     #[test]
// //     fn test_shr_assign_neg() {
// //         let mut a = Monomial::new(1, 2);
// //         let c = Monomial::new(1, 7);
// //         a >>= -5;
// //         assert_eq!(a, c);
// //     }
// //
// //     #[test]
// //     fn test_shr_to_zero() {
// //         let a = Monomial::new(5, 1);
// //         assert_eq!(a >> 5, Monomial::zero());
// //     }
// //
// //     #[test]
// //     fn test_shr_assign_to_zero() {
// //         let mut a = Monomial::new(5, 1);
// //         a >>= 5;
// //         assert_eq!(a, Monomial::zero());
// //     }
// //
// //     #[test]
// //     fn test_derivative_of_zero() {
// //         let a: Monomial<i32> = Monomial::zero();
// //         assert_eq!(Monomial::zero(), a.derivative());
// //     }
// //
// //     #[test]
// //     fn test_derivative_of_monomial_degree_zero() {
// //         let a = Monomial::new(5, 0);
// //         assert_eq!(Monomial::zero(), a.derivative());
// //     }
// //
// //     #[test]
// //     fn test_derivative() {
// //         let a = Monomial::new(5, 3);
// //         assert_eq!(Monomial::new(15, 2), a.derivative());
// //     }
// //
// //     #[test]
// //     fn test_integral() {
// //         let a = Monomial::new(5, 2);
// //         let integral = a.integral();
// //         assert_eq!(
// //             Polynomial::from_terms(vec![(5 / 3, 3)]),
// //             integral.polynomial
// //         );
// //     }
// //
// //     #[test]
// //     fn test_str() {
// //         let a = Monomial::new(5, 2);
// //         assert_eq!(a.to_string(), "5x^2");
// //     }
// // }
