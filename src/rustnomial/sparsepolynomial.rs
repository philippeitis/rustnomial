use std::ops;
use std::fmt;
use std::ops::{Mul, AddAssign, MulAssign, DivAssign, Div, SubAssign, Neg, Sub};
use std::fmt::{Display};

use rustnomial::integral::Integrable;
use rustnomial::numerics::{HasZero, HasOne, IsNegativeOne, Abs, PowUsize};
use rustnomial::traits::{PolynomialDegreeIterator, GenericPolynomial};
use ::{Integral, Evaluable};
use rustnomial::degree::{Term, Degree};
use std::collections::HashMap;
use rustnomial::degree::Term::ZeroTerm;

#[derive(Debug, Clone)]
pub struct SparsePolynomial<N> {
    pub terms: HashMap<usize, N>,
}

fn map_mul<N>(_lhs: &HashMap<usize, N>, _rhs: &HashMap<usize, N>) -> HashMap<usize, N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    let mut terms = HashMap::new();
    for (&rhs_deg, &rterm) in _rhs {
        if rterm == N::zero() {
            continue;
        }

        for (&lhs_deg, &lterm) in _lhs.iter() {
            match terms.get_mut(&(rhs_deg + lhs_deg)) {
                None => {
                    terms.insert(rhs_deg + lhs_deg, rterm * lterm);
                },
                Some(val) => {
                    *val += rterm * lterm;
                }
            }
        }
    }
    terms
}

impl<N: Copy + HasZero + PartialEq> GenericPolynomial<N> for SparsePolynomial<N> {
    fn len(&self) -> usize {
        match self.degree() {
            Degree::NegInf => 0,
            Degree::Num(deg) => deg + 1,
        }
    }

    fn nth_term(&self, index: usize) -> Term<N> {
        let degree = match self.degree() {
            Degree::NegInf => {return ZeroTerm;},
            Degree::Num(x) => {x - index},
        };
        match self.terms.get(&degree) {
            None => ZeroTerm,
            Some(&val) => {
                if val == N::zero() {
                    ZeroTerm
                } else {
                     Term::Term(val, degree)
                }
            }
        }
    }

    /// Returns an iterator for the `Polynomial`, yielding the term constant and degree. Terms are
    /// iterated over in descending degree order, excluding zero terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, GenericPolynomial};
    /// let polynomial = Polynomial::new(vec![1, 0, 2, 3]);
    /// let mut iter = polynomial.degree_iter();
    /// assert_eq!(Some((1, 3)), iter.next());
    /// assert_eq!(Some((2, 1)), iter.next());
    /// assert_eq!(Some((3, 0)), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn degree_iter(&self) -> PolynomialDegreeIterator<N> {
        PolynomialDegreeIterator::new(self)
    }
}

impl<N> SparsePolynomial<N>
    where N: PartialEq + HasZero + Copy + AddAssign {
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
    /// use rustnomial::SparsePolynomial;
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = SparsePolynomial::from_vec(vec![1.0, 4.0, 4.0]);
    /// ```
    pub fn from_terms(terms: Vec<(N, usize)>) -> SparsePolynomial<N> {
        let mut a = SparsePolynomial::new(HashMap::with_capacity(terms.len()));
        for (term, degree) in terms {
            a.add_term(term, degree);
        }
        a
    }

    fn add_term(&mut self, term: N, degree: usize) {
        match self.terms.get_mut(&degree) {
            None => {
                self.terms.insert(degree, term);
            }
            Some(val) => {
                *val += term;
            }
        }
    }
}

impl<N> SparsePolynomial<N>
    where N: PartialEq + HasZero + Copy + Display {

    pub fn from_vec(term_vec: Vec<N>) -> SparsePolynomial<N> {
        let mut terms: HashMap<usize, N> = HashMap::new();
        if term_vec.len() != 0 {
            let degree = term_vec.len() - 1;
            for (index, &val) in term_vec.iter().enumerate() {
                if val != N::zero() {
                    terms.insert(degree - index, val);
                }
            }
        }
        SparsePolynomial { terms }
    }
}

impl<N> SparsePolynomial<N>
    where N: PartialEq + HasZero + Copy {
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
    /// use rustnomial::SparsePolynomial;
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = SparsePolynomial::from_vec(vec![1.0, 4.0, 4.0]);
    /// ```
    pub fn new(terms: HashMap<usize, N>) -> SparsePolynomial<N> {
        SparsePolynomial{terms}
    }

    // pub fn from_vec(term_vec: Vec<N>) -> SparsePolynomial<N> {
    //     let mut terms: HashMap<usize, N> = HashMap::new();
    //     if terms.len() != 0 {
    //         let degree = terms.len() - 1;
    //         for (index, &val) in term_vec.iter().enumerate() {
    //             println!("terms[{}] = {}", degree - index, val);
    //             terms.insert(degree - index, val);
    //         }
    //     }
    //     SparsePolynomial{terms}
    // }

    /// Returns the degree of the `Polynomial` it is called on, corresponding to the
    /// largest non-zero term.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, Degree};
    /// let polynomial = Polynomial::new(vec![1.0, 4.0, 4.0]);
    /// assert_eq!(Degree::Num(2), polynomial.degree());
    /// ```
    pub fn degree(&self) -> Degree {
        let mut max_term = match self.terms.get(&0) {
            None => N::zero(),
            Some(&x) => x
        };
        let mut max_degree = 0;

        for (&degree, &coeff) in self.terms.iter() {
            if degree > max_degree && coeff != N::zero() {
                max_term = coeff;
                max_degree = degree;
            }
        }

        if max_term == N::zero() {
            Degree::NegInf
        } else {
            Degree::Num(max_degree)
        }
    }

    /// Reduces the size of the `Polynomial` in memory any terms are zero.
    pub fn trim(&mut self) {
        let mut new_map = HashMap::new();
        for (&degree, &coeff) in self.terms.iter() {
            if coeff != N::zero() {
                new_map.insert(degree, coeff);
            }
        }
        self.terms = new_map;
    }

    /// Returns true if all terms are zero, and false if a non-zero term exists.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::SparsePolynomial;
    /// let zero = SparsePolynomial::from_vec(vec![0, 0]);
    /// assert!(zero.is_zero());
    /// let non_zero = SparsePolynomial::from_vec(vec![0, 1]);
    /// assert!(!non_zero.is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.degree() == Degree::NegInf
    }

    // / Returns an iterator for the `Polynomial`, yielding term constants. Terms are iterated over
    // / in descending degree order, exluding leading zero terms.
    // /
    // / # Example
    // /
    // / ```
    // / use rustnomial::Polynomial;
    // / let polynomial = Polynomial::new(vec![0, 1, 0, 2, 3]);
    // / let mut iter = polynomial.iter();
    // / assert_eq!(Some(1), iter.next());
    // / assert_eq!(Some(0), iter.next());
    // / assert_eq!(Some(2), iter.next());
    // / assert_eq!(Some(3), iter.next());
    // / assert_eq!(None, iter.next());
    // / ```
    // pub fn iter(&self) -> PolynomialIterator<N> {
    //     PolynomialIterator {
    //         polynomial: self,
    //         index: 0
    //     }
    // }
}

impl<N> Evaluable<N> for SparsePolynomial<N>
    where N: HasZero + PowUsize + Copy + AddAssign + Mul<Output=N> + PartialEq {
    fn eval(&self, point: N) -> N {
        let mut sum = N::zero();
        for (val, degree) in self.degree_iter() {
            sum += val * point.upow(degree);
        }
        sum
    }
}

impl<N> SparsePolynomial<N>
    where N: PartialEq + HasZero + From<u8> + Copy + Mul<Output=N> {
    /// Returns the derivative of the `Polynomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::SparsePolynomial;
    /// let polynomial = SparsePolynomial::from_vec(vec![4, 1, 5]);
    /// assert_eq!(SparsePolynomial::from_vec(vec![8, 1]), polynomial.derivative());
    /// ```
    pub fn derivative(&self) -> SparsePolynomial<N> {
        let mut terms = HashMap::with_capacity(self.len());
        // TODO: Fix for degrees of arbitrary size.
        for (coeff, degree) in self.degree_iter() {
            if degree != 0 {
                terms.insert(degree - 1, coeff * N::from(degree as u8));
            }
        }
        SparsePolynomial { terms }
    }
}

// TODO: Make Integral generic over Polynomial, SparsePolynomial
// impl<N> Integrable<N> for SparsePolynomial<N>
//     where N: PartialEq + HasZero + From<u8> + Copy + DivAssign + fmt::Display {
//     /// Returns the integral of the `Polynomial`.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::{Polynomial, Integrable};
//     /// let polynomial = Polynomial::new(vec![1.0, 2.0, 5.0]);
//     /// let integral = polynomial.integral();
//     /// assert_eq!(Polynomial::new(vec![1.0/3.0, 1.0, 5.0, 0.0]), integral.polynomial);
//     /// ```
//     fn integral(&self) -> Integral<N> {
//         let index = first_nonzero_index(&self.terms);
//         // TODO: Fix for degrees of arbitrary size.
//         let mut degree = (self.len() - index + 1) as u8;
//         let mut terms = self.terms[index..].to_vec();
//         for term in terms.iter_mut() {
//             degree -= 1;
//             *term /= N::from(degree);
//         }
//         terms.push(N::zero());
//         Integral {
//             polynomial: Polynomial { terms }
//         }
//     }
// }

impl<N> SparsePolynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + HasOne + PartialEq {
    pub fn borrow_mul(&self, _rhs: &SparsePolynomial<N>) -> SparsePolynomial<N> {
        SparsePolynomial{terms: map_mul(&self.terms, &_rhs.terms)}
    }

    /// Raises the `Polynomial` to the power of exp, using exponentiation by squaring.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::SparsePolynomial;
    /// let polynomial = SparsePolynomial::from_vec(vec![1.0, 2.0]);
    /// let polynomial_sqr = polynomial.pow(2);
    /// let polynomial_cub = polynomial.pow(3);
    /// assert_eq!(polynomial.clone() * polynomial.clone(), polynomial_sqr);
    /// assert_eq!(polynomial_sqr.clone() * polynomial.clone(), polynomial_cub);
    /// ```
    pub fn pow(&self, exp: usize) -> SparsePolynomial<N> {
        if exp == 0 {
            SparsePolynomial {
                terms: {
                    let mut terms = HashMap::new();
                    terms.insert(0, N::one());
                    terms
                }
            }
        } else if exp == 1 {
            SparsePolynomial::new(self.terms.clone())
        } else if exp == 2 {
            self.borrow_mul(self)
        } else if exp % 2 == 0 {
            self.pow(exp / 2).pow(2)
        } else {
            self.borrow_mul(&self.pow(exp - 1))
        }
    }
}

// TODO: Implement this.
// impl<N> SparsePolynomial<N>
//     where N: Copy + PartialEq + HasZero + SubAssign + Mul<Output=N> + Div<Output=N> {
//     /// Divides self by the given `Polynomial`, and returns the quotient and remainder.
//     ///
//     /// # Example
//     ///
//     /// ```
//     /// use rustnomial::Polynomial;
//     /// let polynomial = Polynomial::new(vec![1.0, 2.0]);
//     /// let polynomial_sqr = polynomial.pow(2);
//     /// let polynomial_cub = polynomial.pow(3);
//     /// assert_eq!(polynomial.clone() * polynomial.clone(), polynomial_sqr);
//     /// assert_eq!(polynomial_sqr.clone() * polynomial.clone(), polynomial_cub);
//     /// ```
//     pub fn div_mod(&self, _rhs: &Polynomial<N>) -> Result<(Polynomial<N>, Polynomial<N>), &'static str> {
//         fn vec_sub_w_scale<N>(_lhs: &mut Vec<N>, _lhs_degree: usize, _rhs: &Vec<N>, _rhs_deg: usize, _rhs_scale: N)
//             where N: Copy + Mul<Output=N> + SubAssign {
//             let loc = _lhs.len() - _lhs_degree - 1;
//             for (_lhs_t, _rhs_t) in _lhs[loc..].iter_mut().zip(_rhs) {
//                 *_lhs_t -= (*_rhs_t) * _rhs_scale;
//             }
//         }
//
//         let zero = N::zero();
//         let (_rhs_first, _rhs_deg) = match first_term(&_rhs.terms) {
//             Term::ZeroTerm => {
//                 return Err("Can't divide by 0.");
//             },
//             Term::Term(coeff, deg) => {
//                 (coeff, deg)
//             }
//         };
//
//         let (mut term, mut self_degree) = match first_term(&self.terms) {
//             Term::ZeroTerm => {
//                 let zero_vec = vec![zero; 1];
//                 return Ok((Polynomial::new(zero_vec), Polynomial::new(self.terms.clone())));
//             }
//             Term::Term(term, degree) => {
//                 if degree < _rhs_deg {
//                     let zero_vec = vec![zero; 1];
//                     return Ok((Polynomial::new(zero_vec), Polynomial::new(self.terms.clone())));
//                 }
//                 (term, degree)
//             }
//         };
//
//         let mut remainder = self.terms.clone();
//         let offset = self_degree - _rhs_deg;
//         let mut div = vec![zero; offset + 1];
//
//         while self_degree >= _rhs_deg {
//             let scale = term / _rhs_first;
//             vec_sub_w_scale(&mut remainder, self_degree, &_rhs.terms, _rhs_deg, scale);
//             div[offset - (self_degree - _rhs_deg)] = scale;
//             match first_term(&remainder) {
//                 Term::ZeroTerm => {
//                     break;
//                 },
//                 Term::Term(coeff, degree) => {
//                     term = coeff;
//                     self_degree = degree;
//                 }
//             }
//         }
//
//         Ok((Polynomial::new(div), Polynomial::new(remainder)))
//     }
// }

impl<N> PartialEq for SparsePolynomial<N>
    where N: PartialEq + HasZero + Copy {
    /// Returns true if self has the same terms as other.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::SparsePolynomial;
    /// let a = SparsePolynomial::from_vec(vec![1.0, 2.0]);
    /// let b = SparsePolynomial::from_vec(vec![2.0, 2.0]);
    /// let c = SparsePolynomial::from_vec(vec![1.0, 0.0]);
    /// assert_ne!(a, b);
    /// assert_ne!(a, c);
    /// assert_eq!(a, b - c);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        if self.degree() != other.degree() {
            return false;
        }

        let mut self_iter = self.degree_iter();
        let mut other_iter = other.degree_iter();

        loop {
            let self_term = self_iter.next();
            let other_term = other_iter.next();
            if self_term != other_term {
                return false;
            }

            // Can only get here if self_term and other_term are both None
            if self_term == None {
                return true;
            }
        }
    }
}

impl<N> fmt::Display for SparsePolynomial<N>
    where N: HasZero + HasOne + Copy + IsNegativeOne + PartialEq + PartialOrd + Display + Abs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut iter = self.degree_iter();
        let one = N::one();
        let zero = N::zero();
        match iter.next() {
            None => {
                return write!(f, "0");
            }

            Some((term, degree)) => {
                if term.is_negative_one() {
                    write!(f, "-")?;
                } else if (term != one) || (degree == 0) {
                    write!(f, "{}", term)?;
                }

                match degree {
                    0 => {},
                    1 => {write!(f, "x")?;},
                    _ => {write!(f, "x^{}", degree)?;}
                }
            }
        }


        for (term, degree) in iter {
            if term > zero {
                write!(f, " + ")?;
            } else {
                write!(f, " - ")?;
            }

            let term = term.abs();

            if (term != one) || (degree == 0) {
                write!(f, "{}", term)?;
            }

            match degree {
                0 => {},
                1 => {write!(f, "x")?;},
                _ => {write!(f, "x^{}", degree)?;}
            }
        }

        write!(f, "")
    }
}

impl<N> ops::Neg for SparsePolynomial<N>
    where N: PartialEq + HasZero + Copy + Neg<Output=N>{
    type Output = SparsePolynomial<N>;

    fn neg(self) -> SparsePolynomial<N> {
        let mut terms = HashMap::new();
        for (&deg, &coeff) in self.terms.iter() {
            terms.insert(deg, -coeff);
        }
        SparsePolynomial::new(terms)
    }
}

impl<N> ops::Sub<SparsePolynomial<N>> for SparsePolynomial<N>
    where N: PartialEq + HasZero + Copy + Sub<Output=N> + SubAssign + Neg<Output=N>{
    type Output = SparsePolynomial<N>;

    fn sub(self, _rhs: SparsePolynomial<N>) -> SparsePolynomial<N> {
        let mut terms = self.terms.clone();
        for (deg, coeff) in _rhs.terms {
            match terms.get_mut(&deg) {
                None => {
                    terms.insert(deg, -coeff);
                }
                Some(val) => {
                    *val -= coeff;
                }
            }
        }
        SparsePolynomial{terms}
    }
}

impl<N> ops::SubAssign<SparsePolynomial<N>> for SparsePolynomial<N>
    where N: Neg<Output=N> + Sub<Output=N> + SubAssign + Copy + HasZero + PartialEq {
    fn sub_assign(&mut self, _rhs: SparsePolynomial<N>) {
        for (deg, coeff) in _rhs.terms {
            match self.terms.get_mut(&deg) {
                None => {
                    self.terms.insert(deg, -coeff);
                }
                Some(val) => {
                    *val -= coeff;
                }
            }
        }
    }
}

impl<N> ops::Add<SparsePolynomial<N>> for SparsePolynomial<N>
    where N: PartialEq + HasZero + Copy + AddAssign {
    type Output = SparsePolynomial<N>;

    fn add(self, _rhs: SparsePolynomial<N>) -> SparsePolynomial<N> {
        let mut terms = HashMap::new();
        for (deg, coeff) in _rhs.terms {
            terms.insert(deg, coeff);
        }
        for (&deg, &coeff) in self.terms.iter() {
            match terms.get_mut(&deg) {
                None => {
                    terms.insert(deg, coeff);
                }
                Some(val) => {
                    *val += coeff;
                }
            }
        }
        SparsePolynomial{terms}
    }
}

impl<N: Copy + HasZero + PartialEq + AddAssign> ops::AddAssign<SparsePolynomial<N>> for SparsePolynomial<N> {
    fn add_assign(&mut self, _rhs: SparsePolynomial<N>) {
        for (&deg, &coeff) in _rhs.terms.iter() {
            match self.terms.get_mut(&deg) {
                None => {
                    self.terms.insert(deg, coeff);
                }
                Some(val) => {
                    *val += coeff;
                }
            }
        }
    }
}

impl<N> ops::Mul<SparsePolynomial<N>> for SparsePolynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    type Output = SparsePolynomial<N>;

    fn mul(self, _rhs: SparsePolynomial<N>) -> SparsePolynomial<N> {
        SparsePolynomial{terms: map_mul(&self.terms, &_rhs.terms)}
    }
}

impl<N> ops::MulAssign<SparsePolynomial<N>> for SparsePolynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    fn mul_assign(&mut self, _rhs: SparsePolynomial<N>) {
        self.terms = map_mul(&self.terms, &_rhs.terms);
    }
}

impl<N> ops::Mul<&SparsePolynomial<N>> for SparsePolynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    type Output = SparsePolynomial<N>;

    fn mul(self, _rhs: &SparsePolynomial<N>) -> SparsePolynomial<N> {
        SparsePolynomial::new(map_mul(&self.terms, &_rhs.terms))
    }
}

impl<N> ops::MulAssign<&SparsePolynomial<N>> for SparsePolynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    fn mul_assign(&mut self, _rhs: &SparsePolynomial<N>) {
        self.terms = map_mul(&self.terms, &_rhs.terms);
    }
}

impl<N: PartialEq + HasZero + Copy + Mul<Output=N>> ops::Mul<N> for SparsePolynomial<N> {
    type Output = SparsePolynomial<N>;

    fn mul(self, _rhs: N) -> SparsePolynomial<N> {
        let mut terms = HashMap::new();
        for (&deg, &coeff) in self.terms.iter() {
            terms.insert(deg, coeff * _rhs);
        }

        SparsePolynomial::new(terms)
    }
}

impl<N: Copy + MulAssign> ops::MulAssign<N> for SparsePolynomial<N> {
    fn mul_assign(&mut self, _rhs: N) {
        for (_, coeff) in self.terms.iter_mut() {
            *coeff *= _rhs;
        }
    }
}

impl<N> ops::Div<N> for SparsePolynomial<N>
    where N: PartialEq + HasZero + Copy + Div<Output=N> {
    type Output = SparsePolynomial<N>;

    fn div(self, _rhs: N) -> SparsePolynomial<N> {
        let mut terms = HashMap::new();
        for (&deg, &coeff) in self.terms.iter() {
            terms.insert(deg, coeff / _rhs);
        }

        SparsePolynomial::new(terms)
    }
}

impl<N> ops::DivAssign<N> for SparsePolynomial<N>
    where N: PartialEq + Copy + DivAssign {
    fn div_assign(&mut self, _rhs: N) {
        for (_, coeff) in self.terms.iter_mut() {
            *coeff /= _rhs;
        }
    }
}

impl<N: PartialEq + HasZero + Copy> ops::Shl<i32> for SparsePolynomial<N> {
    type Output = SparsePolynomial<N>;

    fn shl(self, _rhs: i32) -> SparsePolynomial<N> {
        if _rhs < 0 {
            self >> -_rhs
        } else {
            let mut terms = HashMap::new();
            let _rhs = _rhs as usize;
            for (&deg, &coeff) in self.terms.iter() {
                terms.insert(deg + _rhs, coeff);
            }
            SparsePolynomial::new(terms)
        }
    }
}

impl<N: HasZero + Copy> ops::ShlAssign<i32> for SparsePolynomial<N> {
    fn shl_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self >>= -_rhs;
        } else {
            let mut terms = HashMap::new();
            let _rhs = _rhs as usize;
        for (&deg, &coeff) in self.terms.iter() {
                terms.insert(deg + _rhs, coeff);
            }
            self.terms = terms;
        }
    }
}

impl<N: PartialEq + HasZero + Copy> ops::Shr<i32> for SparsePolynomial<N> {
    type Output = SparsePolynomial<N>;

    fn shr(self, _rhs: i32) -> SparsePolynomial<N> {
        if _rhs < 0 {
            self << -_rhs
        } else {
            let mut terms = HashMap::new();
            let _rhs = _rhs as usize;
            for (&deg, &coeff) in self.terms.iter() {
                terms.insert(deg - _rhs, coeff);
            }
            SparsePolynomial::new(terms)
        }
    }
}

impl<N: HasZero + Copy> ops::ShrAssign<i32> for SparsePolynomial<N> {
    fn shr_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self <<= -_rhs;
        } else {
            let mut terms = HashMap::new();
            let _rhs = _rhs as usize;
            for (&deg, &coeff) in self.terms.iter() {
                terms.insert(deg - _rhs, coeff);
            }
            self.terms = terms;
        }
    }
}

/// TODO:
/// modulo floordiv
#[cfg(test)]
mod tests {
    use std::fmt::Write;
    use super::SparsePolynomial;
    use ::{Integrable, Degree, Evaluable};

    #[test]
    fn test_eval() {
        let a = SparsePolynomial::from_vec(vec![1, 2, 3]);
        assert_eq!(a.eval(5), 25 + 2 * 5 + 3);
    }

    #[test]
    fn test_derivative() {
        let a = SparsePolynomial::from_vec(vec![1, 2, 3]);
        let b = SparsePolynomial::from_vec(vec![2, 2]);
        assert_eq!(a.derivative(), b);

        let a = SparsePolynomial::from_vec(vec![0, 1, 2, 3]);
        assert_eq!(a.derivative(), b);

        let a = SparsePolynomial::from_vec(vec![1, 2, 3, 4]);
        let b = SparsePolynomial::from_vec(vec![3, 4, 3]);
        assert_eq!(a.derivative(), b);

    }

    // #[test]
    // fn test_integral() {
    //     let a = SparsePolynomial::from_vec(vec![3, 2, 1]);
    //     let b = SparsePolynomial::from_vec(vec![1, 1, 1, 0]);
    //     assert_eq!(a.integral().polynomial, b);
    // }
    //
    // #[test]
    // fn test_integral_eval() {
    //     let a = SparsePolynomial::from_vec(vec![3, 2, 1]);
    //     assert_eq!(a.integral().eval(0, 1), 3);
    // }

    // #[test]
    // fn test_integral_const_substitute() {
    //     let a = SparsePolynomial::from_vec(vec![3, 2, 1]);
    //     let b = SparsePolynomial::from_vec(vec![1, 1, 1, 5]);
    //     assert_eq!(a.integral().replace_c(5), b);
    // }


    #[test]
    fn test_add_lhs_bigger() {
        let a = SparsePolynomial::from_vec(vec![1, 2, 3]);
        let b = SparsePolynomial::from_vec(vec![1, 2, 3, 4]);
        let c = SparsePolynomial::from_vec(vec![1, 3, 5, 7]);
        assert_eq!(b + a, c);
    }

    #[test]
    fn test_add_rhs_bigger() {
        let a = SparsePolynomial::from_vec(vec![1, 2, 3]);
        let b = SparsePolynomial::from_vec(vec![1, 2, 3, 4]);
        let c = SparsePolynomial::from_vec(vec![1, 3, 5, 7]);
        assert_eq!(a + b, c);
    }

    #[test]
    fn test_add_lhs_bigger_assign() {
        let a = SparsePolynomial::from_vec(vec![1, 2, 3]);
        let mut b = SparsePolynomial::from_vec(vec![1, 2, 3, 4]);
        b += a;
        let c = SparsePolynomial::from_vec(vec![1, 3, 5, 7]);
        assert_eq!(b, c);
    }

    #[test]
    fn test_add_rhs_bigger_assign() {
        let mut a = SparsePolynomial::from_vec(vec![1, 2, 3]);
        let b = SparsePolynomial::from_vec(vec![1, 2, 3, 4]);
        a += b;
        let c = SparsePolynomial::from_vec(vec![1, 3, 5, 7]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_sub_lhs_bigger() {
        let a = SparsePolynomial::from_vec(vec![2, 3, 4]);
        let b = SparsePolynomial::from_vec(vec![1, 2, 3, 4]);
        let c = SparsePolynomial::from_vec(vec![1, 0, 0, 0]);
        assert_eq!(b - a, c);
    }

    #[test]
    fn test_sub_rhs_bigger() {
        let a = SparsePolynomial::from_vec(vec![2, 3, 4]);
        let b = SparsePolynomial::from_vec(vec![1, 2, 3, 4]);
        let c = SparsePolynomial::from_vec(vec![-1, 0, 0, 0]);
        assert_eq!(a - b, c);
    }

    #[test]
    fn test_sub_lhs_bigger_assign() {
        let a = SparsePolynomial::from_vec(vec![2, 3, 4]);
        let mut b = SparsePolynomial::from_vec(vec![1, 2, 3, 4]);
        b -= a;
        let c = SparsePolynomial::from_vec(vec![1, 0, 0, 0]);
        assert_eq!(b, c);
    }

    #[test]
    fn test_sub_rhs_bigger_assign() {
        let mut a = SparsePolynomial::from_vec(vec![2, 3, 4]);
        let b = SparsePolynomial::from_vec(vec![1, 2, 3, 4]);
        a -= b;
        let c = SparsePolynomial::from_vec(vec![-1, 0, 0, 0]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_negate() {
        let a = SparsePolynomial::from_vec(vec![1, 2, 3, 0, -5]);
        let c = SparsePolynomial::from_vec(vec![-1, -2, -3, 0, 5]);
        assert_eq!(-a, c);
    }

    #[test]
    fn test_mul_poly() {
        let a = SparsePolynomial::from_vec(vec![1, 2]);
        let b = a.clone();
        let c = SparsePolynomial::from_vec(vec![1, 4, 4]);
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_mul_assign_poly() {
        let mut a = SparsePolynomial::from_vec(vec![1, 2]);
        let b = a.clone();
        a *= b;
        let c = SparsePolynomial::from_vec(vec![1, 4, 4]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_mul_num() {
        let a = SparsePolynomial::from_vec(vec![1, 2]);
        let c = SparsePolynomial::from_vec(vec![10, 20]);
        assert_eq!(a * 10, c);
    }

    #[test]
    fn test_mul_assign_num() {
        let mut a = SparsePolynomial::from_vec(vec![1, 2]);
        a *= 10;
        let c = SparsePolynomial::from_vec(vec![10, 20]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_equality() {
        let a = SparsePolynomial::from_vec(vec![1, 2]);
        let c = SparsePolynomial::from_vec(vec![0, 0, 0, 1, 2]);
        assert_eq!(a, c);

        let c = SparsePolynomial::from_vec(vec![1, 2, 0, 0, 0]);

        assert_ne!(a, c);
    }

    #[test]
    fn test_equality_first_match() {
        let a = SparsePolynomial::from_vec(vec![1, 2]);
        let b = SparsePolynomial::from_vec(vec![1, 0]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_equality_different() {
        let a = SparsePolynomial::from_vec(vec![1, 2]);
        let b = SparsePolynomial::from_vec(vec![3, 7, 4]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_shl_pos() {
        let a = SparsePolynomial::from_vec(vec![1, 2]);
        let c = SparsePolynomial::from_vec(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a << 5, c);
    }

    #[test]
    fn test_shl_assign_pos() {
        let mut a = SparsePolynomial::from_vec(vec![1, 2]);
        a <<= 5;
        let c = SparsePolynomial::from_vec(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shl_neg() {
        let a = SparsePolynomial::from_vec(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = SparsePolynomial::from_vec(vec![1, 2]);
        assert_eq!(a << -5, c);
    }

    #[test]
    fn test_shl_assign_neg() {
        let mut a = SparsePolynomial::from_vec(vec![1, 2, 0, 0, 0, 0, 0]);
        a <<= -5;
        let c = SparsePolynomial::from_vec(vec![1, 2]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shr_pos() {
        let a = SparsePolynomial::from_vec(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = SparsePolynomial::from_vec(vec![1, 2]);
        assert_eq!(a >> 5, c);
    }

    #[test]
    fn test_shr_assign_pos() {
        let mut a = SparsePolynomial::from_vec(vec![1, 2, 0, 0, 0, 0, 0]);
        a >>= 5;
        let c = SparsePolynomial::from_vec(vec![1, 2]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shr_neg() {
        let a = SparsePolynomial::from_vec(vec![1, 2]);
        let c = SparsePolynomial::from_vec(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a >> -5, c);
    }

    #[test]
    fn test_shr_assign_neg() {
        let mut a = SparsePolynomial::from_vec(vec![1, 2]);
        a >>= -5;
        let c = SparsePolynomial::from_vec(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_exp() {
        let a = &SparsePolynomial::from_vec(vec![1, 2]);
        let mut b = a.clone();
        assert_eq!(SparsePolynomial::from_vec(vec![1]), a.pow(0));
        for i in 1..10 {
            assert_eq!(b, a.pow(i));
            b *= a;
        }
    }

    #[test]
    fn test_polynomial_str_all_zeroes() {
        let a = SparsePolynomial::from_vec(vec![0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a: SparsePolynomial<i8> = SparsePolynomial::from_vec(vec![]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a = SparsePolynomial::from_vec(vec![0, 0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a = SparsePolynomial::from_vec(vec![0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

    }

    #[test]
    fn test_polynomial_str() {
        let a = SparsePolynomial::from_vec(vec![-1, -2, 3]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^2 - 2x + 3");
    }

    #[test]
    fn test_polynomial_str_has_zeroes() {
        let a = SparsePolynomial::from_vec(vec![-1, -2, 0, 0, 3]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^4 - 2x^3 + 3");
    }

    #[test]
    fn test_polynomial_str_has_ones() {
        let a = SparsePolynomial::from_vec(vec![-1, -1, -1, 0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^3 - x^2 - x");
    }

    #[test]
    fn test_polynomial_str_has_negative() {
        let a = SparsePolynomial::from_vec(vec![-2, -1, -1, 0]);
        println!("a.terms: {:?}", a.terms);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-2x^3 - x^2 - x");
    }

    // #[test]
    // fn test_integral_str() {
    //     let a = SparsePolynomial::from_vec(vec![-3, -2, 1]).integral();
    //     let mut a_str = String::new();
    //     write!(&mut a_str, "{}", a).unwrap();
    //     assert_eq!(a_str, "-x^3 - x^2 + x + C");
    // }

    #[test]
    fn test_degree() {
        let a = SparsePolynomial::from_vec(vec![0, 0, 0, -1, -2, 3]);
        assert_eq!(Degree::Num(2), a.degree());
    }

    // #[test]
    // fn test_iter() {
    //     let mut num_iters = 0;
    //     let a = SparsePolynomial::from_vec(vec![0, 0, 0, -1, -2, 3]);
    //     let b=  vec![-1, -2, 3];
    //     for (a_val, b_val) in a.iter().zip(b) {
    //         num_iters += 1;
    //         assert_eq!(a_val, b_val);
    //     }
    //
    //     assert_eq!(num_iters, 3);
    // }
}