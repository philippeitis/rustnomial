use std::ops;
use std::fmt;
use std::ops::{Mul, AddAssign, MulAssign, DivAssign, Div, SubAssign, Neg, Sub};
use std::fmt::{Display};

use rustnomial::integral::Integrable;
use rustnomial::numerics::{HasZero, HasOne, IsNegativeOne, Abs};
use rustnomial::traits::{PolynomialDegreeIterator, GenericPolynomial};
use Integral;
use rustnomial::degree::{Term, Degree};

#[macro_export]
macro_rules! polynomial {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            Polynomial::new(temp_vec)
        }
    };
}

#[macro_export]
macro_rules! derivative {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            Polynomial::new(temp_vec).derivative()
        }
    };
}
//
// trait GenericPolynomial<N: Copy + Mul<Output=N>> {
//     fn from_vec(terms: Vec<N>) -> GenericPolynomial<N>;
//
//     fn add_term(&self, term: &N, deg: usize);
//
//     fn scale(&mut self, _rhs: &N);
//
//     fn mul(&self, _rhs: &GenericPolynomial<N>) -> &GenericPolynomial<N> {
//        let mut result = GenericPolynomial::from_vec(vec![]);
//        for (rterm, rdeg) in self.degree_iter() {
//            for (lterm, ldeg) in _rhs.degree_iter() {
//                result.add_term(rterm * lterm, rdeg, ldeg);
//            }
//        }
//
//        result
//    }
//
//     fn iter(&self) -> PolynomialDegreeIterator<N>;
//
//     fn degree_iter(&self) -> PolynomialDegreeIterator<N>;
// }

#[derive(Debug, Clone)]
pub struct Polynomial<N> {
    pub terms: Vec<N>,
}

pub struct PolynomialIterator<'a, N> {
    polynomial: &'a Polynomial<N>,
    index: usize,
}

impl<N: Copy + HasZero + PartialEq> Iterator for PolynomialIterator<'_, N> {
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.polynomial.len() {
            let ret_val = self.polynomial.terms[self.index];
            self.index += 1;
            Some(ret_val)
        } else {
            None
        }
    }
}

fn first_nonzero_index<N>(terms: &Vec<N>) -> usize
    where N: PartialEq + HasZero + Copy {
    let mut ind = 0;
    let zero = N::zero();
    while (ind < terms.len()) && (terms[ind] == zero) {
        ind += 1;
    }
    ind
}

fn vec_mul<N>(_lhs: &Vec<N>, _rhs: &Vec<N>) -> Vec<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    let _rhs_ind = first_nonzero_index(&_rhs);
    let _lhs = &_lhs[first_nonzero_index(&_lhs)..];
    let zero = N::zero();
    let mut terms = vec![zero; _rhs.len() + _lhs.len() - 1 - _rhs_ind];
    for (index, &rterm) in _rhs[_rhs_ind..].iter().enumerate() {
        if rterm == zero {
            continue;
        }
        for (&lterm, term) in _lhs.iter().zip(terms[index..].iter_mut()) {
            *term += rterm * lterm;
        }
    }
    terms
}

// fn vec_sub(_lhs: &mut Vec<i32>, _rhs: Vec<i32>) {
//     for (_lhs_t, _rhs_t) in _lhs[_lhs.len() - _rhs.len()..].iter_mut().zip(_rhs) {
//         *_lhs_t -= _rhs_t;
//     }
// }

fn degree<N>(poly_vec: &Vec<N>) -> Degree
    where N: PartialEq + HasZero + Copy {
    let zero = N::zero();
    for (ind, &val) in poly_vec.iter().enumerate() {
        if val != zero {
            return Degree::Num(poly_vec.len() - ind - 1);
        }
    }

    Degree::NegInf
}

fn degree_and_first_val<N>(poly_vec: &Vec<N>) -> (usize, N)
    where N: PartialEq + HasZero + Copy {
    let zero = N::zero();
    for (ind, &val) in poly_vec.iter().enumerate() {
        if val != zero {
            return (poly_vec.len() - ind - 1, val);
        }
    }

    (0, zero)
}
//
// fn generic_pow<N>(base: N, exp: usize) -> N
//     where N: Copy + Mul<Output=N> + HasOne {
//     if exp == 0 {
//         N::one()
//     } else if exp == 1 {
//         base
//     } else if exp == 2 {
//         base * base
//     } else if exp % 2 == 0 {
//         generic_pow(generic_pow(base, exp / 2), 2)
//     } else {
//         base * generic_pow(base, exp - 1)
//     }
// }

impl<N: Copy + HasZero + PartialEq> GenericPolynomial<N> for Polynomial<N> {
    fn len(&self) -> usize {
        self.terms.len()
    }

    fn nth_term(&self, index: usize) -> Term<N> {
        let x = self.terms[index];
        if x == N::zero() {
            Term::ZeroTerm
        } else {
            Term::Term(x, self.len() - index - 1)
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

impl<N> Polynomial<N>
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
    /// use rustnomial::Polynomial;
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = Polynomial::new(vec![1.0, 4.0, 4.0]);
    /// ```
    pub fn from_terms(terms: Vec<(N, usize)>) -> Polynomial<N> {
        let mut a = Polynomial::new(vec![]);
        for (term, degree) in terms {
            a.add_term(term, degree);
        }
        a
    }

    fn add_term(&mut self, term: N, degree: usize) {
        if self.len() < degree + 1 {
            let mut terms = vec![N::zero(); degree + 1 - self.len()];
            terms.extend(&self.terms);
            self.terms = terms;
        }
        let index = self.len() - degree - 1;
        self.terms[index] += term;
    }
}

impl<N> Polynomial<N>
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
    /// use rustnomial::Polynomial;
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = Polynomial::new(vec![1.0, 4.0, 4.0]);
    /// ```
    pub fn new(terms: Vec<N>) -> Polynomial<N> {
        let first_non_zero = first_nonzero_index(&terms);
        Polynomial {
            terms: if first_non_zero != 0 {
                terms[first_non_zero..].to_vec()
            } else {
                terms
            }
        }
    }

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
        degree(&self.terms)
    }

    /// Reduces the size of the `Polynomial` in memory if the leading terms are zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Polynomial;
    /// let mut polynomial = Polynomial::new(vec![1.0, 4.0, 4.0]);
    /// polynomial.terms = vec![0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 4.0];
    /// polynomial.trim();
    /// assert_eq!(vec![1.0, 4.0, 4.0], polynomial.terms);
    /// ```
    pub fn trim(&mut self) {
        let ind = first_nonzero_index(&self.terms);
        if ind != 0 {
            self.terms = self.terms[ind..].to_vec();
        };
    }

    /// Returns true if all terms are zero, and false if a non-zero term exists.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Polynomial;
    /// let zero = Polynomial::new(vec![0, 0]);
    /// assert!(zero.is_zero());
    /// let non_zero = Polynomial::new(vec![0, 1]);
    /// assert!(!non_zero.is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.degree() == Degree::NegInf
    }

    /// Returns an iterator for the `Polynomial`, yielding term constants. Terms are iterated over
    /// in descending degree order, exluding leading zero terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Polynomial;
    /// let polynomial = Polynomial::new(vec![0, 1, 0, 2, 3]);
    /// let mut iter = polynomial.iter();
    /// assert_eq!(Some(1), iter.next());
    /// assert_eq!(Some(0), iter.next());
    /// assert_eq!(Some(2), iter.next());
    /// assert_eq!(Some(3), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn iter(&self) -> PolynomialIterator<N> {
        PolynomialIterator{
            polynomial: self,
            index: if self.len() == 0 {
                0
            } else {
                match self.degree() {
                    Degree::NegInf => self.len(),
                    Degree::Num(x) => self.len() - x - 1
                }
            }
        }
    }
}

// impl<N> Polynomial<N> {
//     /// Returns the value of the `Polynomial` at the given point.
//     /// Intended usage is large polynomials with very few terms.
//     pub fn eval_sparse(&self, point: N) -> N {
//         let mut sum = N::zero();
//         let mut pow = 1;
//         for (val, degree) in self.degree_iter() {
//             sum += val * point.upow(degree);
//             pow += 1;
//         }
//         sum
//     }
// }
//
impl<N> Polynomial<N>
    where N: HasZero + HasOne + Copy + AddAssign + MulAssign + Mul<Output=N> {
    /// Returns the value of the `Polynomial` at the given point.
    pub fn eval(&self, point: N) -> N {
        let mut sum = N::zero();
        let mut pow = N::one();
        for &val in self.terms.iter().rev() {
            sum += pow * val;
            pow *= point;
        }
        sum
    }
}

impl<N> Polynomial<N>
    where N: PartialEq + HasZero + From<u8> + Copy + MulAssign {
    /// Returns the derivative of the `Polynomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Polynomial;
    /// let polynomial = Polynomial::new(vec![4, 1, 5]);
    /// assert_eq!(Polynomial::new(vec![8, 1]), polynomial.derivative());
    /// ```
    pub fn derivative(&self) -> Polynomial<N> {
        let index = first_nonzero_index(&self.terms);
        // TODO: Fix for degrees of arbitrary size.
        let mut degree = (self.len() - index) as u8;
        let mut terms = self.terms[0..self.len()-1].to_vec();
        for term in terms.iter_mut() {
            degree -= 1;
            *term *= N::from(degree);
        }
        Polynomial { terms }
    }
}

impl<N> Integrable<N> for Polynomial<N>
    where N: PartialEq + HasZero + From<u8> + Copy + DivAssign + fmt::Display {
    /// Returns the integral of the `Polynomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, Integrable};
    /// let polynomial = Polynomial::new(vec![1.0, 2.0, 5.0]);
    /// let integral = polynomial.integral();
    /// assert_eq!(Polynomial::new(vec![1.0/3.0, 1.0, 5.0, 0.0]), integral.polynomial);
    /// ```
    fn integral(&self) -> Integral<N> {
        let index = first_nonzero_index(&self.terms);
        // TODO: Fix for degrees of arbitrary size.
        let mut degree = (self.len() - index + 1) as u8;
        let mut terms = self.terms[index..].to_vec();
        for term in terms.iter_mut() {
            degree -= 1;
            *term /= N::from(degree);
        }
        terms.push(N::zero());
        Integral {
            polynomial: Polynomial { terms }
        }
    }
}

impl<N> Polynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + HasOne + PartialEq {
    pub fn borrow_mul(&self, _rhs: &Polynomial<N>) -> Polynomial<N> {
        Polynomial{terms: vec_mul(&self.terms, &_rhs.terms)}
    }

    /// Raises the `Polynomial` to the power of exp, using exponentiation by squaring.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Polynomial;
    /// let polynomial = Polynomial::new(vec![1.0, 2.0]);
    /// let polynomial_sqr = polynomial.pow(2);
    /// let polynomial_cub = polynomial.pow(3);
    /// assert_eq!(polynomial.clone() * polynomial.clone(), polynomial_sqr);
    /// assert_eq!(polynomial_sqr.clone() * polynomial.clone(), polynomial_cub);
    /// ```
    pub fn pow(&self, exp: usize) -> Polynomial<N> {
        if exp == 0 {
            Polynomial{terms: vec![N::one(); 1]}
        } else if exp == 1 {
            Polynomial::new(self.terms.clone())
        } else if exp == 2 {
            self.borrow_mul(self)
        } else if exp % 2 == 0 {
            self.pow(exp / 2).pow(2)
        } else {
            self.borrow_mul(&self.pow(exp - 1))
        }
    }
}

impl<N> Polynomial<N>
    where N: Copy + PartialEq + HasZero + SubAssign + Mul<Output=N> + Div<Output=N> {
    /// Divides self by the given `Polynomial`, and returns the quotient and remainder.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Polynomial;
    /// let polynomial = Polynomial::new(vec![1.0, 2.0]);
    /// let polynomial_sqr = polynomial.pow(2);
    /// let polynomial_cub = polynomial.pow(3);
    /// assert_eq!(polynomial.clone() * polynomial.clone(), polynomial_sqr);
    /// assert_eq!(polynomial_sqr.clone() * polynomial.clone(), polynomial_cub);
    /// ```
    pub fn div_mod(&self, _rhs: &Polynomial<N>) -> Result<(Polynomial<N>, Polynomial<N>), &'static str> {
        fn vec_sub_w_scale<N>(_lhs: &mut Vec<N>, _lhs_degree: usize, _rhs: &Vec<N>, _rhs_deg: usize, _rhs_scale: N)
            where N: Copy + Mul<Output=N> + SubAssign {
            let loc = _lhs.len() - _lhs_degree - 1;
            for (_lhs_t, _rhs_t) in _lhs[loc..].iter_mut().zip(_rhs) {
                *_lhs_t -= (*_rhs_t) * _rhs_scale;
            }
        }

        let (_rhs_deg, _rhs_first) = degree_and_first_val(&_rhs.terms);
        let zero = N::zero();

        if _rhs_deg == 0 {
            match _rhs.terms.last() {
                None => {
                return Err("Can't divide by 0.");
                }
                Some(&x) => {
                    if x == zero {
                        return Err("Can't divide by 0.");
                    }
                }
            }
        }

        let (mut self_degree, mut term) = degree_and_first_val(&self.terms);


        if self_degree < _rhs_deg {
            let zero_vec = vec![zero; 1];
            return Ok((Polynomial::new(zero_vec), Polynomial::new(self.terms.clone())));
        }

        let mut remainder = self.terms.clone();
        let offset = self_degree - _rhs_deg;
        let mut div = vec![zero; offset + 1];

        while self_degree >= _rhs_deg {
            let scale = term / _rhs_first;
            vec_sub_w_scale(&mut remainder, self_degree, &_rhs.terms, _rhs_deg, scale);
            div[offset - (self_degree - _rhs_deg)] = scale;
            let (sd, t) = degree_and_first_val(&remainder);
            self_degree = sd;
            term = t;
        }

        Ok((Polynomial::new(div), Polynomial::new(remainder)))
    }
}

impl<N> PartialEq for Polynomial<N>
    where N: PartialEq + HasZero + Copy {
    /// Returns true if self has the same terms as other.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::Polynomial;
    /// let a = Polynomial::new(vec![1.0, 2.0]);
    /// let b = Polynomial::new(vec![2.0, 2.0]);
    /// let c = Polynomial::new(vec![1.0, 0.0]);
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

impl<N> fmt::Display for Polynomial<N>
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

impl<N> ops::Neg for Polynomial<N>
    where N: PartialEq + HasZero + Copy + Neg<Output=N>{
    type Output = Polynomial<N>;

    fn neg(self) -> Polynomial<N> {
        Polynomial::new(self.terms.iter().map(|&x| -x).collect())
    }
}

impl<N> ops::Sub<Polynomial<N>> for Polynomial<N>
    where N: PartialEq + HasZero + Copy + Sub<Output=N> + SubAssign + Neg<Output=N>{
    type Output = Polynomial<N>;

    fn sub(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        if _rhs.len() > self.len() {
            let mut terms = _rhs.terms.clone();
            let offset = _rhs.len() - self.len();

            for index in terms[..offset].iter_mut() {
                *index = -*index;
            }

            for (index, val) in terms[offset..].iter_mut().zip(self.terms) {
                *index = val - *index;
            }
            Polynomial::new(terms)
        } else {
            let mut terms = self.terms.clone();
            let offset = terms.len() - _rhs.len();
            for (index, val) in terms[offset..].iter_mut().zip(_rhs.terms) {
                *index -= val;
            }
            Polynomial::new(terms)
        }
    }
}

impl<N> ops::SubAssign<Polynomial<N>> for Polynomial<N>
    where N: Neg<Output=N> + Sub<Output=N> + SubAssign + Copy + HasZero + PartialEq {
    fn sub_assign(&mut self, _rhs: Polynomial<N>) {
        if _rhs.len() > self.len() {
            let mut terms = _rhs.terms.clone();
            let offset = _rhs.len() - self.len();

            for index in terms[..offset].iter_mut() {
                *index = -*index;
            }

            for (index, &val) in terms[offset..].iter_mut().zip(&self.terms) {
                *index = val - *index;
            }
            self.terms = terms;
        } else {
            let offset = self.len() - _rhs.len();
            for (index, val) in self.terms[offset..].iter_mut().zip(_rhs.terms) {
                *index -= val;
            }
        }
    }
}

impl<N> ops::Add<Polynomial<N>> for Polynomial<N>
    where N: PartialEq + HasZero + Copy + AddAssign {
    type Output = Polynomial<N>;

    fn add(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        let (mut terms, small) = if _rhs.len() > self.len() {
            (_rhs.terms.clone(), &self.terms)
        } else {
            (self.terms.clone(), &_rhs.terms)
        };

        let offset = terms.len() - small.len();

        for (index, &val) in terms[offset..].iter_mut().zip(small) {
            *index += val;
        }

        Polynomial::new(terms)
    }
}

impl<N: Copy + HasZero + PartialEq + AddAssign> ops::AddAssign<Polynomial<N>> for Polynomial<N> {
    fn add_assign(&mut self, _rhs: Polynomial<N>) {
        if _rhs.len() > self.len() {
            let offset = _rhs.len() - self.len();
            let mut terms = _rhs.terms.clone();
            for (index, &val) in terms[offset..].iter_mut().zip(&self.terms) {
                *index += val;
            }
            self.terms = terms;
        } else {
            let offset = self.len() - _rhs.len();
            for (index, val) in self.terms[offset..].iter_mut().zip(_rhs.terms) {
                *index += val;
            }
        }
    }
}

impl<N> ops::Mul<Polynomial<N>> for Polynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    type Output = Polynomial<N>;

    fn mul(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        Polynomial{terms: vec_mul(&self.terms, &_rhs.terms)}
    }
}

impl<N> ops::MulAssign<Polynomial<N>> for Polynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    fn mul_assign(&mut self, _rhs: Polynomial<N>) {
        self.terms = vec_mul(&self.terms, &_rhs.terms);
    }
}


impl<N> ops::Mul<&Polynomial<N>> for Polynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    type Output = Polynomial<N>;

    fn mul(self, _rhs: &Polynomial<N>) -> Polynomial<N> {
        Polynomial::new(vec_mul(&self.terms, &_rhs.terms))
    }
}

impl<N> ops::MulAssign<&Polynomial<N>> for Polynomial<N>
    where N: Mul<Output=N> + AddAssign + Copy + HasZero + PartialEq {
    fn mul_assign(&mut self, _rhs: &Polynomial<N>) {
        self.terms = vec_mul(&self.terms, &_rhs.terms);
    }
}

impl<N: PartialEq + HasZero + Copy + Mul<Output=N>> ops::Mul<N> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn mul(self, _rhs: N) -> Polynomial<N> {
        Polynomial::new(self.terms.iter().map(|&x| x * _rhs).collect())
    }
}

impl<N: Copy + MulAssign> ops::MulAssign<N> for Polynomial<N> {
    fn mul_assign(&mut self, _rhs: N) {
        for p in self.terms.iter_mut() {
            *p *= _rhs;
        }
    }
}

impl<N> ops::Div<N> for Polynomial<N>
    where N: PartialEq + HasZero + Copy + Div<Output=N> {
    type Output = Polynomial<N>;

    fn div(self, _rhs: N) -> Polynomial<N> {
        Polynomial::new(self.terms.iter().map(|&x| x / _rhs).collect())
    }
}

impl<N> ops::DivAssign<N> for Polynomial<N>
    where N: PartialEq + Copy + DivAssign {
    fn div_assign(&mut self, _rhs: N) {
        for p in self.terms.iter_mut() {
            *p /= _rhs;
        }
    }
}

impl<N: PartialEq + HasZero + Copy> ops::Shl<i32> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn shl(self, _rhs: i32) -> Polynomial<N> {
        if _rhs < 0 {
            self >> -_rhs
        } else {
            let index = first_nonzero_index(&self.terms);
            let mut terms = self.terms[index..].to_vec();
            terms.extend(vec![N::zero(); _rhs as usize]);
            Polynomial{terms}
        }
    }
}

impl<N: HasZero + Copy> ops::ShlAssign<i32> for Polynomial<N> {
    fn shl_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self >>= -_rhs;
        } else {
            self.terms.extend(vec![N::zero(); _rhs as usize]);
        }
    }
}

impl<N: PartialEq + HasZero + Copy> ops::Shr<i32> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn shr(self, _rhs: i32) -> Polynomial<N> {
        if _rhs < 0 {
            self << -_rhs
        } else {
            let index = first_nonzero_index(&self.terms);
            Polynomial{terms: self.terms[index..self.terms.len() - (_rhs as usize)].to_vec()}
        }
    }
}

impl<N: HasZero + Copy> ops::ShrAssign<i32> for Polynomial<N> {
    fn shr_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self <<= -_rhs;
        } else {
            self.terms = self.terms[..self.terms.len() - (_rhs as usize)].to_vec();
        }
    }
}

/// TODO:
/// modulo floordiv
#[cfg(test)]
mod tests {
    use std::fmt::Write;
    use super::Polynomial;
    use ::{Integrable, Degree};

    #[test]
    fn test_eval() {
        let a = Polynomial::new(vec![1, 2, 3]);
        assert_eq!(a.eval(5), 25 + 2 * 5 + 3);
    }

    #[test]
    fn test_derivative() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![2, 2]);
        assert_eq!(a.derivative(), b);

        let a = Polynomial::new(vec![0, 1, 2, 3]);
        assert_eq!(a.derivative(), b);

        let a = Polynomial::new(vec![1, 2, 3, 4]);
        let b = Polynomial::new(vec![3, 4, 3]);
        assert_eq!(a.derivative(), b);

    }

    #[test]
    fn test_integral() {
        let a = Polynomial::new(vec![3, 2, 1]);
        let b = Polynomial::new(vec![1, 1, 1, 0]);
        assert_eq!(a.integral().polynomial, b);
    }

    #[test]
    fn test_integral_eval() {
        let a = Polynomial::new(vec![3, 2, 1]);
        assert_eq!(a.integral().eval(0, 1), 3);
    }

    #[test]
    fn test_integral_const_substitute() {
        let a = Polynomial::new(vec![3, 2, 1]);
        let b = Polynomial::new(vec![1, 1, 1, 5]);
        assert_eq!(a.integral().replace_c(5), b);
    }


    #[test]
    fn test_add_lhs_bigger() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(b + a, c);
    }

    #[test]
    fn test_add_rhs_bigger() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(a + b, c);
    }

    #[test]
    fn test_add_lhs_bigger_assign() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let mut b = Polynomial::new(vec![1, 2, 3, 4]);
        b += a;
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(b, c);
    }

    #[test]
    fn test_add_rhs_bigger_assign() {
        let mut a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        a += b;
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_sub_lhs_bigger() {
        let a = Polynomial::new(vec![2, 3, 4]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 0, 0, 0]);
        assert_eq!(b - a, c);
    }

    #[test]
    fn test_sub_rhs_bigger() {
        let a = Polynomial::new(vec![2, 3, 4]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![-1, 0, 0, 0]);
        assert_eq!(a - b, c);
    }

    #[test]
    fn test_sub_lhs_bigger_assign() {
        let a = Polynomial::new(vec![2, 3, 4]);
        let mut b = Polynomial::new(vec![1, 2, 3, 4]);
        b -= a;
        let c = Polynomial::new(vec![1, 0, 0, 0]);
        assert_eq!(b, c);
    }

    #[test]
    fn test_sub_rhs_bigger_assign() {
        let mut a = Polynomial::new(vec![2, 3, 4]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        a -= b;
        let c = Polynomial::new(vec![-1, 0, 0, 0]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_negate() {
        let a = Polynomial::new(vec![1, 2, 3, 0, -5]);
        let c = Polynomial::new(vec![-1, -2, -3, 0, 5]);
        assert_eq!(-a, c);
    }

    #[test]
    fn test_mul_poly() {
        let a = Polynomial::new(vec![1, 2]);
        let b = a.clone();
        let c = Polynomial::new(vec![1, 4, 4]);
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_mul_assign_poly() {
        let mut a = Polynomial::new(vec![1, 2]);
        let b = a.clone();
        a *= b;
        let c = Polynomial::new(vec![1, 4, 4]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_mul_i32() {
        let a = Polynomial::new(vec![1, 2]);
        let c = Polynomial::new(vec![10, 20]);
        assert_eq!(a * 10, c);
    }

    #[test]
    fn test_mul_assign_i32() {
        let mut a = Polynomial::new(vec![1, 2]);
        a *= 10;
        let c = Polynomial::new(vec![10, 20]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_equality() {
        let a = Polynomial::new(vec![1, 2]);
        let mut c = Polynomial::new(vec![0, 0, 0, 1, 2]);
        c.terms = vec![0, 0, 0, 1, 2];

        assert_eq!(a, c);

        c.terms = vec![1, 2, 0, 0, 0];

        assert_ne!(a, c);
    }

    #[test]
    fn test_equality_first_match() {
        let a = Polynomial::new(vec![1, 2]);
        let b = Polynomial::new(vec![1, 0]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_equality_different() {
        let a = Polynomial::new(vec![1, 2]);
        let b = Polynomial::new(vec![3, 7, 4]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_shl_pos() {
        let a = Polynomial::new(vec![1, 2]);
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a << 5, c);
    }

    #[test]
    fn test_shl_assign_pos() {
        let mut a = Polynomial::new(vec![1, 2]);
        a <<= 5;
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shl_neg() {
        let a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(a << -5, c);
    }

    #[test]
    fn test_shl_assign_neg() {
        let mut a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        a <<= -5;
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shr_pos() {
        let a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(a >> 5, c);
    }

    #[test]
    fn test_shr_assign_pos() {
        let mut a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        a >>= 5;
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shr_neg() {
        let a = Polynomial::new(vec![1, 2]);
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a >> -5, c);
    }

    #[test]
    fn test_shr_assign_neg() {
        let mut a = Polynomial::new(vec![1, 2]);
        a >>= -5;
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_exp() {
        let a = &Polynomial::new(vec![1, 2]);
        let mut b = a.clone();
        assert_eq!(Polynomial::new(vec![1]), a.pow(0));
        for i in 1..10 {
            assert_eq!(b, a.pow(i));
            b *= a;
        }
    }

    #[test]
    fn test_polynomial_str_all_zeroes() {
        let a = Polynomial::new(vec![0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a: Polynomial<i8> = Polynomial::new(vec![]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a = Polynomial::new(vec![0, 0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a: Polynomial<i8> = Polynomial{terms: vec![]};
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a = Polynomial{terms: vec![0]};
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a = Polynomial{terms: vec![0, 0]};
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

    }

    #[test]
    fn test_polynomial_str() {
        let a = Polynomial::new(vec![-1, -2, 3]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^2 - 2x + 3");
    }

    #[test]
    fn test_polynomial_str_has_zeroes() {
        let a = Polynomial::new(vec![-1, -2, 0, 0, 3]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^4 - 2x^3 + 3");
    }

    #[test]
    fn test_polynomial_str_has_ones() {
        let a = Polynomial::new(vec![-1, -1, -1, 0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^3 - x^2 - x");
    }

    #[test]
    fn test_polynomial_str_has_negative() {
        let a = Polynomial::new(vec![-2, -1, -1, 0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-2x^3 - x^2 - x");
    }

    #[test]
    fn test_integral_str() {
        let a = Polynomial::new(vec![-3, -2, 1]).integral();
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^3 - x^2 + x + C");
    }

    #[test]
    fn test_degree() {
        let a = Polynomial::new(vec![0, 0, 0, -1, -2, 3]);
        assert_eq!(Degree::Num(2), a.degree());
    }

    #[test]
    fn test_iter() {
        let mut num_iters = 0;
        let a = Polynomial::new(vec![0, 0, 0, -1, -2, 3]);
        let b=  vec![-1, -2, 3];
        for (a_val, b_val) in a.iter().zip(b) {
            num_iters += 1;
            assert_eq!(a_val, b_val);
        }

        assert_eq!(num_iters, 3);
    }

}
