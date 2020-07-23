use std::collections::HashMap;
use std::fmt;
use std::fmt::{Display, Write};
use std::ops;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Shl, ShlAssign, Shr,
    ShrAssign, Sub, SubAssign,
};
use std::str::FromStr;

use num::{One, Zero};

use rustnomial::degree::TermTokenizer;
use rustnomial::numerics::{Abs, IsNegativeOne, IsPositive, PowUsize};
use rustnomial::strings::{write_leading_term, write_trailing_term};
use rustnomial::traits::TermIterator;
use {Degree, Derivable, Evaluable, GenericPolynomial, Polynomial, Term};

#[derive(Debug, Clone)]
pub struct SparsePolynomial<N> {
    pub terms: HashMap<usize, N>,
}

fn map_mul<N>(_lhs: &HashMap<usize, N>, _rhs: &HashMap<usize, N>) -> HashMap<usize, N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    let mut terms = HashMap::new();
    for (&rhs_deg, &rterm) in _rhs {
        if rterm.is_zero() {
            continue;
        }

        for (&lhs_deg, &lterm) in _lhs.iter() {
            match terms.get_mut(&(rhs_deg + lhs_deg)) {
                None => {
                    terms.insert(rhs_deg + lhs_deg, rterm * lterm);
                }
                Some(val) => {
                    *val += rterm * lterm;
                }
            }
        }
    }
    terms
}

fn map_sub_w_scale<N>(_lhs: &mut HashMap<usize, N>, _rhs: &HashMap<usize, N>, _rhs_scale: N)
where
    N: Copy + Neg<Output = N> + Sub<Output = N> + Mul<Output = N> + SubAssign,
{
    for (rdeg, &rcoeff) in _rhs.iter() {
        match _lhs.get_mut(rdeg) {
            None => {
                _lhs.insert(*rdeg, -rcoeff * _rhs_scale);
            }
            Some(lcoeff) => {
                *lcoeff -= rcoeff * _rhs_scale;
            }
        }
    }
}

fn degree<N: Zero + Copy>(terms: &HashMap<usize, N>) -> Degree {
    let mut term_iter = terms.iter();
    let (mut max_term, mut max_degree) = match term_iter.next() {
        None => {
            return Degree::NegInf;
        }
        Some((&degree, &coeff)) => (coeff, degree),
    };

    for (&degree, &coeff) in term_iter {
        if degree > max_degree && !coeff.is_zero() {
            max_term = coeff;
            max_degree = degree;
        }
    }

    if max_term.is_zero() {
        Degree::NegInf
    } else {
        Degree::Num(max_degree)
    }
}

fn first_term<N: Zero + Copy>(terms: &HashMap<usize, N>) -> Term<N> {
    let degree = match degree(terms) {
        Degree::NegInf => {
            return Term::ZeroTerm;
        }
        Degree::Num(x) => x,
    };
    match terms.get(&degree) {
        None => Term::ZeroTerm,
        Some(&val) => Term::new(val, degree),
    }
}

impl<N: Zero + Copy> GenericPolynomial<N> for SparsePolynomial<N> {
    fn len(&self) -> usize {
        match self.degree() {
            Degree::NegInf => 0,
            Degree::Num(deg) => deg + 1,
        }
    }

    fn nth_term(&self, index: usize) -> Term<N> {
        let degree = match self.degree() {
            Degree::NegInf => {
                return Term::ZeroTerm;
            }
            Degree::Num(x) => x - index,
        };

        match self.terms.get(&degree) {
            None => Term::ZeroTerm,
            Some(&val) => Term::new(val, degree),
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
    /// let mut iter = polynomial.term_iter();
    /// assert_eq!(Some((1, 3)), iter.next());
    /// assert_eq!(Some((2, 1)), iter.next());
    /// assert_eq!(Some((3, 0)), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn term_iter(&self) -> TermIterator<N> {
        TermIterator::new(self)
    }
}

impl<N> FromStr for SparsePolynomial<N>
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
    /// use rustnomial::SparsePolynomial;
    /// use std::str::FromStr;
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = SparsePolynomial::from_str("5x^2 + 11x + 2").unwrap();
    /// assert_eq!(SparsePolynomial::from_vec(vec![5, 11, 2]), polynomial);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut polynomial = SparsePolynomial::zero();
        let mut has_iterated = false;
        for term in TermTokenizer::new(s).map(|s| Term::from_str(s.as_str())) {
            has_iterated = true;
            match term {
                Err(msg) => return Err(msg),
                Ok(Term::ZeroTerm) => {}
                Ok(Term::Term(coeff, deg)) => {
                    polynomial.add_term(coeff, deg);
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

impl<N> SparsePolynomial<N>
where
    N: Zero + Copy + AddAssign,
{
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
        if term.is_zero() {
            return;
        }

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

impl<N> SparsePolynomial<N> {
    pub fn zero() -> SparsePolynomial<N> {
        SparsePolynomial {
            terms: HashMap::new(),
        }
    }

    pub fn new(terms: HashMap<usize, N>) -> SparsePolynomial<N> {
        SparsePolynomial { terms }
    }
}

impl<N> SparsePolynomial<N>
where
    N: Zero + Copy,
{
    /// Returns a `SparsePolynomial` with the corresponding terms,
    /// in order of ax^n + bx^(n-1) + ... + cx + d
    ///
    /// # Arguments
    ///
    /// * ` term_vec ` - A vector of constants, in decreasing order of degree.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::SparsePolynomial;
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = SparsePolynomial::from_vec(vec![1.0, 4.0, 4.0]);
    /// ```
    pub fn from_vec(term_vec: Vec<N>) -> SparsePolynomial<N> {
        let mut terms: HashMap<usize, N> = HashMap::new();
        if term_vec.len() != 0 {
            let degree = term_vec.len() - 1;
            for (index, &val) in term_vec.iter().enumerate() {
                if !val.is_zero() {
                    terms.insert(degree - index, val);
                }
            }
        }
        SparsePolynomial { terms }
    }

    /// Reduces the size of the `Polynomial` in memory any terms are zero.
    pub fn trim(&mut self) {
        let mut new_map = HashMap::new();
        for (&degree, &coeff) in self.terms.iter() {
            if !coeff.is_zero() {
                new_map.insert(degree, coeff);
            }
        }
        self.terms = new_map;
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
}

impl<N> Evaluable<N> for SparsePolynomial<N>
where
    N: Zero + PowUsize + Copy + AddAssign + Mul<Output = N>,
{
    fn eval(&self, point: N) -> N {
        let mut sum = N::zero();
        for (&degree, &val) in self.terms.iter() {
            sum += val * point.upow(degree);
        }
        sum
    }
}

impl<N> Derivable<N> for SparsePolynomial<N>
where
    N: Zero + From<u8> + Copy + Mul<Output = N>,
{
    /// Returns the derivative of the `Polynomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SparsePolynomial, Derivable};
    /// let polynomial = SparsePolynomial::from_vec(vec![4, 1, 5]);
    /// assert_eq!(SparsePolynomial::from_vec(vec![8, 1]), polynomial.derivative());
    /// ```
    fn derivative(&self) -> SparsePolynomial<N> {
        let mut terms = HashMap::with_capacity(self.terms.len());
        // TODO: Fix for degrees of arbitrary size.
        for (&degree, &coeff) in self.terms.iter() {
            if degree != 0 && !coeff.is_zero() {
                terms.insert(degree - 1, coeff * N::from(degree as u8));
            }
        }
        SparsePolynomial { terms }
    }
}

// TODO: Make Integral generic over Polynomial, SparsePolynomial
// impl<N> Integrable<N> for SparsePolynomial<N>
//     where N: PartialEq + Zero + From<u8> + Copy + DivAssign + fmt::Display {
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
where
    N: Mul<Output = N> + AddAssign + Copy + Zero + One,
{
    pub fn borrow_mul(&self, _rhs: &SparsePolynomial<N>) -> SparsePolynomial<N> {
        SparsePolynomial {
            terms: map_mul(&self.terms, &_rhs.terms),
        }
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
                },
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

impl<N> SparsePolynomial<N>
where
    N: Copy
        + Zero
        + Neg<Output = N>
        + Sub<Output = N>
        + SubAssign
        + Mul<Output = N>
        + Div<Output = N>
        + AddAssign,
{
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
    pub fn div_mod(
        &self,
        _rhs: &SparsePolynomial<N>,
    ) -> (SparsePolynomial<N>, SparsePolynomial<N>) {
        let (_rhs_first, _rhs_deg) = match first_term(&_rhs.terms) {
            Term::ZeroTerm => {
                panic!("Can't divide by 0.");
            }
            Term::Term(coeff, deg) => (coeff, deg),
        };

        let (mut scale, mut self_degree) = match first_term(&self.terms) {
            Term::ZeroTerm => {
                return (
                    SparsePolynomial::zero(),
                    SparsePolynomial::new(self.terms.clone()),
                );
            }
            Term::Term(term, degree) => {
                if degree < _rhs_deg {
                    return (
                        SparsePolynomial::zero(),
                        SparsePolynomial::new(self.terms.clone()),
                    );
                }
                (term / _rhs_first, degree)
            }
        };

        let mut remainder = self.terms.clone();
        let offset = self_degree - _rhs_deg;
        let mut div = SparsePolynomial::from_vec(Vec::new());

        while self_degree >= _rhs_deg {
            map_sub_w_scale(&mut remainder, &_rhs.terms, scale);
            div.add_term(scale, offset);
            match first_term(&remainder) {
                Term::ZeroTerm => {
                    break;
                }
                Term::Term(coeff, degree) => {
                    scale = coeff / _rhs_first;
                    self_degree = degree;
                }
            }
        }

        (div, SparsePolynomial::new(remainder))
    }
}

impl<N> Rem<SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Copy
        + Zero
        + Neg<Output = N>
        + Sub<Output = N>
        + SubAssign
        + Mul<Output = N>
        + Div<Output = N>
        + AddAssign,
{
    type Output = SparsePolynomial<N>;
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
    fn rem(self, _rhs: SparsePolynomial<N>) -> SparsePolynomial<N> {
        let (_rhs_first, _rhs_deg) = match first_term(&_rhs.terms) {
            Term::ZeroTerm => {
                panic!("Can't divide by 0.");
            }
            Term::Term(coeff, deg) => (coeff, deg),
        };

        let (mut scale, mut self_degree) = match first_term(&self.terms) {
            Term::ZeroTerm => {
                return SparsePolynomial::new(self.terms.clone());
            }
            Term::Term(term, degree) => {
                if degree < _rhs_deg {
                    return SparsePolynomial::new(self.terms.clone());
                }
                (term / _rhs_first, degree)
            }
        };

        let mut remainder = self.terms.clone();

        while self_degree >= _rhs_deg {
            map_sub_w_scale(&mut remainder, &_rhs.terms, scale);
            match first_term(&remainder) {
                Term::ZeroTerm => {
                    break;
                }
                Term::Term(coeff, degree) => {
                    scale = coeff / _rhs_first;
                    self_degree = degree;
                }
            }
        }

        SparsePolynomial::new(remainder)
    }
}

impl<N> RemAssign<SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Copy
        + Zero
        + Neg<Output = N>
        + Sub<Output = N>
        + SubAssign
        + Mul<Output = N>
        + Div<Output = N>
        + AddAssign,
{
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
    fn rem_assign(&mut self, _rhs: SparsePolynomial<N>) {
        let (_rhs_first, _rhs_deg) = match first_term(&_rhs.terms) {
            Term::ZeroTerm => {
                panic!("Can't divide by 0.");
            }
            Term::Term(coeff, deg) => (coeff, deg),
        };

        let (mut scale, mut self_degree) = match first_term(&self.terms) {
            Term::ZeroTerm => {
                return;
            }
            Term::Term(coeff, degree) => {
                if degree < _rhs_deg {
                    return;
                }
                (coeff / _rhs_first, degree)
            }
        };

        while self_degree >= _rhs_deg {
            map_sub_w_scale(&mut self.terms, &_rhs.terms, scale);
            match first_term(&self.terms) {
                Term::ZeroTerm => {
                    break;
                }
                Term::Term(coeff, degree) => {
                    scale = coeff / _rhs_first;
                    self_degree = degree;
                }
            }
        }
    }
}

impl<N> PartialEq for SparsePolynomial<N>
where
    N: Zero + PartialEq + Copy,
{
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
        self.terms.iter().all(|(key, value)| {
            other
                .terms
                .get(key)
                .map_or(value.is_zero(), |v| *value == *v)
        }) && other.terms.iter().all(|(key, value)| {
            self.terms
                .get(key)
                .map_or(value.is_zero(), |v| *value == *v)
        })
    }
}

impl<N> fmt::Display for SparsePolynomial<N>
where
    N: Zero + One + IsPositive + Copy + IsNegativeOne + PartialEq + Display + Abs,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut iter = self.term_iter();
        if let Some((coeff, degree)) = iter.next() {
            write_leading_term(f, coeff, degree);
            for (coeff, degree) in iter {
                write_trailing_term(f, coeff, degree);
            }
            write!(f, "")
        } else {
            write!(f, "0")
        }
    }
}

impl<N> Neg for SparsePolynomial<N>
where
    N: Zero + Copy + Neg<Output = N>,
{
    type Output = SparsePolynomial<N>;

    fn neg(self) -> SparsePolynomial<N> {
        let mut terms = HashMap::new();
        for (&deg, &coeff) in self.terms.iter() {
            terms.insert(deg, -coeff);
        }
        SparsePolynomial::new(terms)
    }
}

impl<N> Sub<SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Zero + Copy + Sub<Output = N> + SubAssign + Neg<Output = N>,
{
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
        SparsePolynomial::new(terms)
    }
}

impl<N> Sub<Polynomial<N>> for SparsePolynomial<N>
where
    N: Zero + Copy + Sub<Output = N> + SubAssign + Neg<Output = N>,
{
    type Output = SparsePolynomial<N>;

    fn sub(self, _rhs: Polynomial<N>) -> SparsePolynomial<N> {
        let mut terms = self.terms.clone();
        for (coeff, deg) in _rhs.term_iter() {
            match terms.get_mut(&deg) {
                None => {
                    terms.insert(deg, -coeff);
                }
                Some(val) => {
                    *val -= coeff;
                }
            }
        }
        SparsePolynomial::new(terms)
    }
}

impl<N> SubAssign<SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Neg<Output = N> + Sub<Output = N> + SubAssign + Copy,
{
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

impl<N> Add<SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Copy + AddAssign,
{
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
        SparsePolynomial { terms }
    }
}

impl<N: Copy + AddAssign> AddAssign<SparsePolynomial<N>> for SparsePolynomial<N> {
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

impl<N> Mul<SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    type Output = SparsePolynomial<N>;

    fn mul(self, _rhs: SparsePolynomial<N>) -> SparsePolynomial<N> {
        SparsePolynomial {
            terms: map_mul(&self.terms, &_rhs.terms),
        }
    }
}

impl<N> MulAssign<SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    fn mul_assign(&mut self, _rhs: SparsePolynomial<N>) {
        self.terms = map_mul(&self.terms, &_rhs.terms);
    }
}

impl<N> Mul<&SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    type Output = SparsePolynomial<N>;

    fn mul(self, _rhs: &SparsePolynomial<N>) -> SparsePolynomial<N> {
        SparsePolynomial::new(map_mul(&self.terms, &_rhs.terms))
    }
}

impl<N> MulAssign<&SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    fn mul_assign(&mut self, _rhs: &SparsePolynomial<N>) {
        self.terms = map_mul(&self.terms, &_rhs.terms);
    }
}

impl<N: Copy + Mul<Output = N>> Mul<N> for SparsePolynomial<N> {
    type Output = SparsePolynomial<N>;

    fn mul(self, _rhs: N) -> SparsePolynomial<N> {
        let mut terms = HashMap::new();
        for (&deg, &coeff) in self.terms.iter() {
            terms.insert(deg, coeff * _rhs);
        }

        SparsePolynomial::new(terms)
    }
}

impl<N: Copy + MulAssign> MulAssign<N> for SparsePolynomial<N> {
    fn mul_assign(&mut self, _rhs: N) {
        for (_, coeff) in self.terms.iter_mut() {
            *coeff *= _rhs;
        }
    }
}

impl<N> Div<N> for SparsePolynomial<N>
where
    N: Copy + Div<Output = N>,
{
    type Output = SparsePolynomial<N>;

    fn div(self, _rhs: N) -> SparsePolynomial<N> {
        let mut terms = HashMap::new();
        for (&deg, &coeff) in self.terms.iter() {
            terms.insert(deg, coeff / _rhs);
        }

        SparsePolynomial::new(terms)
    }
}

impl<N> DivAssign<N> for SparsePolynomial<N>
where
    N: Copy + DivAssign,
{
    fn div_assign(&mut self, _rhs: N) {
        for (_, coeff) in self.terms.iter_mut() {
            *coeff /= _rhs;
        }
    }
}

impl<N: Copy> Shl<i32> for SparsePolynomial<N> {
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

impl<N: Copy> ShlAssign<i32> for SparsePolynomial<N> {
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

impl<N: Copy> Shr<i32> for SparsePolynomial<N> {
    type Output = SparsePolynomial<N>;

    fn shr(self, _rhs: i32) -> SparsePolynomial<N> {
        if _rhs < 0 {
            self << -_rhs
        } else {
            let mut terms = HashMap::new();
            let _rhs = _rhs as usize;
            for (&deg, &coeff) in self.terms.iter() {
                if deg >= _rhs {
                    terms.insert(deg - _rhs, coeff);
                }
            }
            SparsePolynomial::new(terms)
        }
    }
}

impl<N: Copy> ShrAssign<i32> for SparsePolynomial<N> {
    fn shr_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self <<= -_rhs;
        } else {
            let mut terms = HashMap::new();
            let _rhs = _rhs as usize;
            for (&deg, &coeff) in self.terms.iter() {
                if deg >= _rhs {
                    terms.insert(deg - _rhs, coeff);
                }
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
    use std::str::FromStr;
    use {Degree, Derivable, Evaluable, Polynomial, SparsePolynomial};

    #[test]
    fn test_from_str() {
        match SparsePolynomial::<i32>::from_str("5x^2") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![5, 0, 0]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("255x^2+15x+3") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![255, 15, 3]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("255x^2-15x^1+3x^0") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![255, -15, 3]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("-x^1") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![-1, 0]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("5+x") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![1, 5]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("5x+11x") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![16, 0]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        assert!(
            SparsePolynomial::<i32>::from_str("5+x^").is_err(),
            "Should err on dangling ^"
        );
    }

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
    fn test_shr_to_zero() {
        let a = SparsePolynomial::from_vec(vec![1, 2]);
        assert_eq!(a >> 5, SparsePolynomial::zero());
    }

    #[test]
    fn test_shr_assign_to_zero() {
        let mut a = SparsePolynomial::from_vec(vec![1, 2]);
        a >>= 5;
        assert_eq!(a, SparsePolynomial::zero());
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
        assert_eq!(a.to_string(), "0");

        let a: SparsePolynomial<i8> = SparsePolynomial::from_vec(vec![]);
        assert_eq!(a.to_string(), "0");

        let a = SparsePolynomial::from_vec(vec![0, 0]);
        assert_eq!(a.to_string(), "0");
    }

    #[test]
    fn test_polynomial_str() {
        let a = SparsePolynomial::from_vec(vec![-1, -2, 3]);
        assert_eq!(a.to_string(), "-x^2 - 2x + 3");
    }

    #[test]
    fn test_polynomial_str_has_zeroes() {
        let a = SparsePolynomial::from_vec(vec![-1, -2, 0, 0, 3]);
        assert_eq!(a.to_string(), "-x^4 - 2x^3 + 3");
    }

    #[test]
    fn test_polynomial_str_has_ones() {
        let a = SparsePolynomial::from_vec(vec![-1, -1, -1, 0]);
        assert_eq!(a.to_string(), "-x^3 - x^2 - x");
    }

    #[test]
    fn test_polynomial_str_has_negative() {
        let a = SparsePolynomial::from_vec(vec![-2, -1, -1, 0]);
        assert_eq!(a.to_string(), "-2x^3 - x^2 - x");
    }

    // #[test]
    // fn test_integral_str() {
    //     let a = SparsePolynomial::from_vec(vec![-3, -2, 1]).integral();
    //     assert_eq!(a.to_string(), "-x^3 - x^2 + x + C");
    // }

    #[test]
    fn test_degree() {
        let a = SparsePolynomial::from_vec(vec![0, 0, 0, -1, -2, 3]);
        assert_eq!(Degree::Num(2), a.degree());
    }

    #[test]
    fn test_generic_sub() {
        let a = SparsePolynomial::from_vec(vec![0, 0, 0, -1, -2, 3]);
        let b = Polynomial::new(vec![-1, -2, 3]);
        let c = a - b;
        assert_eq!(SparsePolynomial::from_vec(vec![0]), c);
    }

    #[test]
    fn test_pow() {
        let vec = vec![1u32, 2, 3, 4, 5];
        let a = SparsePolynomial::from_vec(vec.clone());
        let b = Polynomial::new(vec.clone());
        let a = a.pow(8);
        let b = SparsePolynomial::from_vec(b.pow(8).terms);
        assert_eq!(a, b);
    }
}
