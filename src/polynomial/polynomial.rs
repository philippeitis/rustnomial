use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Shl, ShlAssign, Shr,
    ShrAssign, Sub, SubAssign,
};

use num::{One, Zero};

use crate::numerics::{Abs, CanNegate, IsNegativeOne, IsPositive};
use crate::polynomial::find_roots::{find_roots, Roots};
use crate::{
    Degree, Derivable, Evaluable, FreeSizePolynomial, Integrable, Integral, MutablePolynomial,
    SizedPolynomial, Term, TryAddError,
};

#[macro_export]
macro_rules! polynomial {
    ( $( $x:expr ),* ) => {
        {
            use $crate::Polynomial;
            Polynomial::new(vec![$($x,)*])
        }
    };
}

#[derive(Debug, Clone)]
pub struct Polynomial<N> {
    pub terms: Vec<N>,
}

pub(crate) fn first_nonzero_index<N>(terms: &[N]) -> usize
where
    N: Zero + Copy,
{
    for (degree, chunk) in terms.chunks_exact(4).enumerate() {
        for (index, &val) in chunk.iter().enumerate() {
            if !val.is_zero() {
                return degree * 4 + index;
            }
        }
    }

    let mut len = terms.chunks_exact(4).len() * 4;
    for &value in terms.chunks_exact(4).remainder().iter() {
        if !value.is_zero() {
            return len;
        }
        len += 1;
    }

    len
}

fn slice_mul<N>(_lhs: &[N], _rhs: &[N]) -> Vec<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    let _rhs = &_rhs[first_nonzero_index(&_rhs)..];
    let _lhs = &_lhs[first_nonzero_index(&_lhs)..];
    let mut terms = vec![N::zero(); _rhs.len() + _lhs.len() - 1];
    for (index, &rterm) in _rhs.iter().enumerate() {
        if rterm.is_zero() {
            continue;
        }
        for (&lterm, term) in _lhs.iter().zip(terms[index..].iter_mut()) {
            *term += rterm * lterm;
        }
    }
    terms
}

fn vec_sub_w_scale<N>(
    _lhs: &mut [N],
    _lhs_degree: usize,
    _rhs: &[N],
    _rhs_deg: usize,
    _rhs_scale: N,
) where
    N: Copy + Mul<Output = N> + SubAssign,
{
    let loc = _lhs.len() - _lhs_degree - 1;
    for (_lhs_t, _rhs_t) in _lhs[loc..]
        .iter_mut()
        .zip(_rhs[_rhs.len() - _rhs_deg - 1..].iter())
    {
        *_lhs_t -= (*_rhs_t) * _rhs_scale;
    }
}

fn degree<N>(poly_vec: &[N]) -> Degree
where
    N: Zero + Copy,
{
    let index = first_nonzero_index(poly_vec);
    if index == poly_vec.len() {
        Degree::NegInf
    } else {
        Degree::Num(poly_vec.len() - index - 1)
    }
}

pub(crate) fn first_term<N>(poly_vec: &[N]) -> Term<N>
where
    N: Zero + Copy,
{
    for (degree, chunk) in poly_vec.chunks_exact(4).enumerate() {
        for (index, &value) in chunk.iter().enumerate() {
            if !value.is_zero() {
                return Term::Term(value, poly_vec.len() - degree * 4 - index - 1);
            }
        }
    }

    let mut index = poly_vec.chunks_exact(4).len() * 4;
    for &value in poly_vec.chunks_exact(4).remainder().iter() {
        if !value.is_zero() {
            return Term::Term(value, poly_vec.len() - index - 1);
        }
        index += 1;
    }

    Term::ZeroTerm
}

pub(crate) fn term_with_deg<N: Zero + Copy>(terms: &[N], degree: usize) -> Term<N> {
    if degree < terms.len() {
        Term::new(terms[terms.len() - degree - 1], degree)
    } else {
        Term::ZeroTerm
    }
}

impl<N> Polynomial<N>
where
    N: Zero + Copy,
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
            },
        }
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
}

impl<N: Copy + Zero> SizedPolynomial<N> for Polynomial<N> {
    /// Returns the length of the `Polynomial`. Not equal to the number of terms.
    fn len(&self) -> usize {
        self.terms.len()
    }

    fn term_with_degree(&self, degree: usize) -> Term<N> {
        term_with_deg(&self.terms, degree)
    }

    /// Returns the degree of the `Polynomial` it is called on, corresponding to the
    /// largest non-zero term.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, Polynomial, Degree};
    /// let polynomial = Polynomial::new(vec![1.0, 4.0, 4.0]);
    /// assert_eq!(Degree::Num(2), polynomial.degree());
    /// ```
    fn degree(&self) -> Degree {
        degree(&self.terms)
    }

    /// Returns a `Polynomial` with no terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, Polynomial};
    /// let zero = Polynomial::<i32>::zero();
    /// assert!(zero.is_zero());
    /// assert!(zero.term_iter().next().is_none());
    /// assert!(zero.terms.is_empty());
    /// ```
    fn zero() -> Polynomial<N> {
        Polynomial { terms: vec![] }
    }

    /// Sets self to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, SizedPolynomial};
    /// let mut non_zero = Polynomial::from(vec![0, 1]);
    /// assert!(!non_zero.is_zero());
    /// non_zero.set_to_zero();
    /// assert!(non_zero.is_zero());
    /// ```
    fn set_to_zero(&mut self) {
        self.terms.clear()
    }
}

impl<N> MutablePolynomial<N> for Polynomial<N>
where
    N: Zero + Copy + AddAssign + SubAssign + CanNegate,
{
    fn try_add_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        Ok(self.add_term(coeff, degree))
    }

    fn try_sub_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        if self.len() < degree + 1 {
            if !N::can_negate() {
                return Err(TryAddError::CanNotNegate);
            }
            let added_zeros = degree + 1 - self.terms.len();
            self.terms
                .splice(0..0, std::iter::repeat(N::zero()).take(added_zeros));
        }
        let index = self.len() - degree - 1;
        self.terms[index] -= coeff;

        Ok(())
    }
}

impl Polynomial<f64> {
    /// Return the roots of the `Polynomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, Roots, SizedPolynomial};
    /// let zero = Polynomial::<f64>::zero();
    /// assert_eq!(Roots::InfiniteRoots, zero.roots());
    /// let constant = Polynomial::new(vec![1.]);
    /// assert_eq!(Roots::NoRoots, constant.roots());
    /// let monomial = Polynomial::new(vec![1.0, 0.,]);
    /// assert_eq!(Roots::ManyRealRoots(vec![0.]), monomial.roots());
    /// let binomial = Polynomial::new(vec![1.0, 2.0]);
    /// assert_eq!(Roots::ManyRealRoots(vec![-2.0]), binomial.roots());
    /// let trinomial = Polynomial::new(vec![1.0, 4.0, 4.0]);
    /// assert_eq!(Roots::ManyRealRoots(vec![-2.0, -2.0]), trinomial.roots());
    /// let quadnomial = Polynomial::new(vec![1.0, 6.0, 12.0, 8.0]);
    /// assert_eq!(Roots::ManyRealRoots(vec![-2.0, -2.0, -2.0]), quadnomial.roots());
    /// ```
    pub fn roots(self) -> Roots<f64> {
        find_roots(&self)
    }
}

impl<N> FreeSizePolynomial<N> for Polynomial<N>
where
    N: Zero + Copy + AddAssign,
{
    /// Returns a `Polynomial` with the corresponding terms,
    /// in order of ax^n + bx^(n-1) + ... + cx + d
    ///
    /// # Arguments
    ///
    /// * ` terms ` - A slice of (coefficient, degree) pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{FreeSizePolynomial, Polynomial};
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = Polynomial::from_terms(&[(1.0, 2), (4.0, 1), (4.0, 0)]);
    /// assert_eq!(Polynomial::new(vec![1., 4., 4.]), polynomial);
    /// ```
    fn from_terms(terms: &[(N, usize)]) -> Self {
        let mut a = Polynomial::zero();
        for &(term, degree) in terms {
            a.add_term(term, degree);
        }
        a
    }

    fn add_term(&mut self, coeff: N, degree: usize) {
        if self.len() < degree + 1 {
            let added_zeros = degree + 1 - self.terms.len();
            self.terms
                .splice(0..0, std::iter::repeat(N::zero()).take(added_zeros));
        }
        let index = self.len() - degree - 1;
        self.terms[index] += coeff;
    }
}

impl<N> Evaluable<N> for Polynomial<N>
where
    N: Zero + Copy + AddAssign + MulAssign + Mul<Output = N>,
{
    /// Returns the value of the `Polynomial` at the given point.
    ///
    /// # Example
    ///
    /// ```
    ///
    /// ```
    fn eval(&self, point: N) -> N {
        if let Some((&last, first)) = self.terms.split_last() {
            if point.is_zero() {
                return last;
            }

            // Equivalent to
            // sum = (a(x^n) + b(x^(n-1)) + ... + z
            // but instead of having explicit x^n, we do
            // sum = (((ax) + b)x + ..)x + z
            // which allows the same result with
            // the same number of adds, and
            // n - 1 multiplies vs 2n muls.
            let mut sum = N::zero();
            for &val in first.iter() {
                sum += val;
                sum *= point;
            }
            sum + last
        } else {
            N::zero()
        }
    }
}

impl<N> Derivable<N> for Polynomial<N>
where
    N: Zero + From<u8> + Copy + MulAssign,
{
    /// Returns the derivative of the `Polynomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, Derivable};
    /// let polynomial = Polynomial::new(vec![4, 1, 5]);
    /// assert_eq!(Polynomial::new(vec![8, 1]), polynomial.derivative());
    /// ```
    fn derivative(&self) -> Polynomial<N> {
        let index = first_nonzero_index(&self.terms);
        // TODO: Fix for degrees of arbitrary size.
        let mut degree = (self.len() - index) as u8;
        let mut terms = self.terms[0..self.len() - 1].to_vec();
        for term in terms.iter_mut() {
            degree -= 1;
            *term *= N::from(degree);
        }
        Polynomial { terms }
    }
}

impl<N> Integrable<N> for Polynomial<N>
where
    N: Zero + From<u8> + Copy + DivAssign,
{
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
            polynomial: Polynomial { terms },
        }
    }
}

impl<N> Polynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero + One,
{
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
            Polynomial {
                terms: vec![N::one(); 1],
            }
        } else if exp == 1 {
            Polynomial::new(self.terms.clone())
        } else if exp == 2 {
            self * self
        } else if exp % 2 == 0 {
            self.pow(exp / 2).pow(2)
        } else {
            self * &self.pow(exp - 1)
        }
    }
}

impl<N> Polynomial<N>
where
    N: Copy + Zero + SubAssign + Mul<Output = N> + Div<Output = N>,
{
    /// Divides self by the given `Polynomial`, and returns the quotient and remainder.
    pub fn div_mod(&self, _rhs: &Polynomial<N>) -> (Polynomial<N>, Polynomial<N>) {
        let zero = N::zero();

        let (_rhs_first, _rhs_deg) = match first_term(&_rhs.terms) {
            Term::ZeroTerm => panic!("Can't divide polynomial by 0."),
            Term::Term(coeff, deg) => (coeff, deg),
        };

        let (mut coeff, mut self_degree) = match first_term(&self.terms) {
            Term::ZeroTerm => {
                return (Polynomial::zero(), self.clone());
            }
            Term::Term(coeff, degree) => {
                if degree < _rhs_deg {
                    return (Polynomial::zero(), self.clone());
                }
                (coeff, degree)
            }
        };

        let mut remainder = self.terms.clone();
        let mut div = vec![zero; self_degree - _rhs_deg + 1];
        let offset = self_degree;

        while self_degree >= _rhs_deg {
            let scale = coeff / _rhs_first;
            vec_sub_w_scale(&mut remainder, self_degree, &_rhs.terms, _rhs_deg, scale);
            div[offset - self_degree] = scale;
            match first_term(&remainder) {
                Term::ZeroTerm => break,
                Term::Term(coeffx, degree) => {
                    coeff = coeffx;
                    self_degree = degree;
                }
            }
        }

        (Polynomial::new(div), Polynomial::new(remainder))
    }
}

impl<N> Polynomial<N>
where
    N: Copy + Zero + SubAssign + Mul<Output = N> + Div<Output = N>,
{
    /// Divides self by the given `Polynomial`, and returns the quotient.
    pub fn floor_div(&self, _rhs: &Polynomial<N>) -> Polynomial<N> {
        self.div_mod(_rhs).0
    }
}

impl<N> Rem<Polynomial<N>> for Polynomial<N>
where
    N: Copy + Zero + SubAssign + Mul<Output = N> + Div<Output = N>,
{
    type Output = Polynomial<N>;

    /// Returns the remainder of dividing `self` by `_rhs`.
    fn rem(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        let (_rhs_first, _rhs_deg) = match first_term(&_rhs.terms) {
            Term::ZeroTerm => panic!("Can't divide polynomial by 0."),
            Term::Term(coeff, deg) => (coeff, deg),
        };

        let (mut scale, mut self_degree) = match first_term(&self.terms) {
            Term::ZeroTerm => return self.clone(),
            Term::Term(coeff, degree) => {
                if degree < _rhs_deg {
                    return self.clone();
                }
                (coeff / _rhs_first, degree)
            }
        };

        let mut remainder = self.terms.clone();

        while self_degree >= _rhs_deg {
            vec_sub_w_scale(&mut remainder, self_degree, &_rhs.terms, _rhs_deg, scale);
            match first_term(&self.terms) {
                Term::ZeroTerm => break,
                Term::Term(coeff, degree) => {
                    scale = coeff / _rhs_first;
                    self_degree = degree;
                }
            }
        }

        Polynomial::new(remainder)
    }
}

impl<N> RemAssign<Polynomial<N>> for Polynomial<N>
where
    N: Copy + Zero + SubAssign + Mul<Output = N> + Div<Output = N>,
{
    /// Assign the remainder of dividing `self` by `_rhs` to `self`.
    fn rem_assign(&mut self, _rhs: Polynomial<N>) {
        let (_rhs_first, _rhs_deg) = match first_term(&_rhs.terms) {
            Term::ZeroTerm => panic!("Can't divide polynomial by 0."),
            Term::Term(coeff, deg) => (coeff, deg),
        };

        let (mut scale, mut self_degree) = match first_term(&self.terms) {
            Term::ZeroTerm => return,
            Term::Term(coeff, degree) => {
                if degree < _rhs_deg {
                    return;
                }
                (coeff / _rhs_first, degree)
            }
        };

        while self_degree >= _rhs_deg {
            vec_sub_w_scale(&mut self.terms, self_degree, &_rhs.terms, _rhs_deg, scale);
            match first_term(&self.terms) {
                Term::ZeroTerm => break,
                Term::Term(coeff, degree) => {
                    scale = coeff / _rhs_first;
                    self_degree = degree;
                }
            }
        }
    }
}

impl<N> PartialEq for Polynomial<N>
where
    N: PartialEq + Zero + Copy,
{
    /// Returns true if self and other have the same terms.
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
        self.term_iter().eq(other.term_iter())
    }
}

impl<N> From<Vec<N>> for Polynomial<N>
where
    N: Copy + Zero,
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
    /// use rustnomial::{Polynomial};
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = Polynomial::from(vec![1.0, 4.0, 4.0]);
    /// let polynomial: Polynomial<f64> = vec![1.0, 4.0, 4.0].into();
    /// ```
    fn from(term_vec: Vec<N>) -> Self {
        Polynomial::new(term_vec)
    }
}

macro_rules! from_poly_a_to_b {
    ($A:ty, $B:ty) => {
        impl From<Polynomial<$A>> for Polynomial<$B> {
            fn from(item: Polynomial<$A>) -> Self {
                let mut vec = vec![<$B>::zero(); item.terms.len()];
                for (&val, dest) in item.terms.iter().zip(vec.iter_mut()) {
                    *dest = val as $B;
                }
                Polynomial::new(vec)
            }
        }
    };
}

upcast!(from_poly_a_to_b);
poly_from_str!(Polynomial);
fmt_poly!(Polynomial);

impl<N> Neg for Polynomial<N>
where
    N: Zero + Copy + Neg<Output = N>,
{
    type Output = Polynomial<N>;

    fn neg(self) -> Polynomial<N> {
        Polynomial::new(self.terms.iter().map(|&x| -x).collect())
    }
}

impl<N> Sub<Polynomial<N>> for Polynomial<N>
where
    N: Zero + Copy + Sub<Output = N> + SubAssign + Neg<Output = N>,
{
    type Output = Polynomial<N>;

    fn sub(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        if _rhs.len() > self.len() {
            let mut terms = _rhs.terms.clone();
            let offset = _rhs.len() - self.len();

            terms[..offset].iter_mut().for_each(|term| *term = -*term);
            terms[offset..]
                .iter_mut()
                .zip(self.terms)
                .for_each(|(term, val)| *term = val - *term);

            Polynomial::new(terms)
        } else {
            let mut terms = self.terms.clone();
            terms[self.terms.len() - _rhs.len()..]
                .iter_mut()
                .zip(_rhs.terms)
                .for_each(|(term, val)| *term -= val);
            Polynomial::new(terms)
        }
    }
}

impl<N> SubAssign<Polynomial<N>> for Polynomial<N>
where
    N: Neg<Output = N> + Sub<Output = N> + SubAssign + Copy + Zero,
{
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

impl<N> Add<Polynomial<N>> for Polynomial<N>
where
    N: Zero + Copy + AddAssign,
{
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

impl<N: Copy + Zero + AddAssign> AddAssign<Polynomial<N>> for Polynomial<N> {
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

impl<N> Mul<Polynomial<N>> for Polynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    type Output = Polynomial<N>;

    fn mul(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        Polynomial {
            terms: slice_mul(&self.terms, &_rhs.terms),
        }
    }
}

impl<N> Mul<&Polynomial<N>> for Polynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    type Output = Polynomial<N>;

    fn mul(self, _rhs: &Polynomial<N>) -> Polynomial<N> {
        Polynomial::new(slice_mul(&self.terms, &_rhs.terms))
    }
}

impl<N> Mul<Polynomial<N>> for &Polynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    type Output = Polynomial<N>;

    fn mul(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        Polynomial {
            terms: slice_mul(&self.terms, &_rhs.terms),
        }
    }
}

impl<N> Mul<&Polynomial<N>> for &Polynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    type Output = Polynomial<N>;

    fn mul(self, _rhs: &Polynomial<N>) -> Polynomial<N> {
        Polynomial::new(slice_mul(&self.terms, &_rhs.terms))
    }
}

impl<N> MulAssign<Polynomial<N>> for Polynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    fn mul_assign(&mut self, _rhs: Polynomial<N>) {
        self.terms = slice_mul(&self.terms, &_rhs.terms);
    }
}

impl<N> MulAssign<&Polynomial<N>> for Polynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    fn mul_assign(&mut self, _rhs: &Polynomial<N>) {
        self.terms = slice_mul(&self.terms, &_rhs.terms);
    }
}

impl<N: Zero + Copy + Mul<Output = N>> Mul<N> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn mul(self, _rhs: N) -> Polynomial<N> {
        Polynomial::new(self.terms.iter().map(|&x| x * _rhs).collect())
    }
}

impl<N: Copy + MulAssign> MulAssign<N> for Polynomial<N> {
    fn mul_assign(&mut self, _rhs: N) {
        for p in self.terms.iter_mut() {
            *p *= _rhs;
        }
    }
}

impl<N> Div<N> for Polynomial<N>
where
    N: Zero + Copy + Div<Output = N>,
{
    type Output = Polynomial<N>;

    fn div(self, _rhs: N) -> Polynomial<N> {
        Polynomial::new(self.terms.iter().map(|&x| x / _rhs).collect())
    }
}

impl<N: Copy + DivAssign> DivAssign<N> for Polynomial<N> {
    fn div_assign(&mut self, _rhs: N) {
        for p in self.terms.iter_mut() {
            *p /= _rhs;
        }
    }
}

impl<N: Zero + Copy> Shl<i32> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn shl(self, _rhs: i32) -> Polynomial<N> {
        if _rhs < 0 {
            self >> -_rhs
        } else {
            let index = first_nonzero_index(&self.terms);
            let mut terms = self.terms[index..].to_vec();
            terms.extend(vec![N::zero(); _rhs as usize]);
            Polynomial { terms }
        }
    }
}

impl<N: Zero + Copy> ShlAssign<i32> for Polynomial<N> {
    fn shl_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self >>= -_rhs;
        } else {
            self.terms.extend(vec![N::zero(); _rhs as usize]);
        }
    }
}

impl<N: Zero + Copy> Shr<i32> for Polynomial<N> {
    type Output = Polynomial<N>;

    fn shr(self, _rhs: i32) -> Polynomial<N> {
        if _rhs < 0 {
            self << -_rhs
        } else {
            let _rhs = _rhs as usize;
            let index = first_nonzero_index(&self.terms);
            Polynomial {
                terms: if _rhs > self.terms.len() {
                    vec![]
                } else {
                    self.terms[index..self.terms.len() - _rhs].to_vec()
                },
            }
        }
    }
}

impl<N: Zero + Copy> ShrAssign<i32> for Polynomial<N> {
    fn shr_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self <<= -_rhs;
        } else {
            let _rhs = _rhs as usize;
            if _rhs > self.terms.len() {
                self.terms = vec![];
            } else {
                self.terms = self.terms[..self.terms.len() - _rhs].to_vec();
            }
        }
    }
}

/// TODO:
/// modulo floordiv
#[cfg(test)]
mod test {
    use crate::{
        polynomial, Degree, Derivable, Evaluable, Integrable, Polynomial, SizedPolynomial,
    };

    #[test]
    fn test_polynomial_macro() {
        assert_eq!(polynomial!(1, 2, 3, 4), Polynomial::new(vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_eval() {
        let a = Polynomial::new(vec![1, 2, 3]);
        assert_eq!(25 + 2 * 5 + 3, a.eval(5));
    }

    #[test]
    fn test_derivative() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![2, 2]);
        assert_eq!(b, a.derivative());

        let a = Polynomial::new(vec![0, 1, 2, 3]);
        assert_eq!(b, a.derivative());

        let a = Polynomial::new(vec![1, 2, 3, 4]);
        let b = Polynomial::new(vec![3, 4, 3]);
        assert_eq!(b, a.derivative());
    }

    #[test]
    fn test_integral() {
        let a = Polynomial::new(vec![3, 2, 1]);
        let b = Polynomial::new(vec![1, 1, 1, 0]);
        assert_eq!(b, a.integral().polynomial);
    }

    #[test]
    fn test_integral_eval() {
        let a = Polynomial::new(vec![3, 2, 1]);
        assert_eq!(3, a.integral().eval(0, 1));
    }

    #[test]
    fn test_integral_const_substitute() {
        let a = Polynomial::new(vec![3, 2, 1]);
        let b = Polynomial::new(vec![1, 1, 1, 5]);
        assert_eq!(b, a.integral().replace_c(5));
    }

    #[test]
    fn test_add_lhs_bigger() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(c, b + a);
    }

    #[test]
    fn test_add_rhs_bigger() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(c, a + b);
    }

    #[test]
    fn test_add_lhs_bigger_assign() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let mut b = Polynomial::new(vec![1, 2, 3, 4]);
        b += a;
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(c, b);
    }

    #[test]
    fn test_add_rhs_bigger_assign() {
        let mut a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        a += b;
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_sub_lhs_bigger() {
        let a = Polynomial::new(vec![2, 3, 4]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 0, 0, 0]);
        assert_eq!(c, b - a);
    }

    #[test]
    fn test_sub_rhs_bigger() {
        let a = Polynomial::new(vec![2, 3, 4]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![-1, 0, 0, 0]);
        assert_eq!(c, a - b);
    }

    #[test]
    fn test_sub_lhs_bigger_assign() {
        let a = Polynomial::new(vec![2, 3, 4]);
        let mut b = Polynomial::new(vec![1, 2, 3, 4]);
        b -= a;
        let c = Polynomial::new(vec![1, 0, 0, 0]);
        assert_eq!(c, b);
    }

    #[test]
    fn test_sub_rhs_bigger_assign() {
        let mut a = Polynomial::new(vec![2, 3, 4]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        a -= b;
        let c = Polynomial::new(vec![-1, 0, 0, 0]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_negate() {
        let a = Polynomial::new(vec![1, 2, 3, 0, -5]);
        let c = Polynomial::new(vec![-1, -2, -3, 0, 5]);
        assert_eq!(c, -a);
    }

    #[test]
    fn test_mul_poly() {
        let a = Polynomial::new(vec![1, 2]);
        let b = a.clone();
        let c = Polynomial::new(vec![1, 4, 4]);
        assert_eq!(c, a * b);
    }

    #[test]
    fn test_mul_assign_poly() {
        let mut a = Polynomial::new(vec![1, 2]);
        let b = a.clone();
        a *= b;
        let c = Polynomial::new(vec![1, 4, 4]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_mul_num() {
        let a = Polynomial::new(vec![1, 2]);
        let c = Polynomial::new(vec![10, 20]);
        assert_eq!(c, a * 10);
    }

    #[test]
    fn test_mul_assign_num() {
        let mut a = Polynomial::new(vec![1, 2]);
        a *= 10;
        let c = Polynomial::new(vec![10, 20]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_equality() {
        let a = Polynomial::new(vec![1, 2]);
        let mut c = Polynomial::new(vec![0, 0, 0, 1, 2]);
        c.terms = vec![0, 0, 0, 1, 2];

        assert_eq!(c, a);

        c.terms = vec![1, 2, 0, 0, 0];

        assert_ne!(c, a);
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
        assert_eq!(c, a << 5);
    }

    #[test]
    fn test_shl_assign_pos() {
        let mut a = Polynomial::new(vec![1, 2]);
        a <<= 5;
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_shl_neg() {
        let a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(c, a << -5);
    }

    #[test]
    fn test_shl_assign_neg() {
        let mut a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        a <<= -5;
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_shr_pos() {
        let a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(c, a >> 5);
    }

    #[test]
    fn test_shr_assign_pos() {
        let mut a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        a >>= 5;
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_shr_neg() {
        let a = Polynomial::new(vec![1, 2]);
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(c, a >> -5);
    }

    #[test]
    fn test_shr_assign_neg() {
        let mut a = Polynomial::new(vec![1, 2]);
        a >>= -5;
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_shr_to_zero() {
        let a = Polynomial::new(vec![1, 2]);
        assert_eq!(Polynomial::zero(), a >> 5);
    }

    #[test]
    fn test_shr_assign_to_zero() {
        let mut a = Polynomial::new(vec![1, 2]);
        a >>= 5;
        assert_eq!(Polynomial::zero(), a);
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
    fn test_degree() {
        let a = Polynomial::new(vec![0, 0, 0, -1, -2, 3]);
        assert_eq!(Degree::Num(2), a.degree());
    }
}