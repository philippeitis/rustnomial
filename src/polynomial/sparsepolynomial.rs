use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Shl, ShlAssign, Shr,
    ShrAssign, Sub, SubAssign,
};

use num::{One, Zero};

use crate::numerics::{Abs, CanNegate, IsNegativeOne, IsPositive, PowUsize, TryFromUsizeExact};
use crate::polynomial::find_roots::find_roots;
use crate::{
    Degree, Derivable, Evaluable, FreeSizePolynomial, Integrable, Integral, MutablePolynomial,
    Polynomial, Roots, SizedPolynomial, Term, TryAddError,
};

#[derive(Debug, Clone)]
/// A type which stores the terms of a polynomial in a map. It is intended to store the terms
/// of polynomials where the degree is significantly larger than the number of non-zero terms.
/// Operations are significantly slower than with `Polynomial` if the number of non-zero terms
/// is very close to the degree.
pub struct SparsePolynomial<N> {
    pub terms: BTreeMap<usize, N>,
}

fn map_mul<N>(lhs: &BTreeMap<usize, N>, rhs: &BTreeMap<usize, N>) -> BTreeMap<usize, N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    let mut terms = BTreeMap::new();
    for (&rhs_deg, &rterm) in rhs {
        if rterm.is_zero() {
            continue;
        }

        for (&lhs_deg, &lterm) in lhs.iter() {
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

fn map_sub_w_scale<N>(lhs: &mut BTreeMap<usize, N>, rhs: &BTreeMap<usize, N>, rhs_scale: N)
where
    N: Copy + Neg<Output = N> + Sub<Output = N> + Mul<Output = N> + SubAssign,
{
    for (rdeg, &rcoeff) in rhs.iter() {
        match lhs.get_mut(rdeg) {
            None => {
                lhs.insert(*rdeg, -rcoeff * rhs_scale);
            }
            Some(lcoeff) => {
                *lcoeff -= rcoeff * rhs_scale;
            }
        }
    }
}

fn degree<N: Zero + Copy>(terms: &BTreeMap<usize, N>) -> Degree {
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

fn first_term<N: Zero + Copy>(terms: &BTreeMap<usize, N>) -> Term<N> {
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

impl<N: Zero + Copy> SizedPolynomial<N> for SparsePolynomial<N> {
    fn term_with_degree(&self, degree: usize) -> Term<N> {
        match self.terms.get(&degree) {
            None => Term::ZeroTerm,
            Some(&coeff) => Term::new(coeff, degree),
        }
    }

    fn terms_as_vec(&self) -> Vec<(N, usize)> {
        self.ordered_term_iter().collect()
    }

    /// Returns the degree of the `SparsePolynomial` it is called on, corresponding to the
    /// largest non-zero term.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, SparsePolynomial, Degree};
    /// let polynomial = SparsePolynomial::from(vec![1.0, 4.0, 4.0]);
    /// assert_eq!(Degree::Num(2), polynomial.degree());
    /// ```
    fn degree(&self) -> Degree {
        degree(&self.terms)
    }

    /// Returns a `SparsePolynomial` with no terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, SparsePolynomial};
    /// let zero = SparsePolynomial::<i32>::zero();
    /// assert!(zero.is_zero());
    /// assert!(zero.ordered_term_iter().next().is_none());
    /// assert!(zero.terms.is_empty());
    /// ```
    fn zero() -> SparsePolynomial<N> {
        SparsePolynomial {
            terms: BTreeMap::new(),
        }
    }

    /// Sets self to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SparsePolynomial, SizedPolynomial};
    /// let mut non_zero = SparsePolynomial::from(vec![0, 1]);
    /// assert!(!non_zero.is_zero());
    /// non_zero.set_to_zero();
    /// assert!(non_zero.is_zero());
    /// ```
    fn set_to_zero(&mut self) {
        self.terms.clear()
    }
}

impl<N> MutablePolynomial<N> for SparsePolynomial<N>
where
    N: Zero + Copy + AddAssign + SubAssign + CanNegate,
{
    fn try_add_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        Ok(self.add_term(coeff, degree))
    }

    fn try_sub_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        if coeff.is_zero() {
            return Ok(());
        }

        match self.terms.get_mut(&degree) {
            None => {
                if !N::can_negate() {
                    return Err(TryAddError::CanNotNegate);
                }
                let mut c = N::zero();
                c -= coeff;
                self.terms.insert(degree, c);
            }
            Some(val) => {
                *val -= coeff;
            }
        }
        Ok(())
    }
}

impl SparsePolynomial<f64> {
    /// Return the roots of the `SparsePolynomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SparsePolynomial, Roots, SizedPolynomial};
    /// let zero = SparsePolynomial::<f64>::zero();
    /// assert_eq!(Roots::InfiniteRoots, zero.roots());
    /// let constant = SparsePolynomial::from(vec![1.]);
    /// assert_eq!(Roots::NoRoots, constant.roots());
    /// let monomial = SparsePolynomial::from(vec![1.0, 0.,]);
    /// assert_eq!(Roots::ManyRealRoots(vec![0.]), monomial.roots());
    /// let binomial = SparsePolynomial::from(vec![1.0, 2.0]);
    /// assert_eq!(Roots::ManyRealRoots(vec![-2.0]), binomial.roots());
    /// let trinomial = SparsePolynomial::from(vec![1.0, 4.0, 4.0]);
    /// assert_eq!(Roots::ManyRealRoots(vec![-2.0, -2.0]), trinomial.roots());
    /// let quadnomial = SparsePolynomial::from(vec![1.0, 6.0, 12.0, 8.0]);
    /// assert_eq!(Roots::ManyRealRoots(vec![-2.0, -2.0, -2.0]), quadnomial.roots());
    /// ```
    pub fn roots(self) -> Roots<f64> {
        find_roots(&self)
    }
}

// Returns a `Polynomial` with the corresponding terms,
// in order of ax^n + bx^(n-1) + ... + cx + d
//
// # Arguments
//
// * ` terms ` - A vector of constants, in decreasing order of degree.
//
// # Example
//
// ```
// use polynomial::SparsePolynomial;
// use std::str::FromStr;
// // Corresponds to 1.0x^2 + 4.0x + 4.0
// let polynomial = SparsePolynomial::from_str("5x^2 + 11x + 2").unwrap();
// assert_eq!(SparsePolynomial::from(vec![5, 11, 2]), polynomial);
// ```

macro_rules! from_sparse_a_to_b {
    ($A:ty, $B:ty) => {
        impl From<SparsePolynomial<$A>> for SparsePolynomial<$B> {
            fn from(item: SparsePolynomial<$A>) -> Self {
                SparsePolynomial::new(item.terms.iter().map(|(&k, &v)| (k, v as $B)).collect())
            }
        }
    };
}

upcast!(from_sparse_a_to_b);
poly_from_str!(SparsePolynomial);
fmt_poly!(SparsePolynomial);

impl<N> FreeSizePolynomial<N> for SparsePolynomial<N>
where
    N: Zero + Copy + AddAssign,
{
    /// Returns a `SparsePolynomial` with the corresponding terms.
    ///
    /// # Arguments
    ///
    /// * ` terms ` - A slice of (coefficient, degree) pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SparsePolynomial, FreeSizePolynomial};
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = SparsePolynomial::from_terms(&[(1.0, 2), (4.0, 1), (4.0, 1)]);
    /// ```
    fn from_terms(terms: &[(N, usize)]) -> Self {
        let mut a = SparsePolynomial::new(BTreeMap::new());
        for &(term, degree) in terms {
            a.add_term(term, degree);
        }
        a
    }

    fn add_term(&mut self, coeff: N, degree: usize) {
        if coeff.is_zero() {
            return;
        }

        match self.terms.get_mut(&degree) {
            None => {
                self.terms.insert(degree, coeff);
            }
            Some(val) => {
                *val += coeff;
            }
        }
    }
}

impl<N: Copy> SparsePolynomial<N> {
    pub fn new(terms: BTreeMap<usize, N>) -> SparsePolynomial<N> {
        SparsePolynomial { terms }
    }
}

#[cfg(feature = "sparse_poly_trim")]
impl<N> SparsePolynomial<N>
where
    N: Zero + Copy,
{
    /// Reduces the size of the `SparsePolynomial` in memory if there exist terms which are zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::SparsePolynomial;
    /// use std::collections::BTreeMap;
    /// let mut polynomial = SparsePolynomial::from(vec![0.0f32, 0.0, 1.0, 0.0, 4.0, 4.0]);
    /// polynomial.trim();
    /// let expected: BTreeMap<usize, f32> = vec![(3usize, 1.0f32), (1, 4.0), (0, 4.0)].into_iter().collect();
    /// assert_eq!(expected, polynomial.terms);
    /// ```
    pub fn trim(&mut self) {
        self.terms.retain(|_, val| !val.is_zero());
    }
}

impl<N: Copy + Zero> SparsePolynomial<N> {
    pub fn ordered_term_iter(&self) -> impl Iterator<Item = (N, usize)> + '_ {
        self.terms.iter().rev().filter_map(|(&deg, &coeff)| {
            if coeff.is_zero() {
                None
            } else {
                Some((coeff, deg))
            }
        })
    }
}

impl<N> From<Vec<N>> for SparsePolynomial<N>
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
    /// use rustnomial::SparsePolynomial;
    /// // Corresponds to 1.0x^2 + 4.0x + 4.0
    /// let polynomial = SparsePolynomial::from(vec![1.0, 4.0, 4.0]);
    /// ```
    fn from(term_vec: Vec<N>) -> Self {
        let mut terms = BTreeMap::new();
        if !term_vec.is_empty() {
            let degree = term_vec.len() - 1;
            for (index, &val) in term_vec.iter().enumerate() {
                if !val.is_zero() {
                    terms.insert(degree - index, val);
                }
            }
        }
        SparsePolynomial { terms }
    }
}

// impl<N> SparsePolynomial<N>
// where
//     N: Zero + Copy,
// {
//     /// Reduces the size of the `SparsePolynomial` in memory by removing zero terms.
//     pub fn trim(&mut self) {
//         self.terms.retain(|_, coeff| !coeff.is_zero());
//     }
// }

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
    N: Zero + TryFromUsizeExact + Copy + Mul<Output = N>,
{
    /// Returns the derivative of the `SparsePolynomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SparsePolynomial, Derivable};
    /// let polynomial = SparsePolynomial::from(vec![4, 1, 5]);
    /// assert_eq!(SparsePolynomial::from(vec![8, 1]), polynomial.derivative());
    /// ```
    ///
    /// # Errors
    /// Will panic if a term has a degree which does not have a lossless
    /// representation in `N`.
    fn derivative(&self) -> SparsePolynomial<N> {
        let mut terms = BTreeMap::new();
        for (&degree, &coeff) in self.terms.iter() {
            if !coeff.is_zero() && degree != 0 {
                terms.insert(
                    degree - 1,
                    coeff
                        * N::try_from_usize_exact(degree)
                            .expect("Degree has no lossless representation in N."),
                );
            }
        }
        SparsePolynomial { terms }
    }
}

impl<N> Integrable<N, SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: PartialEq
        + Zero
        + Copy
        + Div<Output = N>
        + Mul<Output = N>
        + PowUsize
        + AddAssign
        + TryFromUsizeExact,
{
    /// Returns the integral of the `Polynomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SparsePolynomial, Integrable};
    /// let polynomial = SparsePolynomial::from(vec![1.0, 2.0, 5.0]);
    /// let integral = polynomial.integral();
    /// assert_eq!(&SparsePolynomial::from(vec![1.0/3.0, 1.0, 5.0, 0.0]), integral.inner());
    /// ```
    /// # Errors
    /// Will panic if a term has a degree, which when incremented by one, does not
    /// have a lossless representation in `N`.
    fn integral(&self) -> Integral<N, SparsePolynomial<N>> {
        let mut new_terms = BTreeMap::new();
        for (&deg, &coeff) in self.terms.iter() {
            new_terms.insert(
                deg + 1,
                coeff
                    / N::try_from_usize_exact(deg + 1)
                        .expect("Degree has no lossless representation in N."),
            );
        }

        Integral::new(SparsePolynomial::new(new_terms))
    }
}

impl<N> SparsePolynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero + One,
{
    pub fn borrow_mul(&self, rhs: &SparsePolynomial<N>) -> SparsePolynomial<N> {
        SparsePolynomial {
            terms: map_mul(&self.terms, &rhs.terms),
        }
    }

    /// Raises the `SparsePolynomial` to the power of exp, using exponentiation by squaring.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::SparsePolynomial;
    /// let polynomial = SparsePolynomial::from(vec![1.0, 2.0]);
    /// let polynomial_sqr = polynomial.pow(2);
    /// let polynomial_cub = polynomial.pow(3);
    /// assert_eq!(polynomial.clone() * polynomial.clone(), polynomial_sqr);
    /// assert_eq!(polynomial_sqr.clone() * polynomial.clone(), polynomial_cub);
    /// ```
    pub fn pow(&self, exp: usize) -> SparsePolynomial<N> {
        if exp == 0 {
            SparsePolynomial {
                terms: {
                    let mut terms = BTreeMap::new();
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
    /// Divides self by the given `SparsePolynomial`, and returns the quotient and remainder.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::SparsePolynomial;
    /// let polynomial = SparsePolynomial::from(vec![1.0, 2.0]);
    /// ```
    pub fn div_mod(&self, rhs: &SparsePolynomial<N>) -> (SparsePolynomial<N>, SparsePolynomial<N>) {
        let (rhs_first, rhs_deg) = match first_term(&rhs.terms) {
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
                if degree < rhs_deg {
                    return (
                        SparsePolynomial::zero(),
                        SparsePolynomial::new(self.terms.clone()),
                    );
                }
                (term / rhs_first, degree)
            }
        };

        let mut remainder = self.terms.clone();
        let offset = self_degree - rhs_deg;
        let mut div = SparsePolynomial::from(Vec::new());

        while self_degree >= rhs_deg {
            map_sub_w_scale(&mut remainder, &rhs.terms, scale);
            div.add_term(scale, offset);
            match first_term(&remainder) {
                Term::ZeroTerm => {
                    break;
                }
                Term::Term(coeff, degree) => {
                    scale = coeff / rhs_first;
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

    /// Returns the remainder of dividing `self` by `rhs`.
    fn rem(self, rhs: SparsePolynomial<N>) -> SparsePolynomial<N> {
        let (rhs_first, rhs_deg) = match first_term(&rhs.terms) {
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
                if degree < rhs_deg {
                    return SparsePolynomial::new(self.terms.clone());
                }
                (term / rhs_first, degree)
            }
        };

        let mut remainder = self.terms;

        while self_degree >= rhs_deg {
            map_sub_w_scale(&mut remainder, &rhs.terms, scale);
            match first_term(&remainder) {
                Term::ZeroTerm => {
                    break;
                }
                Term::Term(coeff, degree) => {
                    scale = coeff / rhs_first;
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
    /// Assign the remainder of dividing `self` by `rhs` to `self`.
    fn rem_assign(&mut self, rhs: SparsePolynomial<N>) {
        let (rhs_first, rhs_deg) = match first_term(&rhs.terms) {
            Term::ZeroTerm => {
                panic!("Can't divide polynomial by 0.");
            }
            Term::Term(coeff, deg) => (coeff, deg),
        };

        let (mut scale, mut self_degree) = match first_term(&self.terms) {
            Term::ZeroTerm => {
                return;
            }
            Term::Term(coeff, degree) => {
                if degree < rhs_deg {
                    return;
                }
                (coeff / rhs_first, degree)
            }
        };

        while self_degree >= rhs_deg {
            map_sub_w_scale(&mut self.terms, &rhs.terms, scale);
            match first_term(&self.terms) {
                Term::ZeroTerm => {
                    break;
                }
                Term::Term(coeff, degree) => {
                    scale = coeff / rhs_first;
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
    /// let a = SparsePolynomial::from(vec![1.0, 2.0]);
    /// let b = SparsePolynomial::from(vec![2.0, 2.0]);
    /// let c = SparsePolynomial::from(vec![1.0, 0.0]);
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

impl<N> Neg for SparsePolynomial<N>
where
    N: Zero + Copy + Neg<Output = N>,
{
    type Output = SparsePolynomial<N>;

    fn neg(self) -> SparsePolynomial<N> {
        let mut terms = BTreeMap::new();
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

    fn sub(mut self, rhs: SparsePolynomial<N>) -> SparsePolynomial<N> {
        for (deg, coeff) in rhs.terms {
            match self.terms.get_mut(&deg) {
                None => {
                    self.terms.insert(deg, -coeff);
                }
                Some(val) => {
                    *val -= coeff;
                }
            }
        }
        self
    }
}

impl<N> Sub<Polynomial<N>> for SparsePolynomial<N>
where
    N: Zero + Copy + Sub<Output = N> + SubAssign + Neg<Output = N>,
{
    type Output = SparsePolynomial<N>;

    fn sub(mut self, rhs: Polynomial<N>) -> SparsePolynomial<N> {
        for (coeff, deg) in rhs.ordered_term_iter() {
            match self.terms.get_mut(&deg) {
                None => {
                    self.terms.insert(deg, -coeff);
                }
                Some(val) => {
                    *val -= coeff;
                }
            }
        }
        self
    }
}

impl<N> SubAssign<SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Neg<Output = N> + Sub<Output = N> + SubAssign + Copy,
{
    fn sub_assign(&mut self, rhs: SparsePolynomial<N>) {
        for (deg, coeff) in rhs.terms {
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

    fn add(self, rhs: SparsePolynomial<N>) -> SparsePolynomial<N> {
        let mut terms = BTreeMap::new();
        for (deg, coeff) in rhs.terms {
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
    fn add_assign(&mut self, rhs: SparsePolynomial<N>) {
        for (&deg, &coeff) in rhs.terms.iter() {
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

    fn mul(self, rhs: SparsePolynomial<N>) -> SparsePolynomial<N> {
        SparsePolynomial {
            terms: map_mul(&self.terms, &rhs.terms),
        }
    }
}

impl<N> MulAssign<SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    fn mul_assign(&mut self, rhs: SparsePolynomial<N>) {
        self.terms = map_mul(&self.terms, &rhs.terms);
    }
}

impl<N> Mul<&SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    type Output = SparsePolynomial<N>;

    fn mul(self, rhs: &SparsePolynomial<N>) -> SparsePolynomial<N> {
        SparsePolynomial::new(map_mul(&self.terms, &rhs.terms))
    }
}

impl<N> MulAssign<&SparsePolynomial<N>> for SparsePolynomial<N>
where
    N: Mul<Output = N> + AddAssign + Copy + Zero,
{
    fn mul_assign(&mut self, rhs: &SparsePolynomial<N>) {
        self.terms = map_mul(&self.terms, &rhs.terms);
    }
}

impl<N: Copy + Mul<Output = N>> Mul<N> for SparsePolynomial<N> {
    type Output = SparsePolynomial<N>;

    fn mul(self, rhs: N) -> SparsePolynomial<N> {
        let mut terms = BTreeMap::new();
        for (&deg, &coeff) in self.terms.iter() {
            terms.insert(deg, coeff * rhs);
        }

        SparsePolynomial::new(terms)
    }
}

impl<N: Copy + MulAssign> MulAssign<N> for SparsePolynomial<N> {
    fn mul_assign(&mut self, rhs: N) {
        for (_, coeff) in self.terms.iter_mut() {
            *coeff *= rhs;
        }
    }
}

impl<N> Div<N> for SparsePolynomial<N>
where
    N: Copy + Div<Output = N>,
{
    type Output = SparsePolynomial<N>;

    fn div(self, rhs: N) -> SparsePolynomial<N> {
        let mut terms = BTreeMap::new();
        for (&deg, &coeff) in self.terms.iter() {
            terms.insert(deg, coeff / rhs);
        }

        SparsePolynomial::new(terms)
    }
}

impl<N> DivAssign<N> for SparsePolynomial<N>
where
    N: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: N) {
        for (_, coeff) in self.terms.iter_mut() {
            *coeff /= rhs;
        }
    }
}

impl<N: Copy> Shl<i32> for SparsePolynomial<N> {
    type Output = SparsePolynomial<N>;

    fn shl(self, rhs: i32) -> SparsePolynomial<N> {
        if rhs < 0 {
            self >> -rhs
        } else {
            let mut terms = BTreeMap::new();
            let rhs = rhs as usize;
            for (&deg, &coeff) in self.terms.iter() {
                terms.insert(deg + rhs, coeff);
            }
            SparsePolynomial::new(terms)
        }
    }
}

impl<N: Copy> ShlAssign<i32> for SparsePolynomial<N> {
    fn shl_assign(&mut self, rhs: i32) {
        if rhs < 0 {
            *self >>= -rhs;
        } else {
            let mut terms = BTreeMap::new();
            let rhs = rhs as usize;
            for (&deg, &coeff) in self.terms.iter() {
                terms.insert(deg + rhs, coeff);
            }
            self.terms = terms;
        }
    }
}

impl<N: Copy> Shr<i32> for SparsePolynomial<N> {
    type Output = SparsePolynomial<N>;

    fn shr(self, rhs: i32) -> SparsePolynomial<N> {
        if rhs < 0 {
            self << -rhs
        } else {
            let mut terms = BTreeMap::new();
            let rhs = rhs as usize;
            for (&deg, &coeff) in self.terms.iter() {
                if deg >= rhs {
                    terms.insert(deg - rhs, coeff);
                }
            }
            SparsePolynomial::new(terms)
        }
    }
}

impl<N: Copy> ShrAssign<i32> for SparsePolynomial<N> {
    fn shr_assign(&mut self, rhs: i32) {
        if rhs < 0 {
            *self <<= -rhs;
        } else {
            let mut terms = BTreeMap::new();
            let rhs = rhs as usize;
            for (&deg, &coeff) in self.terms.iter() {
                if deg >= rhs {
                    terms.insert(deg - rhs, coeff);
                }
            }
            self.terms = terms;
        }
    }
}

/// TODO:
/// modulo floordiv
#[cfg(test)]
mod test {
    use crate::{
        Degree, Derivable, Evaluable, Integrable, Polynomial, SizedPolynomial, SparsePolynomial,
    };

    #[test]
    fn test_from() {
        let a = SparsePolynomial::from(vec![1u8, 2, 3, 4]);
        let b: SparsePolynomial<u16> = a.into();
        assert_eq!(b, SparsePolynomial::from(vec![1u16, 2, 3, 4]))
    }

    #[test]
    fn test_eval() {
        let a = SparsePolynomial::from(vec![1, 2, 3]);
        assert_eq!(25 + 2 * 5 + 3, a.eval(5));
    }

    #[test]
    fn test_derivative() {
        let a = SparsePolynomial::from(vec![1, 2, 3]);
        let b = SparsePolynomial::from(vec![2, 2]);
        assert_eq!(b, a.derivative());

        let a = SparsePolynomial::from(vec![0, 1, 2, 3]);
        assert_eq!(b, a.derivative());

        let a = SparsePolynomial::from(vec![1, 2, 3, 4]);
        let b = SparsePolynomial::from(vec![3, 4, 3]);
        assert_eq!(b, a.derivative());
    }

    #[test]
    fn test_integral() {
        let a = SparsePolynomial::from(vec![3, 2, 1]);
        let b = SparsePolynomial::from(vec![1, 1, 1, 0]);
        assert_eq!(&b, a.integral().inner());
    }

    #[test]
    fn test_integral_eval() {
        let a = SparsePolynomial::from(vec![3, 2, 1]);
        assert_eq!(3, a.integral().eval(0, 1));
    }

    #[test]
    fn test_integral_const_substitute() {
        let a = SparsePolynomial::from(vec![3, 2, 1]);
        let b = SparsePolynomial::from(vec![1, 1, 1, 5]);
        assert_eq!(b, a.integral().replace_c(5));
    }

    #[test]
    fn test_add_lhs_bigger() {
        let a = SparsePolynomial::from(vec![1, 2, 3]);
        let b = SparsePolynomial::from(vec![1, 2, 3, 4]);
        let c = SparsePolynomial::from(vec![1, 3, 5, 7]);
        assert_eq!(c, b + a);
    }

    #[test]
    fn test_add_rhs_bigger() {
        let a = SparsePolynomial::from(vec![1, 2, 3]);
        let b = SparsePolynomial::from(vec![1, 2, 3, 4]);
        let c = SparsePolynomial::from(vec![1, 3, 5, 7]);
        assert_eq!(c, a + b);
    }

    #[test]
    fn test_add_lhs_bigger_assign() {
        let a = SparsePolynomial::from(vec![1, 2, 3]);
        let mut b = SparsePolynomial::from(vec![1, 2, 3, 4]);
        b += a;
        let c = SparsePolynomial::from(vec![1, 3, 5, 7]);
        assert_eq!(c, b);
    }

    #[test]
    fn test_add_rhs_bigger_assign() {
        let mut a = SparsePolynomial::from(vec![1, 2, 3]);
        let b = SparsePolynomial::from(vec![1, 2, 3, 4]);
        a += b;
        let c = SparsePolynomial::from(vec![1, 3, 5, 7]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_sub_lhs_bigger() {
        let a = SparsePolynomial::from(vec![2, 3, 4]);
        let b = SparsePolynomial::from(vec![1, 2, 3, 4]);
        let c = SparsePolynomial::from(vec![1, 0, 0, 0]);
        assert_eq!(c, b - a);
    }

    #[test]
    fn test_sub_rhs_bigger() {
        let a = SparsePolynomial::from(vec![2, 3, 4]);
        let b = SparsePolynomial::from(vec![1, 2, 3, 4]);
        let c = SparsePolynomial::from(vec![-1, 0, 0, 0]);
        assert_eq!(c, a - b);
    }

    #[test]
    fn test_sub_lhs_bigger_assign() {
        let a = SparsePolynomial::from(vec![2, 3, 4]);
        let mut b = SparsePolynomial::from(vec![1, 2, 3, 4]);
        b -= a;
        let c = SparsePolynomial::from(vec![1, 0, 0, 0]);
        assert_eq!(c, b);
    }

    #[test]
    fn test_sub_rhs_bigger_assign() {
        let mut a = SparsePolynomial::from(vec![2, 3, 4]);
        let b = SparsePolynomial::from(vec![1, 2, 3, 4]);
        a -= b;
        let c = SparsePolynomial::from(vec![-1, 0, 0, 0]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_negate() {
        let a = SparsePolynomial::from(vec![1, 2, 3, 0, -5]);
        let c = SparsePolynomial::from(vec![-1, -2, -3, 0, 5]);
        assert_eq!(c, -a);
    }

    #[test]
    fn test_mul_poly() {
        let a = SparsePolynomial::from(vec![1, 2]);
        let b = a.clone();
        let c = SparsePolynomial::from(vec![1, 4, 4]);
        assert_eq!(c, a * b);
    }

    #[test]
    fn test_mul_assign_poly() {
        let mut a = SparsePolynomial::from(vec![1, 2]);
        let b = a.clone();
        a *= b;
        let c = SparsePolynomial::from(vec![1, 4, 4]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_mul_num() {
        let a = SparsePolynomial::from(vec![1, 2]);
        let c = SparsePolynomial::from(vec![10, 20]);
        assert_eq!(c, a * 10);
    }

    #[test]
    fn test_mul_assign_num() {
        let mut a = SparsePolynomial::from(vec![1, 2]);
        a *= 10;
        let c = SparsePolynomial::from(vec![10, 20]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_equality() {
        let a = SparsePolynomial::from(vec![1, 2]);
        let c = SparsePolynomial::from(vec![0, 0, 0, 1, 2]);
        assert_eq!(c, a);

        let c = SparsePolynomial::from(vec![1, 2, 0, 0, 0]);

        assert_ne!(c, a);
    }

    #[test]
    fn test_equality_first_match() {
        let a = SparsePolynomial::from(vec![1, 2]);
        let b = SparsePolynomial::from(vec![1, 0]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_equality_different() {
        let a = SparsePolynomial::from(vec![1, 2]);
        let b = SparsePolynomial::from(vec![3, 7, 4]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_shl_pos() {
        let a = SparsePolynomial::from(vec![1, 2]);
        let c = SparsePolynomial::from(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(c, a << 5);
    }

    #[test]
    fn test_shl_assign_pos() {
        let mut a = SparsePolynomial::from(vec![1, 2]);
        a <<= 5;
        let c = SparsePolynomial::from(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_shl_neg() {
        let a = SparsePolynomial::from(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = SparsePolynomial::from(vec![1, 2]);
        assert_eq!(c, a << -5);
    }

    #[test]
    fn test_shl_assign_neg() {
        let mut a = SparsePolynomial::from(vec![1, 2, 0, 0, 0, 0, 0]);
        a <<= -5;
        let c = SparsePolynomial::from(vec![1, 2]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_shr_pos() {
        let a = SparsePolynomial::from(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = SparsePolynomial::from(vec![1, 2]);
        assert_eq!(c, a >> 5);
    }

    #[test]
    fn test_shr_assign_pos() {
        let mut a = SparsePolynomial::from(vec![1, 2, 0, 0, 0, 0, 0]);
        a >>= 5;
        let c = SparsePolynomial::from(vec![1, 2]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_shr_neg() {
        let a = SparsePolynomial::from(vec![1, 2]);
        let c = SparsePolynomial::from(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(c, a >> -5);
    }

    #[test]
    fn test_shr_assign_neg() {
        let mut a = SparsePolynomial::from(vec![1, 2]);
        a >>= -5;
        let c = SparsePolynomial::from(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(c, a);
    }

    #[test]
    fn test_shr_to_zero() {
        let a = SparsePolynomial::from(vec![1, 2]);
        assert_eq!(SparsePolynomial::zero(), a >> 5);
    }

    #[test]
    fn test_shr_assign_to_zero() {
        let mut a = SparsePolynomial::from(vec![1, 2]);
        a >>= 5;
        assert_eq!(SparsePolynomial::zero(), a);
    }

    #[test]
    fn test_exp() {
        let a = &SparsePolynomial::from(vec![1, 2]);
        let mut b = a.clone();
        assert_eq!(SparsePolynomial::from(vec![1]), a.pow(0));
        for i in 1..10 {
            assert_eq!(b, a.pow(i));
            b *= a;
        }
    }

    #[test]
    fn test_degree() {
        let a = SparsePolynomial::from(vec![0, 0, 0, -1, -2, 3]);
        assert_eq!(Degree::Num(2), a.degree());
    }

    #[test]
    fn test_generic_sub() {
        let a = SparsePolynomial::from(vec![0, 0, 0, -1, -2, 3]);
        let b = Polynomial::new(vec![-1, -2, 3]);
        let c = a - b;
        assert_eq!(SparsePolynomial::from(vec![0]), c);
    }

    #[test]
    fn test_pow() {
        let vec = vec![1u32, 2, 3, 4, 5];
        let a = SparsePolynomial::from(vec.clone());
        let b = Polynomial::new(vec.clone());
        let a = a.pow(8);
        let b = SparsePolynomial::from(b.pow(8).terms);
        assert_eq!(b, a);
    }
}
