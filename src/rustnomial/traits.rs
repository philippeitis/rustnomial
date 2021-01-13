use std::ops::AddAssign;

use num::Zero;

use {Degree, Term, TryAddError};

pub trait SizedPolynomial<N> {
    fn len(&self) -> usize;

    /// Returns the `index`th term in order of decreasing degree from `SizedPolynomial`,
    /// if it exists.
    fn nth_term(&self, index: usize) -> Option<Term<N>>;

    /// Returns an in-order iterator over the terms of `SizedPolynomial`.
    fn term_iter(&self) -> TermIterator<N>;

    /// Returns the degree of `SizedPolynomial`.
    fn degree(&self) -> Degree;

    /// Returns the zero-instance of `SizedPolynomial`.
    fn zero() -> Self
    where
        Self: Sized;

    /// Returns whether `SizedPolynomial` is zero or not.
    fn is_zero(&self) -> bool;

    /// Sets the terms of `SizedPolynomial` to zero.
    fn set_to_zero(&mut self);
}

pub trait GenericPolynomial<N>: SizedPolynomial<N> + MutablePolynomial<N> + Evaluable<N> {}

pub trait MutablePolynomial<N> {
    /// Adds the term with given coefficient and `degree` to self, returning an error
    /// if the particular term can not be added to self without violating constraints.
    fn try_add_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError>;
}

pub trait FreeSizePolynomial<N>
where
    N: Zero + Copy + AddAssign,
{
    /// Creates an instance of `Self` with the provided terms
    fn from_terms(terms: &[(N, usize)]) -> Self;

    /// Adds the term with given coefficient and `degree` to self.
    fn add_term(&mut self, coeff: N, degree: usize);
}

pub trait Evaluable<N> {
    fn eval(&self, point: N) -> N;
}

pub struct TermIterator<'a, N> {
    polynomial: &'a dyn SizedPolynomial<N>,
    index: usize,
}

impl<N> TermIterator<'_, N> {
    pub(crate) fn new(polynomial: &dyn SizedPolynomial<N>) -> TermIterator<N> {
        TermIterator {
            polynomial,
            index: 0,
        }
    }
}
impl<N: Zero + Copy> Iterator for TermIterator<'_, N> {
    type Item = (N, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.polynomial.len() {
            let nth_term = self.polynomial.nth_term(self.index);
            self.index += 1;
            if let Some(Term::Term(coeff, deg)) = nth_term {
                return Some((coeff, deg));
            }
        }

        None
    }
}
