use std::ops::AddAssign;

use num::Zero;

use {Degree, Term, TryAddError};

pub trait SizedPolynomial<N> {
    fn len(&self) -> usize;

    fn nth_term(&self, index: usize) -> Option<Term<N>>;

    fn term_iter(&self) -> TermIterator<N>;

    fn degree(&self) -> Degree;

    fn zero() -> Self
    where
        Self: Sized;

    fn is_zero(&self) -> bool;

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
