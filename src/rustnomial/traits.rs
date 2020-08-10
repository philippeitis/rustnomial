use std::ops::AddAssign;

use num::Zero;

use rustnomial::err::TryAddError;
use {Degree, Term};

pub trait GenericPolynomial<N> {
    fn zero() -> Self
    where
        Self: Sized;

    fn len(&self) -> usize;

    fn nth_term(&self, index: usize) -> Term<N>;

    fn term_iter(&self) -> TermIterator<N>;

    fn degree(&self) -> Degree;

    fn is_zero(&self) -> bool;
}

pub trait MutablePolynomial<N> {
    fn try_add_term(&mut self, term: N, degree: usize) -> Result<(), TryAddError>;

    fn set_to_zero(&mut self);
}

pub trait FreeSizePolynomial<N>
where
    N: Zero + Copy + AddAssign,
{
    fn from_terms(terms: Vec<(N, usize)>) -> Self;

    fn add_term(&mut self, term: N, degree: usize);
}

pub trait Evaluable<N> {
    fn eval(&self, point: N) -> N;
}

pub struct TermIterator<'a, N> {
    polynomial: &'a dyn GenericPolynomial<N>,
    index: usize,
}

impl<N> TermIterator<'_, N> {
    pub fn new(polynomial: &dyn GenericPolynomial<N>) -> TermIterator<N> {
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
            if let Term::Term(coeff, deg) = nth_term {
                return Some((coeff, deg));
            }
        }

        None
    }
}
