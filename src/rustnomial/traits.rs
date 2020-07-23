use rustnomial::degree::Term;
use num::Zero;

pub trait GenericPolynomial<N> {
    fn len(&self) -> usize;

    fn nth_term(&self, index: usize) -> Term<N>;

    fn term_iter(&self) -> TermIterator<N>;
}

pub trait Evaluable<N> {
    fn eval(&self, point: N) -> N;
}

pub struct TermIterator<'a, N> {
    polynomial: &'a dyn GenericPolynomial<N>,
    index: usize,
}

impl<N> TermIterator<'_, N> {
   pub fn new(polynomial: & dyn GenericPolynomial<N>) -> TermIterator<N> {
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
