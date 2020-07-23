use rustnomial::numerics::{HasZero, HasOne, IsNegativeOne, Abs};
use std::cmp::PartialEq;
use std::fmt;
use std::fmt::Display;
use rustnomial::degree::Term;

pub trait GenericPolynomial<N> {
    fn len(&self) -> usize;

    fn nth_term(&self, index: usize) -> Term<N>;

    fn term_iter(&self) -> TermIterator<N>;
}

pub trait Evaluable<N> {
    fn eval(&self, point: N) -> N;
}

impl<N> fmt::Display for dyn GenericPolynomial<N>
    where N: HasZero + HasOne + Copy + IsNegativeOne + PartialEq + PartialOrd + Display + Abs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut iter = self.term_iter();
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
                    1 => { write!(f, "x")?; },
                    _ => { write!(f, "x^{}", degree)?; }
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
                1 => { write!(f, "x")?; },
                _ => { write!(f, "x^{}", degree)?; }
            }
        }

        write!(f, "")
    }
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
impl<N: PartialEq + HasZero + Copy> Iterator for TermIterator<'_, N> {
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
