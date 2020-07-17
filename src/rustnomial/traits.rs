use rustnomial::numerics::{HasZero, PowUsize, HasOne, IsNegativeOne, Abs};
use std::ops::{AddAssign, Mul};
use std::cmp::PartialEq;
use std::fmt;
use std::fmt::Display;
use rustnomial::degree::Term;

pub trait GenericPolynomial<N> {
    fn len(&self) -> usize;

    fn nth_term(&self, index: usize) -> Term<N>;

    fn degree_iter(&self) -> PolynomialDegreeIterator<N>;
}

pub trait Evaluable<N> {
    fn eval(&self, point: N) -> N;
}

impl<T: GenericPolynomial<N>, N> Evaluable<N> for T
    where N: HasZero + PowUsize + Copy + AddAssign + Mul<Output=N> + PartialEq {
    fn eval(&self, point: N) -> N {
        let mut sum = N::zero();
        for (val, degree) in self.degree_iter() {
            sum += val * point.upow(degree);
        }
        sum
    }
}

 impl<N> fmt::Display for GenericPolynomial<N>
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

pub struct PolynomialDegreeIterator<'a, N> {
    polynomial: &'a dyn GenericPolynomial<N>,
    index: usize,
}

impl<N> PolynomialDegreeIterator<'_, N> {
   pub fn new(polynomial: & dyn GenericPolynomial<N>) -> PolynomialDegreeIterator<N> {
        PolynomialDegreeIterator{
            polynomial,
            index: 0,
        }
    }
}
impl<N: PartialEq + HasZero + Copy> Iterator for PolynomialDegreeIterator<'_, N> {
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
