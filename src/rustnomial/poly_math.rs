use num::Zero;
use rustnomial::traits::FreeSizePolynomial;
use std::ops::{AddAssign, Mul, Neg};
use {GenericPolynomial, Polynomial};

pub fn poly_add<N>(
    src_a: Box<dyn GenericPolynomial<N>>,
    src_b: Box<dyn GenericPolynomial<N>>,
) -> Polynomial<N>
where
    N: Zero + Copy + AddAssign,
{
    let mut sink = Polynomial::zero();
    for (coeff, deg) in src_a.term_iter() {
        sink.add_term(coeff, deg);
    }

    for (coeff, deg) in src_b.term_iter() {
        sink.add_term(coeff, deg);
    }

    sink
}

pub fn poly_mul<N>(
    src_a: Box<dyn GenericPolynomial<N>>,
    src_b: Box<dyn GenericPolynomial<N>>,
) -> Polynomial<N>
where
    N: Zero + Copy + AddAssign + Mul<Output = N>,
{
    let mut sink = Polynomial::zero();
    for (coeff_a, deg_a) in src_a.term_iter() {
        for (coeff_b, deg_b) in src_b.term_iter() {
            sink.add_term(coeff_a * coeff_b, deg_a + deg_b);
        }
    }

    sink
}

pub fn poly_sub<N>(
    src_a: Box<dyn GenericPolynomial<N>>,
    src_b: Box<dyn GenericPolynomial<N>>,
) -> Polynomial<N>
where
    N: Zero + Copy + AddAssign + Neg<Output = N>,
{
    let mut sink = Polynomial::zero();
    for (coeff, deg) in src_a.term_iter() {
        sink.add_term(coeff, deg);
    }

    for (coeff, deg) in src_b.term_iter() {
        sink.add_term(-coeff, deg);
    }

    sink
}
