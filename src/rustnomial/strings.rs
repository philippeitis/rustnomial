use num::{One, Zero};
use rustnomial::numerics::{Abs, IsNegativeOne, IsPositive};
use std::fmt;
use std::fmt::Display;
use ::{GenericPolynomial, FreeSizePolynomial};

#[macro_export]
macro_rules! fmt_poly {
    ($f:expr, $self:expr) => {
        {
            let mut iter = $self.term_iter();
            if let Some((coeff, degree)) = iter.next() {
                write_leading_term($f, coeff, degree)?;
                for (coeff, degree) in iter {
                    write_trailing_term($f, coeff, degree)?;
                }
                Ok(())
            } else {
                write!($f, "0")
            }
        }
    };
}

#[macro_export]
macro_rules! poly_from_str {
    // ($s:expr) => {
    //     {
    //         let mut polynomial = Polynomial::<i32>::zero();
    //         let mut has_iterated = false;
    //         for term in TermTokenizer::new($s).map(|s| Term::from_str(s.as_str())) {
    //             has_iterated = true;
    //             match term {
    //                 Err(msg) => return Err(msg),
    //                 Ok(Term::ZeroTerm) => {}
    //                 Ok(Term::Term(coeff, deg)) => {
    //                     polynomial.add_term(coeff, deg);
    //                 }
    //             }
    //         }
    //
    //         if has_iterated {
    //             Ok(polynomial)
    //         } else {
    //             Err("Given string did not have any terms.".to_string())
    //         }
    //     }
    // };

    ($s:expr, $polynomial:expr) => {
        {
            let mut has_iterated = false;
            for term in TermTokenizer::new($s).map(|s| Term::from_str(s.as_str())) {
                has_iterated = true;
                match term {
                    Err(msg) => return Err(msg),
                    Ok(Term::ZeroTerm) => {}
                    Ok(Term::Term(coeff, deg)) => {
                        $polynomial.add_term(coeff, deg);
                    }
                }
            }
            if has_iterated {
                Ok($polynomial)
            } else {
                Err("Given string did not have any terms.".to_string())
            }
        }
    };
}

pub(crate) fn write_leading_term<N>(f: &mut fmt::Formatter, coeff: N, degree: usize) -> fmt::Result
where
    N: Display + One + IsNegativeOne + PartialEq + Copy,
{
    if degree == 0 {
        return write!(f, "{}", coeff);
    }

    if coeff.is_negative_one() {
        write!(f, "-")?;
    } else if !coeff.is_one() {
        write!(f, "{}", coeff)?;
    }

    if degree > 1 {
        write!(f, "x^{}", degree)
    } else {
        write!(f, "x")
    }
}

pub(crate) fn write_trailing_term<N>(f: &mut fmt::Formatter, coeff: N, degree: usize) -> fmt::Result
where
    N: Display + One + IsPositive + PartialEq + Copy + Abs,
{
    write!(f, " {} ", if coeff.is_positive() { "+" } else { "-" })?;

    let coeff = coeff.abs();

    if degree == 0 {
        return write!(f, "{}", coeff);
    }

    if !coeff.is_one() {
        write!(f, "{}", coeff)?;
    }

    if degree > 1 {
        write!(f, "x^{}", degree)
    } else {
        write!(f, "x")
    }
}