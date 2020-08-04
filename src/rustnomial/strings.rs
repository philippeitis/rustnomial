use num::One;
use rustnomial::numerics::{Abs, IsNegativeOne, IsPositive};
use std::fmt;

#[macro_export]
macro_rules! fmt_poly {
    ($T:ident) => {
        use std::fmt;

        impl<N> fmt::Display for $T<N>
        where
            N: Zero + One + IsPositive + PartialEq + Abs + Copy + IsNegativeOne + fmt::Display,
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let mut iter = self.term_iter();
                if let Some((coeff, degree)) = iter.next() {
                    write_leading_term(f, coeff, degree)?;
                    for (coeff, degree) in iter {
                        write_trailing_term(f, coeff, degree)?;
                    }
                    Ok(())
                } else {
                    write!(f, "0")
                }
            }
        }
    };
}

#[macro_export]
macro_rules! poly_from_str {
    ($T:ident) => {
        use std::str::FromStr;
        use $crate::rustnomial::err::PolynomialFromStringError;
        use $crate::rustnomial::terms::TermTokenizer;

        impl<N> FromStr for $T<N>
        where
            N: Zero + One + Copy + AddAssign + FromStr,
        {
            type Err = PolynomialFromStringError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let mut polynomial = $T::zero();
                let mut has_iterated = false;
                for term in TermTokenizer::new(s).map(|s| Term::from_str(s.as_str())) {
                    has_iterated = true;
                    match term {
                        Err(e) => return Err(PolynomialFromStringError::TermFromString(e)),
                        Ok(Term::ZeroTerm) => {}
                        Ok(Term::Term(coeff, deg)) => {
                            if let Err(e) = polynomial.try_add_term(coeff, deg) {
                                return Err(PolynomialFromStringError::AddingTerm(e));
                            }
                        }
                    }
                }

                if has_iterated {
                    Ok(polynomial)
                } else {
                    Err(PolynomialFromStringError::NoTermsFound)
                }
            }
        }
    };
}

pub(crate) fn write_leading_term<N>(f: &mut fmt::Formatter, coeff: N, degree: usize) -> fmt::Result
where
    N: fmt::Display + One + IsNegativeOne + PartialEq + Copy,
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
    N: fmt::Display + One + IsPositive + PartialEq + Copy + Abs,
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
