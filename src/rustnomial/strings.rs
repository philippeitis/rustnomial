use std::fmt;

use num::One;

use rustnomial::numerics::{Abs, IsNegativeOne, IsPositive};

#[macro_export]
macro_rules! fmt_poly {
    ($T:ident) => {
        use std::fmt;
        use $crate::rustnomial::strings::{write_leading_term, write_trailing_term};

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

#[cfg(test)]
mod tests {
    use {GenericPolynomial, Polynomial, Integrable, SparsePolynomial};
    use std::str::FromStr;
    use Monomial;

    #[test]
    fn test_from_str() {
        match SparsePolynomial::<i32>::from_str("5x^2") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![5, 0, 0]);
                assert_eq!(b, a);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("255x^2+15x+3") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![255, 15, 3]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("255x^2-15x^1+3x^0") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![255, -15, 3]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("-x^1") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![-1, 0]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("5+x") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![1, 5]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        match SparsePolynomial::<i32>::from_str("5x+11x") {
            Ok(a) => {
                let b = SparsePolynomial::from_vec(vec![16, 0]);
                assert_eq!(a, b);
            }
            Err(e) => {
                assert!(false, e);
            }
        }

        assert!(
            SparsePolynomial::<i32>::from_str("5+x^").is_err(),
            "Should err on dangling ^"
        );
    }

    #[test]
    fn test_sparse_polynomial_str_all_zeroes() {
        let a = SparsePolynomial::from_vec(vec![0]);
        assert_eq!("0", a.to_string());

        let a: SparsePolynomial<i8> = SparsePolynomial::from_vec(vec![]);
        assert_eq!("0", a.to_string());

        let a = SparsePolynomial::from_vec(vec![0, 0]);
        assert_eq!("0", a.to_string());
    }

    #[test]
    fn test_sparse_polynomial_str() {
        let a = SparsePolynomial::from_vec(vec![-1, -2, 3]);
        assert_eq!("-x^2 - 2x + 3", a.to_string());
    }

    #[test]
    fn test_sparse_polynomial_str_has_zeroes() {
        let a = SparsePolynomial::from_vec(vec![-1, -2, 0, 0, 3]);
        assert_eq!("-x^4 - 2x^3 + 3", a.to_string());
    }

    #[test]
    fn test_sparse_polynomial_str_has_ones() {
        let a = SparsePolynomial::from_vec(vec![-1, -1, -1, 0]);
        assert_eq!("-x^3 - x^2 - x", a.to_string());
    }

    #[test]
    fn test_sparse_polynomial_str_has_negative() {
        let a = SparsePolynomial::from_vec(vec![-2, -1, -1, 0]);
        assert_eq!( "-2x^3 - x^2 - x", a.to_string());
    }

    #[test]
    fn test_sparse_integral_str() {
        let a = Polynomial::new(vec![-3, -2, 1]).integral();
        assert_eq!("-x^3 - x^2 + x + C", a.to_string());
    }

    #[test]
    fn test_str_monomial() {
        let a = Monomial::new(5, 2);
        assert_eq!("5x^2", a.to_string());
    }


    #[test]
    fn test_polynomial_str_all_zeroes() {
        let a = Polynomial::new(vec![0]);
        assert_eq!("0", a.to_string());

        let a: Polynomial<i8> = Polynomial::zero();
        assert_eq!("0", a.to_string());

        let a = Polynomial::new(vec![0, 0]);
        assert_eq!("0", a.to_string());

        let a: Polynomial<i8> = Polynomial { terms: vec![] };
        assert_eq!("0", a.to_string());

        let a = Polynomial { terms: vec![0] };
        assert_eq!("0", a.to_string());

        let a = Polynomial { terms: vec![0, 0] };
        assert_eq!("0", a.to_string());
    }

    #[test]
    fn test_polynomial_str() {
        let a = Polynomial::new(vec![-1, -2, 3]);
        assert_eq!("-x^2 - 2x + 3", a.to_string());
    }

    #[test]
    fn test_polynomial_str_has_zeroes() {
        let a = Polynomial::new(vec![-1, -2, 0, 0, 3]);
        assert_eq!("-x^4 - 2x^3 + 3", a.to_string());
    }

    #[test]
    fn test_polynomial_str_has_ones() {
        let a = Polynomial::new(vec![-1, -1, -1, 0]);
        assert_eq!("-x^3 - x^2 - x", a.to_string());
    }

    #[test]
    fn test_polynomial_str_has_negative() {
        let a = Polynomial::new(vec![-2, -1, -1, 0]);
        assert_eq!("-2x^3 - x^2 - x", a.to_string());
    }

    #[test]
    fn test_polynomial_str_negative_one() {
        let a = Polynomial::new(vec![-1]);
        assert_eq!("-1", a.to_string());
    }

}