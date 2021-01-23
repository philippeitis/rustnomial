use core::fmt;

use num::One;

use crate::numerics::{Abs, IsNegativeOne, IsPositive};

macro_rules! fmt_poly {
    ($T:ident) => {
        use core::fmt;
        use $crate::strings::{write_leading_term, write_trailing_term};

        impl<N> fmt::Display for $T<N>
        where
            N: Zero + One + IsPositive + PartialEq + Abs + Copy + IsNegativeOne + fmt::Display,
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let mut iter = self.ordered_term_iter();
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

macro_rules! poly_from_str {
    ($T:ident) => {
        use core::str::FromStr;

        use $crate::err::PolynomialFromStringError;
        use $crate::terms::TermTokenizer;

        impl<N> FromStr for $T<N>
        where
            N: Zero + One + Copy + SubAssign + AddAssign + FromStr + $crate::numerics::CanNegate,
        {
            type Err = PolynomialFromStringError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let mut polynomial = $T::zero();
                let mut has_iterated = false;
                for term in TermTokenizer::new(s).map(|r| Term::from_str(&s[r])) {
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
mod test {
    use crate::{Integrable, Monomial, Polynomial, SizedPolynomial, SparsePolynomial};
    use alloc::string::ToString;
    use core::str::FromStr;

    #[test]
    fn test_from_str_all_terms() {
        let a = SparsePolynomial::<i32>::from_str("255x^2+15x+3").unwrap();
        let b = SparsePolynomial::from(vec![255, 15, 3]);
        assert_eq!(b, a);
    }

    #[test]
    fn test_from_str_missing_trailing_terms() {
        let a = SparsePolynomial::<i32>::from_str("5x^2").unwrap();
        let b = SparsePolynomial::from(vec![5, 0, 0]);
        assert_eq!(b, a);
    }

    #[test]
    fn test_from_str_term_degree_0() {
        let a = SparsePolynomial::<i32>::from_str("255x^2-15x^1+3x^0").unwrap();
        let b = SparsePolynomial::from(vec![255, -15, 3]);
        assert_eq!(
            b, a,
            "from_str should handle terms with explicit degree zero"
        );
    }

    #[test]
    fn test_from_str_negative_implicit_coefficient() {
        let a = SparsePolynomial::<i32>::from_str("-x^1").unwrap();
        let b = SparsePolynomial::from(vec![-1, 0]);
        assert_eq!(
            b, a,
            "from_str should handle a negative term without an explicit coefficient"
        );
    }

    #[test]
    fn test_from_str_terms_out_of_order() {
        let a = SparsePolynomial::<i32>::from_str("5+x").unwrap();
        let b = SparsePolynomial::from(vec![1, 5]);
        assert_eq!(
            b, a,
            "from_str should not fail if terms are not in specific order"
        );
    }

    #[test]
    fn test_from_str_repeated_degree() {
        let a = SparsePolynomial::<i32>::from_str("5x+11x").unwrap();
        let b = SparsePolynomial::from(vec![16, 0]);
        assert_eq!(b, a, "from_str should combine terms with equal degree");
    }

    #[test]
    fn test_dangling_caret_errors() {
        assert!(
            SparsePolynomial::<i32>::from_str("5+x^").is_err(),
            "Should err on dangling ^"
        );
    }

    #[test]
    fn test_sparse_polynomial_str_all_zeroes() {
        let a = SparsePolynomial::from(vec![0]);
        assert_eq!("0", a.to_string());

        let a: SparsePolynomial<i8> = SparsePolynomial::from(vec![]);
        assert_eq!("0", a.to_string());

        let a = SparsePolynomial::from(vec![0, 0]);
        assert_eq!("0", a.to_string());
    }

    #[test]
    fn test_sparse_polynomial_str() {
        let a = SparsePolynomial::from(vec![-1, -2, 3]);
        assert_eq!("-x^2 - 2x + 3", a.to_string());
    }

    #[test]
    fn test_sparse_polynomial_str_has_zeroes() {
        let a = SparsePolynomial::from(vec![-1, -2, 0, 0, 3]);
        assert_eq!("-x^4 - 2x^3 + 3", a.to_string());
    }

    #[test]
    fn test_sparse_polynomial_str_has_ones() {
        let a = SparsePolynomial::from(vec![-1, -1, -1, 0]);
        assert_eq!("-x^3 - x^2 - x", a.to_string());
    }

    #[test]
    fn test_sparse_polynomial_str_has_negative() {
        let a = SparsePolynomial::from(vec![-2, -1, -1, 0]);
        assert_eq!("-2x^3 - x^2 - x", a.to_string());
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
