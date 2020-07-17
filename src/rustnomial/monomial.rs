use rustnomial::traits::{PolynomialDegreeIterator, GenericPolynomial};
use rustnomial::numerics::{HasZero, HasOne, IsNegativeOne, Abs};
use ::{Integrable, Integral};
use Polynomial;
use std::ops::{Div, AddAssign};
use std::fmt;
use std::fmt::Display;
use rustnomial::degree::{Degree, Term};

#[derive(Debug, Clone)]
pub struct Monomial<N> {
    pub coefficient: N,
    pub deg: usize,
}

impl<N> Monomial<N> {
    pub fn new(coefficient: N, degree: usize) -> Monomial<N> {
        Monomial{coefficient, deg: degree}
    }
}

impl<N: Copy + HasZero + PartialEq> Monomial<N> {
    pub fn degree(&self) -> Degree {
        if self.coefficient == N::zero() {
            Degree::NegInf
        } else {
            Degree::Num(self.deg)
        }
    }
}

impl<N: Copy + HasZero + PartialEq> GenericPolynomial<N> for Monomial<N> {
    fn len(&self) -> usize {
        if self.coefficient != N::zero() {
            1
        } else {
            0
        }
    }

    fn nth_term(&self, index: usize) -> Term<N> {
        if index != 0 {
            Term::ZeroTerm
        } else {
            match self.degree() {
                Degree::NegInf => {
                    Term::ZeroTerm
                },
                Degree::Num(x) => {
                    Term::Term(self.coefficient, x)
                }
            }
        }
    }

    /// Returns an iterator for the `Polynomial`, yielding the term constant and degree. Terms are
    /// iterated over in descending degree order, excluding zero terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Monomial, GenericPolynomial};
    /// let monomial = Monomial::new(5, 2);
    /// let mut iter = monomial.degree_iter();
    /// assert_eq!(Some((5, 2)), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn degree_iter(&self) -> PolynomialDegreeIterator<N> {
        PolynomialDegreeIterator::new(self)
    }
}

impl<N: PartialEq + HasZero + Copy + AddAssign + Div<Output=N> + From<u8>> Integrable<N> for Monomial<N> {
    fn integral(&self) -> Integral<N> {
        match self.degree() {
            Degree::NegInf => Integral {
                polynomial: Polynomial::new(vec![N::zero()])
            },
            Degree::Num(x) => Integral {
                polynomial: Polynomial::from_terms(
                    vec![(self.coefficient / N::from((x + 1) as u8), x + 1)]
                )
            }
        }
    }
}



 impl<N> fmt::Display for Monomial<N>
    where N: HasZero + HasOne + Copy + IsNegativeOne + PartialEq + PartialOrd + Display + Abs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut iter = self.degree_iter();
        return match iter.next() {
            None => {
                write!(f, "0")
            }
            Some((term, degree)) => {
                if term.is_negative_one() {
                    write!(f, "-")?;
                } else if (term != N::one()) || (degree == 0) {
                    write!(f, "{}", term)?;
                }

                match degree {
                    0 => {write!(f, "")}
                    1 => {write!(f, "x")},
                    _ => {write!(f, "x^{}", degree)}
                }
            }
        };
    }
}

mod tests {
    use std::fmt::{Write, Debug};
    use ::{Monomial, Integrable};
    use rustnomial::traits::{Evaluable, GenericPolynomial, PolynomialDegreeIterator};
    use Polynomial;

    #[test]
    fn test_eval() {
        let a = Monomial::new(5, 2);
        assert_eq!(a.eval(5), 125);
    }

    #[test]
    fn test_integral() {
        let a = Monomial::new(5, 2);
        let integral = a.integral();
        assert_eq!(Polynomial::from_terms(vec![(5/3, 3)]), integral.polynomial);
    }

    #[test]
    fn test_str() {
        let a = Monomial::new(5, 2);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "5x^2");
    }


}
