use num::Zero;

use crate::{Degree, Term, TryAddError};

pub trait SizedPolynomial<N> {
    /// Returns the term with the given degree from `self`.
    /// If the term degree is larger than the actual degree, `ZeroTerm` will be returned.
    /// However, terms which are zero will also be returned as `ZeroTerm`, so this does
    /// not indicate that the final term has been reached.
    fn term_with_degree(&self, degree: usize) -> Term<N>;

    /// Returns an iterator over the terms of `self`, yielding the coefficient and degree of each
    /// non-zero term, in order of descending degree.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, SizedPolynomial};
    /// let polynomial = Polynomial::new(vec![1, 0, 2, 3]);
    /// let mut iter = polynomial.term_iter();
    /// assert_eq!(Some((1, 3)), iter.next());
    /// assert_eq!(Some((2, 1)), iter.next());
    /// assert_eq!(Some((3, 0)), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn term_iter(&self) -> TermIterator<N>
    where
        Self: Sized,
    {
        TermIterator::new(self)
    }

    /// Returns the degree of `self`.
    fn degree(&self) -> Degree;

    /// Returns the zero-instance of `Self`.
    fn zero() -> Self
    where
        Self: Sized;

    /// Returns true if all terms of `self` are zero, and false if a non-zero term exists.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, Polynomial};
    /// let zero = Polynomial::new(vec![0, 0]);
    /// assert!(zero.is_zero());
    /// let non_zero = Polynomial::new(vec![0, 1]);
    /// assert!(!non_zero.is_zero());
    /// ```
    fn is_zero(&self) -> bool {
        self.degree() == Degree::NegInf
    }

    /// Sets the terms of `self` to zero.
    fn set_to_zero(&mut self);
}

pub trait GenericPolynomial<N>: SizedPolynomial<N> + MutablePolynomial<N> + Evaluable<N> {}

pub trait MutablePolynomial<N> {
    /// Tries to add the term with given coefficient and `degree` to `self`, returning an error
    /// if the particular term can not be added to self without violating constraints.
    ///
    /// # Errors
    /// Fails if the term with coefficient `coeff` and degree `degree` can not be added
    /// to `self` without violating one or more of `self`'s invariants.
    fn try_add_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError>;

    /// Tries to subtract the term with given coefficient and `degree` from `self`, returning
    /// an error if the particular term can not be subtracted from self without violating
    /// constraints.
    ///
    /// # Errors
    /// Fails if the term with coefficient `coeff` and degree `degree` can not be subtracted from
    /// `self` without violating one or more of `self`'s invariants.
    fn try_sub_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError>;
}

pub trait FreeSizePolynomial<N> {
    /// Creates an instance of `self` with the provided terms.
    fn from_terms(terms: &[(N, usize)]) -> Self
    where
        Self: Sized;

    /// Adds the term with given coefficient `coeff` and degree `degree` to `self`.
    fn add_term(&mut self, coeff: N, degree: usize);
}

pub trait Evaluable<N> {
    /// Evaluates `self` at `point`, and returns the result.
    fn eval(&self, point: N) -> N;
}

/// An iterator over terms.
pub struct TermIterator<'a, N> {
    polynomial: &'a dyn SizedPolynomial<N>,
    top: Degree,
}

impl<N> TermIterator<'_, N> {
    pub(crate) fn new(polynomial: &dyn SizedPolynomial<N>) -> TermIterator<N> {
        TermIterator {
            top: polynomial.degree(),
            polynomial,
        }
    }
}

impl<N: Zero + Copy> Iterator for TermIterator<'_, N> {
    type Item = (N, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Degree::Num(deg) = self.top {
            let term = self.polynomial.term_with_degree(deg);
            self.top = if deg != 0 {
                Degree::Num(deg - 1)
            } else {
                Degree::NegInf
            };
            if let Term::Term(coeff, deg) = term {
                return Some((coeff, deg));
            }
        }

        None
    }
}
