use alloc::vec::Vec;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Shr, ShrAssign, Sub, SubAssign,
};

use num::{Complex, One, Zero};

use crate::numerics::{
    Abs, AbsSqrt, IsNegativeOne, IsPositive, TryFromUsizeContinuous, TryFromUsizeExact,
};
use crate::polynomial::find_roots::{discriminant_trinomial, trinomial_roots};
use crate::polynomial::polynomial::{degree, term_with_deg};
use crate::{
    Degree, Derivable, Evaluable, Integrable, Integral, MutablePolynomial, Polynomial,
    QuadraticTrinomial, Roots, SizedPolynomial, Term, TryAddError,
};

#[derive(Debug, Clone)]
/// A type that stores terms of a linear binomial in a static array. Operations are
/// much faster than on Polynomial for the same size polynomial, but terms can not
/// be added freely.
pub struct ArrayPolynomial<N: Sized, const SIZE: usize> {
    coefficients: [N; SIZE],
}

impl<N: Sized, const SIZE: usize> ArrayPolynomial<N, SIZE> {
    pub fn new(coefficients: [N; SIZE]) -> Self {
        Self { coefficients }
    }
}

impl<N, const SIZE: usize> MutablePolynomial<N> for ArrayPolynomial<N, SIZE>
where
    N: Zero + SubAssign + AddAssign + Copy,
{
    fn try_add_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        if degree < SIZE {
            self.coefficients[SIZE - degree - 1] += coeff;
            Ok(())
        } else {
            Err(TryAddError::DegreeOutOfBounds)
        }
    }

    fn try_sub_term(&mut self, coeff: N, degree: usize) -> Result<(), TryAddError> {
        if degree < SIZE {
            self.coefficients[SIZE - degree - 1] -= coeff;
            Ok(())
        } else {
            Err(TryAddError::DegreeOutOfBounds)
        }
    }
}

impl<N, const SIZE: usize> Evaluable<N> for ArrayPolynomial<N, SIZE>
where
    N: AddAssign + MulAssign + Copy + Zero,
{
    fn eval(&self, point: N) -> N {
        if let Some((&last, first)) = self.coefficients.split_last() {
            if point.is_zero() {
                return last;
            }

            let mut sum = N::zero();
            for &val in first.iter() {
                sum += val;
                sum *= point;
            }
            sum + last
        } else {
            N::zero()
        }
    }
}

impl<N, const SIZE: usize> Derivable<N> for ArrayPolynomial<N, SIZE>
where
    N: Zero + One + TryFromUsizeContinuous + Copy + Mul<Output = N> + SubAssign,
{
    fn derivative(&self) -> Self {
        let mut result = [N::zero(); SIZE];
        let mut degree =
            N::try_from_usize_cont(SIZE).expect("Degree has no lossless representation in N.");
        if SIZE >= 1 {
            for (&coeff_l, coeff_r) in self.coefficients.iter().zip(result[1..].iter_mut()) {
                degree -= N::one();
                *coeff_r = degree * coeff_l;
            }
        }
        ArrayPolynomial::new(result)
    }
}

impl<N, const SIZE: usize> Integrable<N, Polynomial<N>> for ArrayPolynomial<N, SIZE>
where
    N: Zero
        + One
        + Copy
        + DivAssign
        + Mul<Output = N>
        + MulAssign
        + AddAssign
        + TryFromUsizeContinuous
        + SubAssign,
{
    fn integral(&self) -> Integral<N, Polynomial<N>> {
        let mut degree =
            N::try_from_usize_cont(SIZE).expect("Degree can not be losslessly represented.");
        let mut terms = self.coefficients.to_vec();
        for term in terms.iter_mut() {
            *term /= degree;
            degree -= N::one();
        }
        terms.push(N::zero());
        Integral::new(Polynomial { terms })
    }
}

impl<N, const SIZE: usize> PartialEq for ArrayPolynomial<N, SIZE>
where
    N: Zero + PartialEq + Copy,
{
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }
}

impl<N> ArrayPolynomial<N, 3>
where
    N: Copy
        + Zero
        + Mul<Output = N>
        + Neg<Output = N>
        + Sub<Output = N>
        + From<u8>
        + Div<Output = N>
        + AbsSqrt
        + IsPositive
        + One,
{
    pub fn discriminant(&self) -> N {
        let [a, b, c] = self.coefficients;
        discriminant_trinomial(a, b, c)
    }

    /// Return the roots of `QuadraticTrinomial` with largest
    /// first, smallest second.
    pub fn roots(&self) -> Roots<N> {
        let [a, b, c] = self.coefficients;
        trinomial_roots(a, b, c)
    }

    pub fn complex_factors(
        &self,
    ) -> (
        N,
        ArrayPolynomial<Complex<N>, 2>,
        ArrayPolynomial<Complex<N>, 2>,
    ) {
        match self.roots() {
            Roots::TwoComplexRoots(root_a, root_b) => (
                self.coefficients[0],
                ArrayPolynomial::new([Complex::new(N::one(), -N::zero()), root_a]),
                ArrayPolynomial::new([Complex::new(N::one(), -N::zero()), root_b]),
            ),
            Roots::TwoRealRoots(a, b) => (
                self.coefficients[0],
                ArrayPolynomial::new([
                    Complex::new(N::one(), N::zero()),
                    Complex::new(-a, N::zero()),
                ]),
                ArrayPolynomial::new([
                    Complex::new(N::one(), N::zero()),
                    Complex::new(-b, N::zero()),
                ]),
            ),
            _ => unreachable!(),
        }
    }

    pub fn real_factors(&self) -> Option<(N, ArrayPolynomial<N, 2>, ArrayPolynomial<N, 2>)> {
        if let Roots::TwoRealRoots(root_a, root_b) = self.roots() {
            Some((
                self.coefficients[0],
                ArrayPolynomial::new([N::one(), -root_a]),
                ArrayPolynomial::new([N::one(), -root_b]),
            ))
        } else {
            None
        }
    }
}

impl<N: Zero + Copy, const SIZE: usize> ArrayPolynomial<N, SIZE> {
    pub fn ordered_term_iter(&self) -> impl Iterator<Item = (N, usize)> + '_ {
        self.coefficients
            .iter()
            .enumerate()
            .filter_map(|(index, &coeff)| {
                if coeff.is_zero() {
                    None
                } else {
                    Some((coeff, 1 - index))
                }
            })
    }
}

impl<N: Copy + Zero, const SIZE: usize> SizedPolynomial<N> for ArrayPolynomial<N, SIZE> {
    fn term_with_degree(&self, degree: usize) -> Term<N> {
        term_with_deg(&self.coefficients, degree)
    }

    fn terms_as_vec(&self) -> Vec<(N, usize)> {
        self.ordered_term_iter().collect()
    }

    /// Returns the degree of the `Polynomial` it is called on, corresponding to the
    /// largest non-zero term.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, Polynomial, Degree};
    /// let polynomial = Polynomial::new(vec![1.0, 4.0, 4.0]);
    /// assert_eq!(Degree::Num(2), polynomial.degree());
    /// ```
    fn degree(&self) -> Degree {
        degree(&self.coefficients)
    }

    /// Returns a `Polynomial` with no terms.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{SizedPolynomial, Polynomial};
    /// let zero = Polynomial::<i32>::zero();
    /// assert!(zero.is_zero());
    /// assert!(zero.ordered_term_iter().next().is_none());
    /// assert!(zero.terms.is_empty());
    /// ```
    fn zero() -> Self {
        ArrayPolynomial::new([N::zero(); SIZE])
    }

    /// Sets self to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{Polynomial, SizedPolynomial};
    /// let mut non_zero = Polynomial::from(vec![0, 1]);
    /// assert!(!non_zero.is_zero());
    /// non_zero.set_to_zero();
    /// assert!(non_zero.is_zero());
    /// ```
    fn set_to_zero(&mut self) {
        self.coefficients = [N::zero(); SIZE]
    }
}

impl<N> ArrayPolynomial<N, 2>
where
    N: Copy + Neg<Output = N> + Div<Output = N> + Zero,
{
    /// Return the root of `LinearBinomial`.
    ///
    /// # Example
    ///
    /// ```
    /// use rustnomial::{LinearBinomial, Roots, SizedPolynomial};
    /// let binomial = LinearBinomial::new([1.0, 2.0]);
    /// assert_eq!(Roots::OneRealRoot(-2.0), binomial.root());
    /// let zero = LinearBinomial::<i32>::zero();
    /// assert_eq!(Roots::InfiniteRoots, zero.root());
    /// let constant = LinearBinomial::new([0, 1]);
    /// assert_eq!(Roots::NoRoots, constant.root());
    /// ```
    pub fn root(&self) -> Roots<N> {
        let [a, b] = self.coefficients;
        if a.is_zero() {
            if b.is_zero() {
                Roots::InfiniteRoots
            } else {
                Roots::NoRoots
            }
        } else {
            Roots::OneRealRoot(-b / a)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::polynomial::array_polynomial::ArrayPolynomial;
    use crate::{Derivable, Evaluable, Integrable, Polynomial, QuadraticTrinomial};

    #[test]
    fn test_init() {
        let a = ArrayPolynomial::new([1, 2, 3]);
    }

    #[test]
    fn test_eval() {
        let a = QuadraticTrinomial::new([1, 2, 3]);
        let b = ArrayPolynomial::new([1, 2, 3]);
        assert_eq!(a.eval(5), b.eval(5));
    }

    #[test]
    fn test_derivative() {
        let polynomial = ArrayPolynomial::new([4, 1, 5]);
        assert_eq!(ArrayPolynomial::new([0, 8, 1]), polynomial.derivative());
        let polynomial = ArrayPolynomial::new([4, 1]);
        assert_eq!(ArrayPolynomial::new([0, 4]), polynomial.derivative());
        let polynomial = ArrayPolynomial::new([99]);
        assert_eq!(ArrayPolynomial::new([0]), polynomial.derivative());
        let polynomial: ArrayPolynomial<u32, 0> = ArrayPolynomial::new([]);
        assert_eq!(ArrayPolynomial::new([]), polynomial.derivative());
    }

    #[test]
    fn test_integral() {
        let polynomial = Polynomial::new(vec![1.0, 2.0, 5.0]);
        let arr_poly = ArrayPolynomial::new([1.0, 2.0, 5.0]);
        assert_eq!(polynomial.integral().inner(), arr_poly.integral().inner());
    }
}
