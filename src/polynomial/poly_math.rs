use core::ops::{AddAssign, Mul, SubAssign};

use num::Zero;

use crate::numerics::CanNegate;
use crate::MutablePolynomial;
use crate::TryAddError;

#[macro_export]
macro_rules! poly_add {
    ($($x:expr),+) => {{
        use $crate::{Polynomial, SizedPolynomial};
        let mut sink = Polynomial::zero();
        // This will never error.
        let _ = poly_add!(sink; $($x),*);
        sink
    }};
    // sink += (x_0 + x_1 + ... + x_z)
    ($sink:expr; $($x:expr),*) => {{
        use $crate::poly_math::add_poly;

        let res: Result<Vec<_>, _> = vec![
            $(add_poly($x.ordered_term_iter(), &mut $sink)),*
        ].into_iter().collect();

        if let Err(x) = res {
            Err(x)
        } else {
            Ok(())
        }
    }};
}

/// Adds `poly` to `sink`.
///
/// # Examples
/// ```
/// use rustnomial::{poly_math::add_poly, SparsePolynomial, Polynomial, SizedPolynomial};
/// let source = SparsePolynomial::from(vec![1, 2, 3, 4]);
/// let mut sink = Polynomial::zero();
/// assert!(add_poly(source.ordered_term_iter(), &mut sink).is_ok());
/// assert_eq!(Polynomial::new(vec![1, 2, 3, 4]), sink);
/// ```
///
/// # Errors
/// Returns an error if adding any of `poly`'s terms to `sink` fails.
pub fn add_poly<N, S: MutablePolynomial<N>>(
    poly: impl Iterator<Item = (N, usize)>,
    sink: &mut S,
) -> Result<(), TryAddError>
where
    N: Zero + AddAssign + Copy,
{
    for (coeff, deg) in poly {
        sink.try_add_term(coeff, deg)?;
    }

    Ok(())
}

#[macro_export]
macro_rules! poly_mul {
    ($lhs:expr, $rhs:expr) => {{
        use $crate::{FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
        use $crate::poly_math::mul_poly;
        let mut sink = Polynomial::zero();
        mul_poly_vec($lhs.ordered_term_iter(), &$rhs.ordered_term_iter().collect(), &mut sink);
        sink
    }};

    ($sink:expr; $lhs:expr) => {{
        use $crate::{FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
        use $crate::poly_math::{mul_poly, mul_poly_vec};
        let sink_terms = $sink.ordered_term_iter().collect();
        $sink.set_to_zero();
        mul_poly_vec($lhs.ordered_term_iter(), sink_terms, &mut $sink)
    }};

    ($lhs:expr, $rhs:expr $(,$x:expr )+) => {{
        poly_mul!(poly_mul!($lhs, $rhs) $(,$x)*)
    }};

    ($sink:expr; $lhs:expr, $(,$x:expr )*) => {{
        use $crate::{FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
        use $crate::poly_math::mul_poly;
        poly_mul!($sink; $lhs);
        poly_mul!($sink; $($x),*)
    }};
}

/// Multiplies `rhs` with the terms in `lhs`, storing the result in `sink`.
///
/// # Examples
/// ```
/// use rustnomial::{poly_math::mul_poly_vec, SparsePolynomial, Polynomial, SizedPolynomial};
/// let source = SparsePolynomial::from(vec![1, 2, 3, 4]);
/// let mut sink = Polynomial::zero();
/// assert!(mul_poly_vec(source.ordered_term_iter(), &[], &mut sink).is_ok());
/// assert_eq!(Polynomial::zero(), sink);
/// ```
///
/// # Errors
/// Returns an error if adding any of the terms resulting from multiplying `rhs` with `lhs` to
/// `sink` fails.
pub fn mul_poly_vec<N, S: MutablePolynomial<N>>(
    rhs: impl Iterator<Item = (N, usize)>,
    lhs: &[(N, usize)],
    sink: &mut S,
) -> Result<(), TryAddError>
where
    N: Zero + AddAssign + Copy + Mul<Output = N>,
{
    for (rcoeff, rdeg) in rhs {
        for &(lcoeff, ldeg) in lhs.iter() {
            sink.try_add_term(rcoeff * lcoeff, rdeg + ldeg)?;
        }
    }

    Ok(())
}

/// Subtracts the terms of `poly` from `sink`.
///
/// # Examples
/// ```
/// use rustnomial::{poly_math::sub_poly, SparsePolynomial, Polynomial, SizedPolynomial};
/// let source = SparsePolynomial::from(vec![1, 2, 3, 4]);
/// let mut sink = Polynomial::zero();
/// assert!(sub_poly(source.ordered_term_iter(), &mut sink).is_ok());
/// assert_eq!(Polynomial::from(vec![-1, -2, -3, -4]), sink);
/// ```
///
/// # Errors
/// Returns an error if subtracting any of the terms from `poly` fails.
pub fn sub_poly<N, S: MutablePolynomial<N>>(
    poly: impl Iterator<Item = (N, usize)>,
    sink: &mut S,
) -> Result<(), TryAddError>
where
    N: Zero + SubAssign + Copy + CanNegate,
{
    for (coeff, deg) in poly {
        sink.try_sub_term(coeff, deg)?;
    }

    Ok(())
}

#[macro_export]
macro_rules! poly_sub {
    // lhs - (x_1 + x_2 + ... + x_z)
    ($lhs:expr $(,$x:expr )+) => {{
        use $crate::{Polynomial, SizedPolynomial};
        let mut sink = Polynomial::zero();
        let _ = poly_add!(sink; $lhs);
        let _ = poly_sub!(sink; $($x),*);
        sink
    }};
    // sink -= (x_0 + x_1 + ... + x_z)
    ($sink:expr; $($x:expr ),*) => {{
        use $crate::poly_math::sub_poly;

        let res: Result<Vec<_>, _> = vec![
            $(sub_poly($x.ordered_term_iter(), &mut $sink)),*
        ].into_iter().collect();

        if let Err(x) = res {
            Err(x)
        } else {
            Ok(())
        }
    }};
}

#[cfg(test)]
mod test {
    use alloc::vec::Vec;
    /// Tests macro usage as it would happen outside of this crate,
    /// since we do not know what items have been imported outside.

    fn zero_poly<N: num::Zero + Copy>() -> crate::Polynomial<N> {
        use crate::SizedPolynomial;
        crate::Polynomial::zero()
    }

    fn poly<N: num::Zero + Copy>(v: Vec<N>) -> crate::Polynomial<N> {
        crate::Polynomial::new(v)
    }

    #[test]
    fn test_poly_add() {
        let mut base = zero_poly();
        let new = poly(vec![1u32, 2u32, 3u32]);
        assert_eq!(poly_add!(base, new), new);
        assert!(poly_add!(base; new).is_ok());
        assert_eq!(base, new);
    }

    #[test]
    fn test_poly_sub() {
        let mut base = zero_poly();
        let new = poly(vec![1i32, 2, 3]);
        assert_eq!(poly_sub!(base, new), -new.clone());
        assert!(poly_sub!(base; new).is_ok());
        assert_eq!(base, -new);
    }
}
