use std::ops::{AddAssign, Mul, SubAssign};

use num::Zero;

use crate::numerics::CanNegate;
use crate::TryAddError;
use crate::{MutablePolynomial, SizedPolynomial};

#[macro_export]
macro_rules! poly_add {
    ($mand:expr $(,$x:expr)+) => {{
        use $crate::{Polynomial, SizedPolynomial};
        let mut sink = Polynomial::zero();
        // This will never error.
        let _ = poly_add!(sink; $mand $(,$x)*);
        sink
    }};
    // sink += (x_0 + x_1 + ... + x_z)
    ($sink:expr; $x0:expr $(,$x:expr ),*) => {{
        use $crate::poly_math::add_poly;

        let res: Result<Vec<_>, _> = vec![
            add_poly(&$x0, &mut $sink)
            $(,add_poly(&$x, &mut $sink))*
        ].into_iter().collect();

        if let Err(x) = res {
            Err(x)
        } else {
            Ok(())
        }
    }};
}

pub fn add_poly<N, P: SizedPolynomial<N>, S: MutablePolynomial<N>>(
    poly: &P,
    sink: &mut S,
) -> Result<(), TryAddError>
where
    N: Zero + AddAssign + Copy,
{
    for (coeff, deg) in poly.term_iter() {
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
        mul_poly(&$lhs, &$rhs, &mut sink);
        sink
    }};

    ($sink:expr; $lhs:expr) => {{
        use $crate::{FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
        use $crate::poly_math::{mul_poly, mul_poly_vec};
        let sink_terms = $sink.term_iter().collect();
        $sink.set_to_zero();
        mul_poly_vec(&$lhs, sink_terms, &mut $sink)
    }};

    ($lhs:expr, $rhs:expr $(,$x:expr )+) => {{
        poly_mul!(poly_mul!($lhs, $rhs) $(,$x)*)
    }};

    ($sink:expr; $lhs:expr, $rhs:expr $(,$x:expr )*) => {{
        use $crate::{FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
        use $crate::poly_math::mul_poly;
        poly_mul!($sink; $lhs);
        poly_mul!($sink; $rhs $(,$x)*)
    }};
}

pub fn mul_poly_vec<N, R: SizedPolynomial<N>, S: MutablePolynomial<N>>(
    rhs: &R,
    lhs: Vec<(N, usize)>,
    sink: &mut S,
) -> Result<(), TryAddError>
where
    N: Zero + AddAssign + Copy + Mul<Output = N>,
{
    for (rcoeff, rdeg) in rhs.term_iter() {
        for &(lcoeff, ldeg) in lhs.iter() {
            sink.try_add_term(rcoeff * lcoeff, rdeg + ldeg)?;
        }
    }

    Ok(())
}

pub fn mul_poly<N, R: SizedPolynomial<N>, L: SizedPolynomial<N>, S: MutablePolynomial<N>>(
    rhs: &R,
    lhs: &L,
    sink: &mut S,
) -> Result<(), TryAddError>
where
    N: Zero + AddAssign + Copy + Mul<Output = N>,
{
    for (rcoeff, rdeg) in rhs.term_iter() {
        for (lcoeff, ldeg) in lhs.term_iter() {
            sink.try_add_term(rcoeff * lcoeff, rdeg + ldeg)?;
        }
    }

    Ok(())
}

pub fn sub_poly<N, P: SizedPolynomial<N>, S: MutablePolynomial<N>>(
    poly: &P,
    sink: &mut S,
) -> Result<(), TryAddError>
where
    N: Zero + SubAssign + Copy + CanNegate,
{
    for (coeff, deg) in poly.term_iter() {
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
    ($sink:expr; $x0:expr $(,$x:expr )*) => {{
        use $crate::poly_math::sub_poly;

        let res: Result<Vec<_>, _> = vec![
            sub_poly(&$x0, &mut $sink)
            $(,sub_poly(&$x, &mut $sink))*
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
