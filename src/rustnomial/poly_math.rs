use TryAddError;
use {GenericPolynomial, MutablePolynomial};
use std::ops::{AddAssign, Mul};
use num::Zero;
use std::process::Output;

#[macro_export]
macro_rules! poly_add {
    ($mand:expr, $( $x:expr ),+) => {{
        use $crate::Polynomial;
        let mut sink = Polynomial::zero();
        poly_add!($mand $(,$x)*; sink);
        sink
    }};

    ($( $x:expr ),+; $sink:expr) => {{
        use $crate::{FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
        use $crate::poly_math::add_poly;

        let mut temp_vec: Vec<&dyn GenericPolynomial<_>> = Vec::new();
        $(
            let x = $x;
            temp_vec.push(&x);
        )*
        let res: Result<Vec<_>, _> = temp_vec.into_iter().map(|x| add_poly(x, &mut $sink)).collect();
        if let Err(x) = res {
            Err(x)
        } else {
            Ok(())
        }
    }};
}

pub fn add_poly<N>(
    poly: &dyn GenericPolynomial<N>,
    sink: &mut MutablePolynomial<N>,
) -> Result<(), TryAddError> where N: Zero + AddAssign + Copy {
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

    ($lhs:expr; $sink:expr) => {{
        use $crate::{FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
        use $crate::poly_math::{mul_poly, mul_poly_vec};
        let sink_terms = $sink.term_iter().collect();
        $sink.set_to_zero();
        mul_poly_vec(&$lhs, sink_terms, &mut $sink)
    }};

    ($lhs:expr, $rhs:expr, $( $x:expr ),+) => {{
        poly_mul!(poly_mul!($lhs, $rhs) $(,$x)*)
    }};

    ($lhs:expr, $rhs:expr $(,$x:expr )*; $sink:expr) => {{
        use $crate::{FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
        use $crate::poly_math::mul_poly;
        poly_mul!($lhs; $sink);
        poly_mul!($rhs $(,$x)*; $sink)
    }};
}

pub fn mul_poly_vec<N>(
    rhs: &dyn GenericPolynomial<N>,
    lhs: Vec<(N, usize)>,
    sink: &mut MutablePolynomial<N>,
) -> Result<(), TryAddError> where N: Zero + AddAssign + Copy + Mul<Output=N> {
    for (rcoeff, rdeg) in rhs.term_iter() {
        for &(lcoeff, ldeg) in lhs.iter() {
            sink.try_add_term(rcoeff * lcoeff, rdeg + ldeg)?;
        }
    }

    Ok(())
}

pub fn mul_poly<N>(
    rhs: &dyn GenericPolynomial<N>,
    lhs: &dyn GenericPolynomial<N>,
    sink: &mut MutablePolynomial<N>,
) -> Result<(), TryAddError> where N: Zero + AddAssign + Copy + Mul<Output=N> {
    for (rcoeff, rdeg) in rhs.term_iter() {
        for (lcoeff, ldeg) in lhs.term_iter() {
            sink.try_add_term(rcoeff * lcoeff, rdeg + ldeg)?;
        }
    }

    Ok(())
}

#[macro_export]
macro_rules! poly_sub {
    ($lhs:expr, $rhs:expr) => {{
        use $crate::Polynomial;
        let mut sink = Polynomial::zero();
        poly_sub!($lhs, $rhs, sink)
    }};

    ($lhs:expr, $rhs:expr, $sink:expr) => {{
        use $crate::{FreeSizePolynomial, GenericPolynomial, MutablePolynomial};
        for (coeff, deg) in $lhs.term_iter() {
            $sink.try_add_term(coeff, deg);
        }

        for (coeff, deg) in $rhs.term_iter() {
            $sink.try_add_term(-coeff, deg);
        }

        $sink
    }};
}
