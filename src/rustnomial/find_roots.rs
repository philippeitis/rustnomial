use std::ops::{Add, Div, Mul, Neg, Sub};

use num::{Complex, One, Zero};
use roots::find_roots_sturm;

use rustnomial::numerics::{AbsSqrt, Cbrt, IsPositive};
use GenericPolynomial;

#[derive(Clone, Debug, PartialEq)]
pub enum Roots<N> {
    NoRoots,
    NoRootsFound,
    OneRealRoot(N),
    TwoRealRoots(N, N),
    ThreeRealRoots(N, N, N),
    ManyRealRoots(Vec<N>),
    OneComplexRoot(Complex<N>),
    TwoComplexRoots(Complex<N>, Complex<N>),
    ThreeComplexRoots(Complex<N>, Complex<N>, Complex<N>),
    ManyComplexRoots(Vec<Complex<N>>),
    InfiniteRoots,
    OnlyRealRoots(Vec<f64>),
}

pub fn discriminant_trinomial<N>(a: N, b: N, c: N) -> N
where
    N: Copy + Mul<Output = N> + Sub<Output = N> + From<u8>,
{
    b * b - a * c * N::from(4)
}

pub fn trinomial_roots<N>(a: N, b: N, c: N) -> Roots<N>
where
    N: Copy
        + Mul<Output = N>
        + Div<Output = N>
        + Sub<Output = N>
        + Add<Output = N>
        + AbsSqrt
        + IsPositive
        + Zero
        + Neg<Output = N>
        + From<u8>,
{
    let discriminant = discriminant_trinomial(a, b, c);
    let a = a * N::from(2);
    let b = -b / a;

    if discriminant.is_zero() {
        return Roots::TwoRealRoots(b, b)
    }

    let sqrt = discriminant.abs_sqrt() / a;
    if discriminant.is_positive() {
        Roots::TwoRealRoots(b + sqrt, b - sqrt)
    } else {
        Roots::TwoComplexRoots(Complex::new(b, sqrt), Complex::new(b, -sqrt))
    }
}

pub fn cubic_roots<N>(a: N, b: N, c: N, d: N) -> Roots<N>
where
    N: Copy
        + Mul<Output = N>
        + Div<Output = N>
        + Sub<Output = N>
        + Add<Output = N>
        + AbsSqrt
        + Cbrt
        + IsPositive
        + Zero
        + One
        + Neg<Output = N>
        + From<u8>,
{
    let sqr = |x: N| x * x;
    let cub = |x: N| x * x * x;
    let p = -b / (N::from(3) * a);
    let q = cub(p) + (b * c - N::from(3) * a * d) / (N::from(6) * sqr(a));
    let r = c / (N::from(3) * a);
    let k = (sqr(q) + cub(r - sqr(p))).abs_sqrt();
    let x = (q + k).cbrt() + (q - k).cbrt() + p;

    let b = b / a + x;
    let c = c / a + b * x;
    let roots = trinomial_roots(N::one(), b, c);
    match roots {
        Roots::TwoRealRoots(a, b) => Roots::ThreeRealRoots(x, a, b),
        Roots::TwoComplexRoots(a, b) => Roots::ThreeComplexRoots(Complex::new(x, N::zero()), a, b),
        _ => unreachable!(),
    }
}

// pub fn complex_roots_quartic<N>(a: N, b: N, c: N, d: N, e: N) -> (Complex<N>, Complex<N>, Complex<N>, Complex<N>)
// where
//     N: Copy
//         + Mul<Output = N>
//         + Div<Output = N>
//         + Sub<Output = N>
//         + Add<Output = N>
//         + AbsSqrt
//         + Cbrt
//         + IsPositive
//         + Zero
//         + One
//         + Neg<Output = N>
//         + From<u8>
//         + PartialOrd
// {
//     let sqr = |x: N| x * x;
//     let cub = |x: N| x * x * x;
// }

/// x^4 + 8 x^3 + 24 x^2 + 32 x + 16
/// Finds the roots of the polynomial with terms defined by the given vector, where each element
/// is a tuple consisting of the coefficient and degree. Order is not guaranteed.
pub fn find_roots<N>(poly: &dyn GenericPolynomial<N>) -> Roots<N>
where
    N: Copy
        + Mul<Output = N>
        + Div<Output = N>
        + Sub<Output = N>
        + Add<Output = N>
        + Cbrt
        + AbsSqrt
        + IsPositive
        + Zero
        + One
        + Neg<Output = N>
        + From<u8>
        + Into<f64>,
{
    match poly.term_iter().collect::<Vec<(N, usize)>>().as_slice() {
        [] => Roots::InfiniteRoots,
        [(_, 0)] => Roots::NoRoots,
        [_] => Roots::ManyRealRoots(vec![N::zero()]),
        [(c1, 1), (c2, 0)] => Roots::ManyRealRoots(vec![-*c2 / *c1]),
        [(a, 2), one_or_more @ ..] => {
            let (b, c) = match one_or_more {
                [] => (N::zero(), N::zero()),
                [(xc, 0)] => (N::zero(), *xc),
                [(xb, 1)] => (*xb, N::zero()),
                [(xb, 1), (xc, 0)] => (*xb, *xc),
                _ => unreachable!(),
            };
            match trinomial_roots(*a, b, c) {
                Roots::TwoComplexRoots(a, b) => Roots::ManyComplexRoots(vec![a, b]),
                Roots::TwoRealRoots(a, b) => Roots::ManyRealRoots(vec![a, b]),
                _ => unreachable!(),
            }
        }
        [(a, 3), one_or_more @ ..] => {
            let (b, c, d) = match one_or_more {
                [] => (N::zero(), N::zero(), N::zero()),
                [(xd, 0)] => (N::zero(), N::zero(), *xd),
                [(xc, 1)] => (N::zero(), *xc, N::zero()),
                [(xc, 1), (xd, 0)] => (N::zero(), *xc, *xd),
                [(xb, 2)] => (*xb, N::zero(), N::zero()),
                [(xb, 2), (xd, 0)] => (*xb, N::zero(), *xd),
                [(xb, 2), (xc, 1)] => (*xb, *xc, N::zero()),
                [(xb, 2), (xc, 1), (xd, 0)] => (*xb, *xc, *xd),
                _ => unreachable!(),
            };
            match cubic_roots(*a, b, c, d) {
                Roots::ThreeComplexRoots(a, b, c) => Roots::ManyComplexRoots(vec![a, b, c]),
                Roots::ThreeRealRoots(a, b, c) => Roots::ManyRealRoots(vec![a, b, c]),
                _ => unreachable!(),
            }
        }
        [vals @ ..] => {
            let (leading, degree) = vals[0];
            let leading = leading.into();
            let mut values = vec![0f64; degree];
            for (val, val_deg) in vals[1..].iter() {
                values[degree - val_deg - 1] = (*val).into() / leading;
            }
            Roots::OnlyRealRoots(
                find_roots_sturm(values.as_slice(), &mut 1e-8f64)
                    .into_iter()
                    .filter_map(Result::ok)
                    .collect::<Vec<f64>>(),
            )
            // Roots::OnlyRealRoots(find_roots_eigen(values).into_iter().collect::<Vec<f64>>())
        }
    }
}

#[cfg(test)]
mod test {
    use rustnomial::find_roots::{cubic_roots, find_roots};
    use {GenericPolynomial, LinearBinomial, Monomial, Polynomial, Roots};

    #[test]
    fn test_roots_empty() {
        let p = Polynomial::<f64>::zero();
        assert_eq!(Roots::InfiniteRoots, find_roots(&p));
    }

    #[test]
    fn test_roots_constant() {
        let p = Monomial::new(1., 0);
        assert_eq!(Roots::NoRoots, find_roots(&p));
    }

    #[test]
    fn test_roots_binomial() {
        let p = LinearBinomial::new([1., 2.]);
        assert_eq!(Roots::ManyRealRoots(vec![-2.]), find_roots(&p));
    }

    #[test]
    fn test_roots_cubic_a_equals_one() {
        assert_eq!(
            Roots::ThreeRealRoots(-2.0, -2.0, -2.0),
            cubic_roots(1f64, 6., 12., 8.)
        );
    }

    #[test]
    fn test_roots_cubic_a_does_not_equal_one() {
        assert_eq!(
            Roots::ThreeRealRoots(-2.0, -2.0, -2.0),
            cubic_roots(2f64, 12., 24., 16.)
        );
    }

    #[test]
    fn test_cubic_polynomials() {
        let p = Polynomial::new(vec![1f64, 6., 12., 8.]);
        assert_eq!(Roots::ManyRealRoots(vec![-2., -2., -2.]), find_roots(&p));
    }

    #[test]
    fn test_quartic_polynomials() {
        let p = Polynomial::new(vec![1f64, 2.]).pow(9);
        assert_eq!(Roots::OnlyRealRoots(vec![-2., -2.]), find_roots(&p));
    }

    // #[test]
    // fn test_roots_quartic_a_equals_one() {
    //     let c = Complex::new(-2.0, 0.);
    //     assert_eq!((c, c, c, c), complex_roots_quartic(1f32, 8., 24., 32., 16.));
    // }
    //
    // #[test]
    // fn test_roots_quartic_a_does_not_equal_one() {
    //     let c = Complex::new(-2.0, 0.);
    //     assert_eq!((c, c, c, c), complex_roots_quartic(2f32, 16., 48., 64., 32.));
    // }
}
