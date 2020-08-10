use std::mem;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num::{Complex, Zero, One};
use roots::find_roots_sturm;

use rustnomial::numerics::{AbsSqrt, IsPositive, Cbrt};
use GenericPolynomial;

#[derive(Clone, Debug, PartialEq)]
pub enum Roots<N> {
    NoRoots,
    NoRootsFound,
    RealRoots(Vec<N>),
    ComplexRoots(Vec<Complex<N>>),
    InfiniteRoots,
    OnlyRealRoots(Vec<f64>),
}

pub fn discriminant_trinomial<N>(a: N, b: N, c: N) -> N
where
    N: Copy + Mul<Output = N> + Sub<Output = N> + From<u8>,
{
    b * b - a * c * N::from(4)
}

pub fn complex_roots_trinomial<N>(a: N, b: N, c: N) -> (Complex<N>, Complex<N>)
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
    let sqrt = discriminant.abs_sqrt() / a;
    if discriminant.is_positive() {
        (
            Complex::new(b + sqrt, N::zero()),
            Complex::new(b - sqrt, N::zero()),
        )
    } else {
        (Complex::new(b, sqrt), Complex::new(b, -sqrt))
    }
}

pub fn complex_roots_cubic<N>(a: N, b: N, c: N, d: N) -> (Complex<N>, Complex<N>, Complex<N>)
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
        + From<u8>
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
    let (ca, cb) = complex_roots_trinomial(N::one(), b, c);
    (Complex::new(x, N::zero()), ca, cb)
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
pub fn find_roots<N>(poly: &dyn GenericPolynomial<N>) -> Roots<N> where
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
        + Into<f64>
{
    match poly.term_iter().collect::<Vec<(N, usize)>>().as_slice() {
        [] => Roots::InfiniteRoots,
        [(_, 0)] => Roots::NoRoots,
        [_] => Roots::RealRoots(vec![N::zero()]),
        [(c1, 1), (c2, 0)] => Roots::RealRoots(vec![-*c2 / *c1]),
        [(a, 2), one_or_more @ ..] => {
            let (b, c) = match one_or_more {
                [] => (N::zero(), N::zero()),
                [(xc, 0)] => (N::zero(), *xc),
                [(xb, 1)] => (*xb, N::zero()),
                [(xb, 1), (xc, 0)] => (*xb, *xc),
                _ => unreachable!(),
            };
            let (root_a, root_b) = complex_roots_trinomial(*a, b, c);
            if root_a.im.is_zero() {
                Roots::RealRoots(vec![root_a.re, root_b.re])
            } else {
                Roots::ComplexRoots(vec![root_a, root_b])
            }
        },
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
            let (root_c, root_a, root_b) = complex_roots_cubic(*a, b, c, d);
            if root_a.im.is_zero() {
                Roots::RealRoots(vec![root_a.re, root_b.re, root_c.re])
            } else {
                Roots::ComplexRoots(vec![root_a, root_b, root_c])
            }
        },
        [vals @ ..] => {
            let (leading, degree) = vals[0];
            let leading = leading.into();
            let mut values = vec![0f64; degree];
            for (val, val_deg) in vals[1..].iter() {
                values[degree - val_deg - 1] = (*val).into() / leading;
            }
            Roots::OnlyRealRoots(find_roots_sturm(values.as_slice(), &mut 1e-8f64).into_iter().filter_map(Result::ok).collect::<Vec<f64>>())
            // Roots::OnlyRealRoots(find_roots_eigen(values).into_iter().collect::<Vec<f64>>())
        },
    }
}


#[derive(Clone, Debug, PartialEq)]
pub enum RootFindingErr {
    NotBracketed,
}

pub fn brent_solve<F>(f: F, a: f64, b: f64, eps: f64) -> Result<f64, RootFindingErr>
    where F: Fn(f64) -> f64 {
    let fa = f(a);
    let fb = f(b);
    if fa * fb > 0. {
        return Err(RootFindingErr::NotBracketed);
    }

    if fa.is_zero() {
        return Ok(a);
    } else if fb.is_zero() {
        return Ok(b);
    }

    let (mut a, mut fa, mut b, mut fb) = if fb.abs() > fa.abs() {
        (b, fb, a, fa)
    } else {
        (a, fa, b, fb)
    };

    let (mut c, mut fc) = (a, fa);
    let mut mflag = true;
    let mut s = a;
    let mut fs = fa;
    let mut d = 0.;
    while !fb.is_zero() && !fs.is_zero() && (b - a).abs() > eps {
        s = if fa != fc && fb != fc {
            a * fb * fc / ((fa - fb) * (fa - fc))
            + b * fa * fc / ((fb - fa) * (fb - fc))
            + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            b - fb * (b - a) / (fb - fa)
        };

        if (s < (3. * a + b) / 4. || s > b) ||
            (mflag && (s - b).abs() >= (b - c).abs() / 2.) ||
            (!mflag && (s - b).abs() >= (c - d).abs() / 2.) ||
            (mflag && (b - c).abs() < eps) ||
            (!mflag && (c - d).abs() < eps)
        {
            s = (a + b) / 2.;
            mflag = true;
        } else {
            mflag = false;
        }
        fs = f(s);

        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0. {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if fb.abs() > fa.abs() {
            mem::swap(&mut a, &mut b);
            mem::swap(&mut fa, &mut fb);
        }
    }
    return Ok(s)
}

#[cfg(test)]
mod test {
    use num::Complex;
    use rustnomial::find_roots::{find_roots, complex_roots_cubic};
    use ::{Roots, Polynomial, GenericPolynomial, Monomial, LinearBinomial};

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
        assert_eq!(Roots::RealRoots(vec![-2.]), find_roots(&p));
    }

    #[test]
    fn test_roots_cubic_a_equals_one() {
        let c = Complex::new(-2.0, 0.);
        assert_eq!((c, c, c), complex_roots_cubic(1f64, 6., 12., 8.));
    }

    #[test]
    fn test_roots_cubic_a_does_not_equal_one() {
        let c = Complex::new(-2.0, 0.);
        assert_eq!((c, c, c), complex_roots_cubic(2f64, 12., 24., 16.));
    }

    #[test]
    fn test_cubic_polynomials() {
        let p = Polynomial::new(vec![1f64, 6., 12., 8.]);
        assert_eq!(Roots::RealRoots(vec![-2., -2., -2.]), find_roots(&p));
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