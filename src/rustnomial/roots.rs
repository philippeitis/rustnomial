use num::{Complex, Zero, Float};
use rustnomial::numerics::{AbsSqrt, IsPositive, Abs};
use std::ops::{Add, Div, Mul, Neg, Sub};
use GenericPolynomial;
use rustnomial::roots::RootFindingErr::NotBracketed;
use std::mem;

#[derive(Clone, Debug, PartialEq)]
pub enum Roots<N> {
    NoRoots,
    NoRootsFound,
    RealRoots(Vec<N>),
    ComplexRoots(Vec<Complex<N>>),
    InfiniteRoots,
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

/// Finds the roots of the polynomial with terms defined by the given vector, where each element
/// is a tuple consisting of the coefficient and degree.
pub fn find_roots<N>(poly: &dyn GenericPolynomial<N>) -> Roots<N> where
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
        _ => unimplemented!("A general root finding algorithm has not been implemented."),
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
    use rustnomial::roots::find_roots;
    use ::{Roots, Polynomial};
    use ::{Monomial, LinearBinomial};

    #[test]
    fn test_roots_empty() {
        let p = Polynomial::<i32>::zero();
        assert_eq!(Roots::InfiniteRoots, find_roots(&p));
    }

    #[test]
    fn test_roots_constant() {
        let p = Monomial::new(1i32, 0);
        assert_eq!(Roots::NoRoots, find_roots(&p));
    }

    #[test]
    fn test_roots_binomial() {
        let p = LinearBinomial::new([1i32, 2]);
        assert_eq!(Roots::RealRoots(vec![-2]), find_roots(&p));
    }
}