use alloc::vec::Vec;

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, Neg, Sub, SubAssign};

use num::{Complex, One, Zero};
use roots::find_roots_sturm;

use crate::numerics::{AbsSqrt, Cbrt, IsPositive, PowUsize};
use crate::polynomial::polynomial::{first_nonzero_index, first_term};
use crate::{Degree, SizedPolynomial, Term};

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

pub(crate) fn discriminant_trinomial<N>(a: N, b: N, c: N) -> N
where
    N: Copy + Mul<Output = N> + Sub<Output = N> + From<u8>,
{
    b * b - a * c * N::from(4)
}

pub(crate) fn trinomial_roots<N>(a: N, b: N, c: N) -> Roots<N>
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
        return Roots::TwoRealRoots(b, b);
    }

    let sqrt = discriminant.abs_sqrt() / a;
    if discriminant.is_positive() {
        Roots::TwoRealRoots(b + sqrt, b - sqrt)
    } else {
        Roots::TwoComplexRoots(Complex::new(b, sqrt), Complex::new(b, -sqrt))
    }
}

#[allow(clippy::many_single_char_names)]
pub(crate) fn cubic_roots<N>(a: N, b: N, c: N, d: N) -> Roots<N>
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

fn div<
    N: Zero + Copy + Neg<Output = N> + AddAssign + SubAssign + Mul<Output = N> + Div<Output = N> + One,
>(
    values: &mut Vec<N>,
    root: N,
) -> Vec<N> {
    let zero = N::zero();
    let _rhs_first = N::one();

    let (mut coeff, mut self_degree) = match first_term(&values) {
        Term::ZeroTerm => return vec![],
        Term::Term(_, 1) => return vec![],
        Term::Term(coeff, degree) => (coeff, degree),
    };

    let mut div = vec![zero; self_degree];
    let offset = self_degree;

    while self_degree >= 1 {
        let scale = coeff / _rhs_first;
        let loc = values.len() - self_degree - 1;
        values[loc] -= _rhs_first * scale;
        values[loc + 1] += root * scale;
        div[offset - self_degree] = scale;
        match first_term(&values) {
            Term::ZeroTerm => break,
            Term::Term(coeffx, degree) => {
                coeff = coeffx;
                self_degree = degree;
            }
        }
    }
    div
}

fn normalize<N: Zero + Copy + DivAssign>(values: &mut Vec<N>) {
    let f_i = first_nonzero_index(values);
    if f_i == values.len() {
        return;
    }
    let first = values[f_i];
    for val in values[f_i..].iter_mut() {
        *val /= first;
    }
}

fn eval<S: SizedPolynomial<f64>>(poly: &S, point: f64) -> f64 {
    let mut sum = 0f64;
    for (val, degree) in poly.term_iter() {
        sum += val * point.upow(degree);
    }
    sum
}

/// Finds the roots of the polynomial with terms defined by the given vector, where each element
/// is a tuple consisting of the coefficient and degree. Order is not guaranteed.
pub(crate) fn find_roots<S: SizedPolynomial<f64>>(poly: &S) -> Roots<f64> {
    match poly.term_iter().collect::<Vec<(f64, usize)>>().as_slice() {
        [] => Roots::InfiniteRoots,
        [(_, 0)] => Roots::NoRoots,
        [_] => Roots::ManyRealRoots(vec![0.]),
        [(c1, 1), (c2, 0)] => Roots::ManyRealRoots(vec![-*c2 / *c1]),
        [(a, 2), one_or_more @ ..] => {
            let (b, c) = match one_or_more {
                [] => (0., 0.),
                [(xc, 0)] => (0., *xc),
                [(xb, 1)] => (*xb, 0.),
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
                [] => (0., 0., 0.),
                [(xd, 0)] => (0., 0., *xd),
                [(xc, 1)] => (0., *xc, 0.),
                [(xc, 1), (xd, 0)] => (0., *xc, *xd),
                [(xb, 2)] => (*xb, 0., 0.),
                [(xb, 2), (xd, 0)] => (*xb, 0., *xd),
                [(xb, 2), (xc, 1)] => (*xb, *xc, 0.),
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
            // NOTE: According to
            // https://en.wikipedia.org/wiki/Geometrical_properties_of_polynomial_roots
            // the largest root can be no larger than the largest coefficient divided by the
            // coefficient of the degree 0 term (assuming it isn't zero - but in that case,
            // we can just divide the polynomial by x).
            let (leading, degree) = vals[0];
            let mut values = vec![0f64; degree + 1];
            let mut nvalues = vec![0f64; degree + 1];

            nvalues[0] = leading;
            for (val, val_deg) in vals[1..].iter() {
                values[degree - val_deg] = *val / leading;
                nvalues[degree - val_deg] = *val;
            }

            let mut roots = vec![];
            loop {
                let temp_roots: Vec<f64> = find_roots_sturm(&values[1..], &mut 1e-8f64)
                    .into_iter()
                    .filter_map(Result::ok)
                    .collect();

                if temp_roots.is_empty() {
                    match poly.degree() {
                        Degree::Num(x) => {
                            if x == temp_roots.len() {
                                return Roots::ManyRealRoots(roots);
                            }
                        }
                        _ => unreachable!("Polynomial should not be zero in this stage."),
                    }
                    return Roots::OnlyRealRoots(roots);
                }

                for root in temp_roots {
                    let root = {
                        let x = root.round();
                        if eval(poly, x).abs() < eval(poly, root).abs() {
                            x
                        } else {
                            root
                        }
                    };
                    roots.push(root);
                    nvalues = div(&mut nvalues, root);
                }

                if nvalues.is_empty() {
                    return Roots::ManyRealRoots(roots);
                }
                normalize(&mut nvalues);
                let leading = nvalues[0];
                values = nvalues
                    .iter()
                    .map(|&val| val / leading)
                    .collect::<Vec<f64>>();
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::polynomial::find_roots::{cubic_roots, find_roots};
    use crate::{LinearBinomial, Monomial, Polynomial, Roots, SizedPolynomial};

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
    fn test_large_polynomials() {
        let p = Polynomial::new(vec![1f64, 2.]).pow(9) * Polynomial::new(vec![1f64, 3.]);
        assert_eq!(
            Roots::ManyRealRoots(vec![-3., -2., -2., -2., -2., -2., -2., -2., -2., -2.]),
            find_roots(&p)
        );
    }

    // #[test]
    // fn test_large_polynomials_fractional() {
    //     let p = Polynomial::new(vec![1f64, 2./3.]).pow(6) * Polynomial::new(vec![1f64, 3.]);
    //     assert_eq!(Roots::ManyRealRoots(vec![-3., 2./3., 2./3., 2./3., 2./3., 2./3., 2./3.]), find_roots(&p));
    // }

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
