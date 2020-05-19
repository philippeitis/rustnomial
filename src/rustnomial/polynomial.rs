//#![feature(test)]

use std::ops;
use std::fmt;

#[derive(Debug, Clone)]
struct Polynomial {
    terms: Vec<i32>,
}

struct PolynomialIterator<'a> {
    polynomial: &'a Polynomial,
    index: usize,
}

struct PolynomialDegreeIterator<'a> {
    polynomial: &'a Polynomial,
    index: usize,
    degree: usize,
}

impl Polynomial {
    fn new(terms: Vec<i32>) -> Polynomial {
        let mut first_non_zero = 0;

        for &term in terms.iter() {
            if term != 0 {
                break;
            }
            first_non_zero += 1;
        }

        Polynomial{
            terms: terms[first_non_zero..].to_vec()
        }

    }

    fn len(&self) -> usize {
        return self.terms.len();
    }

    fn degree(&self) -> usize {
        let mut potential_degree = self.len() - 1;
        let mut i = 0;

        while self.terms[i] == 0 {
            potential_degree -= 1;
            i += 1;
        }

        potential_degree
    }

    fn iter(&self) -> PolynomialIterator {
        PolynomialIterator{polynomial: self, index: self.len() - self.degree() - 1}
    }

    fn degree_iter(&self) -> PolynomialDegreeIterator {
        PolynomialDegreeIterator{
            polynomial: self,
            index: self.len() - self.degree() - 1,
            degree: self.degree()
        }
    }

}

impl ops::Sub<Polynomial> for Polynomial {
    type Output = Polynomial;

    fn sub(self, _rhs: Polynomial) -> Polynomial {
        if _rhs.len() > self.len() {
            let mut terms = _rhs.terms.clone();
            let offset = _rhs.len() - self.len();

            for index in terms[..offset].iter_mut() {
                *index = -*index;
            }

            for (index, val) in terms[offset..].iter_mut().zip(self.terms) {
                *index = val - *index;
            }
            Polynomial{terms}
        } else {
            let mut terms = self.terms.clone();
            let offset = terms.len() - _rhs.len();
            for (index, val) in terms[offset..].iter_mut().zip(_rhs.terms) {
                *index -= val;
            }
            Polynomial{terms}
        }
    }
}

impl ops::Add<Polynomial> for Polynomial {
    type Output = Polynomial;

    fn add(self, _rhs: Polynomial) -> Polynomial {
        let (mut output, small) = if _rhs.len() > self.len() {
            (_rhs.terms.clone(), self.terms)
        } else {
            (self.terms.clone(), _rhs.terms)
        };

        let offset = output.len() - small.len();

        for (index, val) in output[offset..].iter_mut().zip(small) {
            *index += val;
        }

        Polynomial {
            terms: output,
        }
    }
}

impl ops::Mul<Polynomial> for Polynomial {
    type Output = Polynomial;

    fn mul(self, _rhs: Polynomial) -> Polynomial {
        let mut terms = vec![0; _rhs.degree() + self.degree() + 1];
        let t_len = terms.len() - 1;
        for (_rhs_v, _rhs_d) in _rhs.degree_iter() {
            for (_lhs_v, _lhs_d) in self.degree_iter() {
                terms[t_len - _rhs_d - _lhs_d] += _rhs_v * _lhs_v;
            }
        }
        Polynomial{terms}
    }
}

impl ops::MulAssign<Polynomial> for Polynomial {
    fn mul_assign(&mut self, _rhs: Polynomial) {
        let mut terms = vec![0; _rhs.degree() + self.degree() + 1];
        let t_len = terms.len() - 1;
        for (_rhs_v, _rhs_d) in _rhs.degree_iter() {
            for (_lhs_v, _lhs_d) in self.degree_iter() {
                terms[t_len - _rhs_d - _lhs_d] += _rhs_v * _lhs_v;
            }
        }
        self.terms = terms;
    }
}


impl Iterator for PolynomialIterator<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.polynomial.len() {
            let ret_val = self.polynomial.terms[self.index];
            self.index += 1;
            Some(ret_val)
        } else {
            None
        }
    }
}

impl Iterator for PolynomialDegreeIterator<'_> {
    type Item = (i32, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.polynomial.len() {
            let ret_val = self.polynomial.terms[self.index];
            self.index += 1;
            let degree = self.degree;
            self.degree = if self.degree >= 1 {self.degree - 1} else {0};
            if ret_val != 0 {
                return Some((ret_val, degree));
            }
        }

        None
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for (&a, &b) in self.terms.iter().zip(other.terms.iter()) {
            if a != b {
                return false;
            }
        }

        true
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.len() == 0 {
            return write!(f, "~");
        }

        let own_degree = self.degree();
        for (term, degree) in self.degree_iter() {
            if degree == own_degree {
                write!(f, "{}", term)?;
            } else {
                let sign = if term > 0 {'+'} else {'-'};
                write!(f, " {} {}", sign, term.abs())?;
            }
            match degree {
                0 => {},
                1 => {write!(f, "x")?;},
                _ => {write!(f, "x^{}", degree)?;}
            }
        }
        write!(f, "")
    }
}


#[cfg(test)]
mod tests {
    use crate::Polynomial;
    use std::fmt::Write;
    #[test]
    fn test_add_rhs_bigger() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(a + b, c);
    }

    #[test]
    fn test_mul() {
        let a = Polynomial::new(vec![1, 2]);
        let b = a.clone();
        let c = Polynomial::new(vec![1, 4, 4]);
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_mul_assign() {
        let mut a = Polynomial::new(vec![1, 2]);
        let b = a.clone();
        a *= b;
        let c = Polynomial::new(vec![1, 4, 4]);
        assert_eq!(a, c);
    }


    #[test]
    fn test_add_lhs_bigger() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(b + a, c);
    }

    #[test]
    fn test_sub_lhs_bigger() {
        let a = Polynomial::new(vec![2, 3, 4]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 0, 0, 0]);
        assert_eq!(b - a, c);
    }

    #[test]
    fn test_sub_rhs_bigger() {
        let a = Polynomial::new(vec![2, 3, 4]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![-1, 0, 0, 0]);
        assert_eq!(a - b, c);
    }

    #[test]
    fn test_str() {
        let a = Polynomial::new(vec![-1, -2, 3]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-1x^2 - 2x + 3");
    }

    #[test]
    fn test_str_has_zeroes() {
        let a = Polynomial::new(vec![-1, -2, 0, 0, 3]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-1x^4 - 2x^3 + 3");
    }

    #[test]
    fn test_degree() {
        let a = Polynomial::new(vec![0, 0, 0, -1, -2, 3]);
        assert_eq!(a.degree(), 2);
    }

    #[test]
    fn test_iter() {
        let mut num_iters = 0;
        let a = Polynomial::new(vec![0, 0, 0, -1, -2, 3]);
        let b=  vec![-1, -2, 3];
        for (a_val, b_val) in a.iter().zip(b) {
            num_iters += 1;
            assert_eq!(a_val, b_val);
        }

        assert_eq!(num_iters, 3);
    }

}

//mod bench {
//    use super::*;
//    extern crate test;
//    use test::Bencher;
//
//    #[bench]
//    fn bench_mul(b: &mut Bencher) {
//        b.iter(|| {
//            let ap = Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
//            let bp = Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
//            let c = ap * bp;
//        });
//    }
//}