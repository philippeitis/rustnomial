use std::ops;
use std::ops::{Mul, AddAssign, Add, MulAssign, DivAssign, Div, SubAssign, Neg, Sub};
use std::fmt::{Error, Display};

//trait GenericPolynomial {
//    fn add_term(&self, term: &Add, deg: usize) {
//    }
//
//    fn scale(&mut self, _rhs: &Mul) {
//        let mut result = Polynomial::new(vec![]);
//        for (term, deg) in self.degree_iter() {
//            result.add_term(term * _rhs, deg);
//        }
//    }
//
//
//    fn mul(&self, _rhs: &GenericPolynomial) -> Polynomial {
//        let mut result = Polynomial::new(vec![]);
//        for (rterm, rdeg) in self.degree_iter() {
//            for (lterm, ldeg) in _rhs.degree_iter() {
//                result.add_term(rterm * lterm, rdeg, ldeg);
//            }
//        }
//        result
//    }
//
//    fn iter(&self) -> PolynomialDegreeIterator {
//    }
//
//    fn degree_iter(&self) -> PolynomialDegreeIterator {
//    }
//}

#[derive(Debug, Clone)]
pub struct Polynomial<N>
    where N: AddAssign + Add + Mul + MulAssign {
    pub terms: Vec<N>,
}

#[derive(Debug, Clone)]
pub struct Integral<N>
    where N: AddAssign + Add + Mul + MulAssign {
    pub polynomial: Polynomial<N>,
}

pub struct PolynomialIterator<'a, N>
    where N: AddAssign + Add + Mul + MulAssign {
    polynomial: &'a Polynomial<N>,
    index: usize,
}

impl<N> Iterator for PolynomialIterator<'_, N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Item = N;

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

pub struct PolynomialDegreeIterator<'a, N>
    where N: AddAssign + Add + Mul + MulAssign + Copy {
    polynomial: &'a Polynomial<N>,
    index: usize,
    degree: usize,
}

impl<N> Iterator for PolynomialDegreeIterator<'_, N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Item = (N, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.polynomial.len() {
            let ret_val = self.polynomial.terms[self.index];
            self.index += 1;
            let degree = self.degree;
            self.degree = if self.degree >= 1 {self.degree - 1} else {0};
            if ret_val != N::from(0) {
                return Some((ret_val, degree));
            }
        }

        None
    }
}

fn first_nonzero_index<N>(terms: &Vec<N>) -> usize
    where N: PartialEq + From<i8> + Copy {
    let mut ind = 0;
    let zero = N::from(0);
    while (ind < terms.len()) && (terms[ind] == zero) {
        ind += 1;
    }
    ind
}

fn vec_mul<N>(_lhs: &Vec<N>, _rhs: &Vec<N>) -> Vec<N>
    where N: Mul<Output=N> + AddAssign + Copy + From<i8> + PartialEq {
    let _rhs_ind = first_nonzero_index(&_rhs);
    let _lhs = &_lhs[first_nonzero_index(&_lhs)..];
    let zero = N::from(0);
    let mut terms = vec![zero; _rhs.len() + _lhs.len() - 1 - _rhs_ind];
    for (index, &rterm) in _rhs[_rhs_ind..].iter().enumerate() {
        if rterm == zero {
            continue;
        }
        for (&lterm, term) in _lhs.iter().zip(terms[index..].iter_mut()) {
            *term += rterm * lterm;
        }
    }
    terms
}

fn degree<N>(poly_vec: &Vec<N>) -> usize
    where N: PartialEq + From<i8> + Copy {
    let zero = N::from(0);
    for (ind, &val) in poly_vec.iter().enumerate() {
        if val != zero {
            return poly_vec.len() - ind - 1;
        }
    }

    0
}

fn degree_and_first_val<N>(poly_vec: &Vec<N>) -> (usize, N)
    where N: PartialEq + From<i8> + Copy {
    let zero = N::from(0);
    for (ind, &val) in poly_vec.iter().enumerate() {
        if val != zero {
            return (poly_vec.len() - ind - 1, val);
        }
    }

    (0, zero)
}


impl<N> Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    pub fn new(terms: Vec<N>) -> Polynomial<N> {
        let first_non_zero = first_nonzero_index(&terms);
        Polynomial{
            terms: if first_non_zero != 0 {
                terms[first_non_zero..].to_vec()
            } else {
                terms
            }
        }

    }

    fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn degree(&self) -> usize {
        degree(&self.terms)
    }

    pub fn trim(&mut self) {
        let ind = first_nonzero_index(&self.terms);
        if ind != 0 {
            self.terms = self.terms[ind..].to_vec();
        };
    }

    pub fn is_zero(&self) -> bool {
        if self.degree() == 0 {
            return match self.terms.last() {
                None => {
                    true
                }
                Some(&x) => {
                    x == N::from(0)
                }
            }
        }

        false
    }

    pub fn iter(&self) -> PolynomialIterator<N> {
        PolynomialIterator{
            polynomial: self,
            index: if self.len() == 0 {
                0
            } else {
                self.len() - self.degree() - 1
            }
        }
    }

    pub fn degree_iter(&self) -> PolynomialDegreeIterator<N> {
        PolynomialDegreeIterator{
            polynomial: self,
            index: if self.len() == 0 {
                0
            } else {
                self.len() - self.degree() - 1
            },
            degree: self.degree()
        }
    }

    pub fn eval(&self, point: N) -> N {
        let mut sum = N::from(0);
        let mut exp = N::from(1);
        for &val in self.terms.iter().rev() {
            sum += exp * val;
            exp *= point;
        }
        sum
    }

    pub fn derivative(&self) -> Polynomial<N> {
        // TODO: Fix for degrees of arbitrary size.
        let index = first_nonzero_index(&self.terms);
        let mut degree = (self.len() - index) as i8;
        let mut terms = self.terms[0..self.len()-1].to_vec();
        for term in terms.iter_mut() {
            degree -= 1;
            *term *= N::from(degree);
        }
        Polynomial { terms }
    }

    pub fn integral(&self) -> Integral<N> {
        let index = first_nonzero_index(&self.terms);
        let mut degree = (self.len() - index + 1) as i8;
        let mut terms = self.terms.clone();
        for term in terms.iter_mut() {
            degree -= 1;
            *term /= N::from(degree);
        }
        terms.push(N::from(0));
        Integral {
            polynomial: Polynomial { terms }
        }
    }

    fn borrow_mul(&self, _rhs: &Polynomial<N>) -> Polynomial<N> {
        Polynomial{terms: vec_mul(&self.terms, &_rhs.terms)}
    }

    pub fn exp(&self, exp: usize) -> Polynomial<N> {
        if exp == 0 {
            Polynomial{terms: vec![N::from(1); 1]}
        } else if exp == 1 {
            Polynomial::new(self.terms.clone())
        } else if exp == 2 {
            self.borrow_mul(self)
        } else if exp % 2 == 0 {
            self.exp(exp / 2).exp(2)
        } else {
            self.borrow_mul(&self.exp(exp - 1))
        }
    }

    pub fn div_mod(&self, _rhs: &Polynomial<N>) -> Result<(Polynomial<N>, Polynomial<N>), &'static str> {
        // fn vec_sub(_lhs: &mut Vec<i32>, _rhs: Vec<i32>) {
        //     for (_lhs_t, _rhs_t) in _lhs[_lhs.len() - _rhs.len()..].iter_mut().zip(_rhs) {
        //         *_lhs_t -= _rhs_t;
        //     }
        // }

        let zero = N::from(0);
        fn vec_sub_w_scale<N>(_lhs: &mut Vec<N>, _lhs_degree: usize, _rhs: &Vec<N>, _rhs_deg: usize, _rhs_scale: N)
            where N: Copy + Mul<Output=N> + SubAssign {
            let loc = _lhs.len() - _lhs_degree - 1;
            for (_lhs_t, _rhs_t) in _lhs[loc..].iter_mut().zip(_rhs) {
                *_lhs_t -= (*_rhs_t) * _rhs_scale;
            }
        }

        let (_rhs_deg, _rhs_first) = degree_and_first_val(&_rhs.terms);

        if _rhs_deg == 0 {
            match _rhs.terms.last() {
                None => {
                return Err("Can't divide by 0.");
                }
                Some(&x) => {
                    if x == zero {
                        return Err("Can't divide by 0.");
                    }
                }
            }
        }

        let (mut self_degree, mut term) = degree_and_first_val(&self.terms);

        if self_degree < _rhs_deg {
            let zero_vec = vec![zero; 1];
            return Ok((Polynomial::new(zero_vec), Polynomial::new(self.terms.clone())));
        }

        let mut remainder = self.terms.clone();
        let offset = self_degree - _rhs_deg;
        let mut div = vec![zero; offset + 1];

        while self_degree >= _rhs_deg {
            let scale = term / _rhs_first;
            vec_sub_w_scale(&mut remainder, self_degree, &_rhs.terms, _rhs_deg, scale);
            div[offset - (self_degree - _rhs_deg)] = scale;
            let (sd, t) = degree_and_first_val(&remainder);
            self_degree = sd;
            term = t;
        }

        Ok((Polynomial::new(div), Polynomial::new(remainder)))
    }
}

impl<N> Integral<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    pub fn eval(&self, start: N, end: N) -> N {
        self.polynomial.eval(end) - self.polynomial.eval(start)
    }

    pub fn replace_c(&self, c: N) -> Polynomial<N> {
        Polynomial{terms: {
            let mut terms = self.polynomial.terms.clone();
            let last_ind = terms.len() - 1;
            terms[last_ind] = c;
            terms
        }}
    }
}

impl<N> PartialEq for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    fn eq(&self, other: &Self) -> bool {
        if self.degree() != other.degree() {
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

impl<N> Display for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut iter = self.degree_iter();
        let one = N::from(1);
        let zero = N::from(0);
        match iter.next() {
            None => {
                return write!(f, "0");
            }

            Some((term, degree)) => {
                if term == -one {
                    write!(f, "-")?;
                } else if (term != one) || (degree == 0) {
                    write!(f, "{}", if term < zero {-term} else {term})?;
                }

                match degree {
                    0 => {},
                    1 => {write!(f, "x")?;},
                    _ => {write!(f, "x^{}", degree)?;}
                }
            }
        }


        for (term, degree) in iter {
            if term > zero {
                write!(f, " + ")?;
            } else {
                write!(f, " - ")?;
            }

            let term = if term < zero {-term} else {term};

            if (term != one) || (degree == 0) {
                write!(f, "{}", term)?;
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

impl<N> fmt::Display for Integral<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.polynomial.terms.len() == 0 {
            return write!(f, "C");
        }

        match self.polynomial.fmt(f) {
            Ok(_) => {
                write!(f, " + C")
            },

            Err(e) => Err(e)
        }
    }
}

impl<N> ops::Neg for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display{
    type Output = Polynomial<N>;

    fn neg(self) -> Polynomial<N> {
        Polynomial::new(self.terms.iter().map(|&x| -x).collect())
    }
}

impl<N> ops::Sub<Polynomial<N>> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Output = Polynomial<N>;

    fn sub(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        if _rhs.len() > self.len() {
            let mut terms = _rhs.terms.clone();
            let offset = _rhs.len() - self.len();

            for index in terms[..offset].iter_mut() {
                *index = -*index;
            }

            for (index, val) in terms[offset..].iter_mut().zip(self.terms) {
                *index = val - *index;
            }
            Polynomial::new(terms)
        } else {
            let mut terms = self.terms.clone();
            let offset = terms.len() - _rhs.len();
            for (index, val) in terms[offset..].iter_mut().zip(_rhs.terms) {
                *index -= val;
            }
            Polynomial::new(terms)
        }
    }
}

impl<N> ops::SubAssign<Polynomial<N>> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    fn sub_assign(&mut self, _rhs: Polynomial<N>) {
        if _rhs.len() > self.len() {
            let mut terms = _rhs.terms.clone();
            let offset = _rhs.len() - self.len();

            for index in terms[..offset].iter_mut() {
                *index = -*index;
            }

            for (index, &val) in terms[offset..].iter_mut().zip(&self.terms) {
                *index = val - *index;
            }
            self.terms = terms;
        } else {
            let offset = self.len() - _rhs.len();
            for (index, val) in self.terms[offset..].iter_mut().zip(_rhs.terms) {
                *index -= val;
            }
        }
    }
}

impl<N> ops::Add<Polynomial<N>> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Output = Polynomial<N>;

    fn add(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        let (mut terms, small) = if _rhs.len() > self.len() {
            (_rhs.terms.clone(), &self.terms)
        } else {
            (self.terms.clone(), &_rhs.terms)
        };

        let offset = terms.len() - small.len();

        for (index, &val) in terms[offset..].iter_mut().zip(small) {
            *index += val;
        }

        Polynomial::new(terms)
    }
}

impl<N> ops::AddAssign<Polynomial<N>> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    fn add_assign(&mut self, _rhs: Polynomial<N>) {
        if _rhs.len() > self.len() {
            let offset = _rhs.len() - self.len();
            let mut terms = _rhs.terms.clone();
            for (index, &val) in terms[offset..].iter_mut().zip(&self.terms) {
                *index += val;
            }
            self.terms = terms;
        } else {
            let offset = self.len() - _rhs.len();
            for (index, val) in self.terms[offset..].iter_mut().zip(_rhs.terms) {
                *index += val;
            }
        }
    }
}

impl<N> ops::Mul<Polynomial<N>> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Output = Polynomial<N>;

    fn mul(self, _rhs: Polynomial<N>) -> Polynomial<N> {
        Polynomial{terms: vec_mul(&self.terms, &_rhs.terms)}
    }
}

impl<N> ops::MulAssign<Polynomial<N>> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    fn mul_assign(&mut self, _rhs: Polynomial<N>) {
        self.terms = vec_mul(&self.terms, &_rhs.terms);
    }
}


impl<N> ops::Mul<&Polynomial<N>> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Output = Polynomial<N>;

    fn mul(self, _rhs: &Polynomial<N>) -> Polynomial<N> {
        Polynomial::new(vec_mul(&self.terms, &_rhs.terms))
    }
}

impl<N> ops::MulAssign<&Polynomial<N>> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    fn mul_assign(&mut self, _rhs: &Polynomial<N>) {
        self.terms = vec_mul(&self.terms, &_rhs.terms);
    }
}

impl<N> ops::Mul<N> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Output = Polynomial<N>;

    fn mul(self, _rhs: N) -> Polynomial<N> {
        Polynomial::new(self.terms.iter().map(|&x| x * _rhs).collect())
    }
}

impl<N> ops::MulAssign<N> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    fn mul_assign(&mut self, _rhs: N) {
        for p in self.terms.iter_mut() {
            *p *= _rhs;
        }
    }
}

impl<N> ops::Div<N> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Output = Polynomial<N>;

    fn div(self, _rhs: N) -> Polynomial<N> {
        Polynomial::new(self.terms.iter().map(|&x| x / _rhs).collect())
    }
}

impl<N> ops::DivAssign<N> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    fn div_assign(&mut self, _rhs: N) {
        for p in self.terms.iter_mut() {
            *p /= _rhs;
        }
    }
}

impl<N> ops::Shl<i32> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Output = Polynomial<N>;

    fn shl(self, _rhs: i32) -> Polynomial<N> {
        if _rhs < 0 {
            self >> -_rhs
        } else {
            let index = first_nonzero_index(&self.terms);
            let mut terms = self.terms[index..].to_vec();
            terms.extend(vec![N::from(0); _rhs as usize]);
            Polynomial{terms}
        }
    }
}

impl<N> ops::ShlAssign<i32> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    fn shl_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self >>= -_rhs;
        } else {
            self.terms.extend(vec![N::from(0); _rhs as usize]);
        }
    }
}

impl<N> ops::Shr<i32> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    type Output = Polynomial<N>;

    fn shr(self, _rhs: i32) -> Polynomial<N> {
        if _rhs < 0 {
            self << -_rhs
        } else {
            let index = first_nonzero_index(&self.terms);
            Polynomial{terms: self.terms[index..self.terms.len() - (_rhs as usize)].to_vec()}
        }
    }
}

impl<N> ops::ShrAssign<i32> for Polynomial<N>
    where N: From<i8> + Copy + Mul<Output=N> + MulAssign + Add<Output=N> + Neg<Output=N> + AddAssign + Sub<Output=N> + SubAssign + Div<Output=N> + DivAssign  + PartialEq + PartialOrd + Display {
    fn shr_assign(&mut self, _rhs: i32) {
        if _rhs < 0 {
            *self <<= -_rhs;
        } else {
            self.terms = self.terms[..self.terms.len() - (_rhs as usize)].to_vec();
        }
    }
}

/// TODO:
/// modulo floordiv
#[cfg(test)]
mod tests {
    use std::fmt::Write;
    use super::Polynomial;

    #[test]
    fn test_eval() {
        let a = Polynomial::new(vec![1, 2, 3]);
        assert_eq!(a.eval(5), 25 + 2 * 5 + 3);
    }

    #[test]
    fn test_derivative() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![2, 2]);
        assert_eq!(a.derivative(), b);

        let a = Polynomial::new(vec![0, 1, 2, 3]);
        assert_eq!(a.derivative(), b);

        let a = Polynomial::new(vec![1, 2, 3, 4]);
        let b = Polynomial::new(vec![3, 4, 3]);
        assert_eq!(a.derivative(), b);

    }

    #[test]
    fn test_integral() {
        let a = Polynomial::new(vec![3, 2, 1]);
        let b = Polynomial::new(vec![1, 1, 1, 0]);
        assert_eq!(a.integral().polynomial, b);
    }

    #[test]
    fn test_integral_eval() {
        let a = Polynomial::new(vec![3, 2, 1]);
        assert_eq!(a.integral().eval(0, 1), 3);
    }

    #[test]
    fn test_integral_const_substitute() {
        let a = Polynomial::new(vec![3, 2, 1]);
        let b = Polynomial::new(vec![1, 1, 1, 5]);
        assert_eq!(a.integral().replace_c(5), b);
    }


    #[test]
    fn test_add_lhs_bigger() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(b + a, c);
    }

    #[test]
    fn test_add_rhs_bigger() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(a + b, c);
    }

    #[test]
    fn test_add_lhs_bigger_assign() {
        let a = Polynomial::new(vec![1, 2, 3]);
        let mut b = Polynomial::new(vec![1, 2, 3, 4]);
        b += a;
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(b, c);
    }

    #[test]
    fn test_add_rhs_bigger_assign() {
        let mut a = Polynomial::new(vec![1, 2, 3]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        a += b;
        let c = Polynomial::new(vec![1, 3, 5, 7]);
        assert_eq!(a, c);
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
    fn test_sub_lhs_bigger_assign() {
        let a = Polynomial::new(vec![2, 3, 4]);
        let mut b = Polynomial::new(vec![1, 2, 3, 4]);
        b -= a;
        let c = Polynomial::new(vec![1, 0, 0, 0]);
        assert_eq!(b, c);
    }

    #[test]
    fn test_sub_rhs_bigger_assign() {
        let mut a = Polynomial::new(vec![2, 3, 4]);
        let b = Polynomial::new(vec![1, 2, 3, 4]);
        a -= b;
        let c = Polynomial::new(vec![-1, 0, 0, 0]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_negate() {
        let a = Polynomial::new(vec![1, 2, 3, 0, -5]);
        let c = Polynomial::new(vec![-1, -2, -3, 0, 5]);
        assert_eq!(-a, c);
    }

    #[test]
    fn test_mul_poly() {
        let a = Polynomial::new(vec![1, 2]);
        let b = a.clone();
        let c = Polynomial::new(vec![1, 4, 4]);
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_mul_assign_poly() {
        let mut a = Polynomial::new(vec![1, 2]);
        let b = a.clone();
        a *= b;
        let c = Polynomial::new(vec![1, 4, 4]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_mul_i32() {
        let a = Polynomial::new(vec![1, 2]);
        let c = Polynomial::new(vec![10, 20]);
        assert_eq!(a * 10, c);
    }

    #[test]
    fn test_mul_assign_i32() {
        let mut a = Polynomial::new(vec![1, 2]);
        a *= 10;
        let c = Polynomial::new(vec![10, 20]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shl_pos() {
        let a = Polynomial::new(vec![1, 2]);
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a << 5, c);
    }

    #[test]
    fn test_shl_assign_pos() {
        let mut a = Polynomial::new(vec![1, 2]);
        a <<= 5;
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shl_neg() {
        let a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(a << -5, c);
    }

    #[test]
    fn test_shl_assign_neg() {
        let mut a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        a <<= -5;
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shr_pos() {
        let a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(a >> 5, c);
    }

    #[test]
    fn test_shr_assign_pos() {
        let mut a = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        a >>= 5;
        let c = Polynomial::new(vec![1, 2]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_shr_neg() {
        let a = Polynomial::new(vec![1, 2]);
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a >> -5, c);
    }

    #[test]
    fn test_shr_assign_neg() {
        let mut a = Polynomial::new(vec![1, 2]);
        a >>= -5;
        let c = Polynomial::new(vec![1, 2, 0, 0, 0, 0, 0]);
        assert_eq!(a, c);
    }

    #[test]
    fn test_exp() {
        let a = &Polynomial::new(vec![1, 2]);
        let mut b = a.clone();
        assert_eq!(Polynomial::new(vec![1]), a.exp(0));
        for i in 1..10 {
            assert_eq!(b, a.exp(i));
            b *= a;
        }
    }

    #[test]
    fn test_polynomial_str_all_zeroes() {
        let a = Polynomial::new(vec![0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a: Polynomial<i8> = Polynomial::new(vec![]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a = Polynomial::new(vec![0, 0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a: Polynomial<i8> = Polynomial{terms: vec![]};
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a = Polynomial{terms: vec![0]};
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

        let a = Polynomial{terms: vec![0, 0]};
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "0");

    }

    #[test]
    fn test_polynomial_str() {
        let a = Polynomial::new(vec![-1, -2, 3]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^2 - 2x + 3");
    }

    #[test]
    fn test_polynomial_str_has_zeroes() {
        let a = Polynomial::new(vec![-1, -2, 0, 0, 3]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^4 - 2x^3 + 3");
    }

    #[test]
    fn test_polynomial_str_has_ones() {
        let a = Polynomial::new(vec![-1, -1, -1, 0]);
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^3 - x^2 - x");
    }

    #[test]
    fn test_integral_str() {
        let a = Polynomial::new(vec![-3, -2, 1]).integral();
        let mut a_str = String::new();
        write!(&mut a_str, "{}", a).unwrap();
        assert_eq!(a_str, "-x^3 - x^2 + x + C");
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

mod bench {
    extern crate test;
    use self::test::Bencher;
    use self::test::black_box;
    use super::Polynomial;

    #[bench]
    fn bench_mul(b: &mut Bencher) {
        b.iter(|| {
            let ap = Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            let bp = Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            ap * bp
        });
    }

    #[bench]
    fn bench_degree(b: &mut Bencher) {
        let mut ap = black_box(Polynomial::new(vec![]));
        ap.terms = vec![0; 100000];

        b.iter(|| black_box(ap.degree()));
    }

    #[bench]
    fn bench_trim(b: &mut Bencher) {
        let mut ap = Polynomial::new(vec![]);
        let terms = vec![0; 10000];
        b.iter(|| black_box({
            ap.terms = terms.clone();
            ap.trim()
        }));
    }
}