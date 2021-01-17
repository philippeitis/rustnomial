use alloc::string::String;
use core::ops::Range;
use core::str::{CharIndices, FromStr};

use num::{One, Zero};

use crate::err::TermFromStringError;

#[derive(Debug, Clone, PartialEq)]
/// Degree is a type which represents the degree of a polynomial.
pub enum Degree {
    /// The degree of a zero-polynomial.
    NegInf,
    /// The degree of a non-zero polynomial.
    Num(usize),
}

#[derive(Debug, Clone, PartialEq)]
/// Term is a type which represents a term in a polynomial.
pub enum Term<N> {
    /// A term with coefficient zero. Has degree -inf.
    ZeroTerm,
    /// A term with non-zero coefficient and a degree.
    Term(N, usize),
}

impl<N: Zero> Term<N> {
    pub fn new(coeff: N, deg: usize) -> Term<N> {
        if coeff.is_zero() {
            Term::ZeroTerm
        } else {
            Term::Term(coeff, deg)
        }
    }
}

impl<N> FromStr for Term<N>
where
    N: Zero + One + FromStr + Copy,
{
    type Err = TermFromStringError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut num_vec = vec![];
        let mut coeff = N::one();
        let mut seen_x = false;
        let mut seen_caret = false;
        for c in s.chars() {
            match c {
                // Allow mixing commas, underscores, and spaces in separating numbers:
                // eg. 123 456 789 x ^ 123 456 is ok
                // eg. 123_456_789
                // Only +, - considered separators.
                ' ' | ',' | '_' => {}
                // Assume that number parser handles +, -, .'s if they shouldn't
                // be there.
                '+' | '-' | '.' | '0'..='9' => num_vec.push(c),
                'x' => {
                    if seen_x {
                        return Err(TermFromStringError::MoreThanOneX);
                    }
                    if !num_vec.is_empty() {
                        if num_vec.len() == 1 && (num_vec[0] == '-' || num_vec[0] == '+') {
                            num_vec.push('1')
                        }
                        let str: String = num_vec.iter().collect();
                        num_vec.clear();
                        coeff = match N::from_str(&str) {
                            Ok(val) => val,
                            Err(_) => {
                                return Err(TermFromStringError::CoeffCouldNotBeParsed);
                            }
                        };
                    }
                    seen_x = true;
                }
                '^' => {
                    if seen_caret {
                        return Err(TermFromStringError::MultipleCarets);
                    }
                    if !seen_x {
                        return Err(TermFromStringError::CaretWithoutXInFront);
                    }
                    seen_caret = true;
                }
                _ => {
                    return Err(TermFromStringError::UnexpectedChar(c));
                }
            }
        }
        return if seen_x {
            if num_vec.is_empty() {
                if seen_caret {
                    return Err(TermFromStringError::CaretWithoutDegree);
                }
                return Ok(Term::new(coeff, 1));
            }
            let str: String = num_vec.iter().collect();
            if let Ok(degree) = usize::from_str(&str) {
                Ok(Term::new(coeff, degree))
            } else {
                Err(TermFromStringError::CoeffCouldNotBeParsed)
            }
        } else {
            let str: String = num_vec.iter().collect();
            return match N::from_str(&str) {
                Ok(val) => Ok(Term::new(val, 0)),
                Err(_) => {
                    return Err(TermFromStringError::CoeffCouldNotBeParsed);
                }
            };
        };
    }
}

pub(crate) struct TermTokenizer<'a> {
    char_indices: CharIndices<'a>,
    seen_start: bool,
    start_index: usize,
    len: usize,
}

impl<'a> TermTokenizer<'a> {
    pub(crate) fn new(s: &'a str) -> Self {
        TermTokenizer {
            char_indices: s.char_indices(),
            seen_start: false,
            start_index: 0,
            len: s.len(),
        }
    }
}

impl<'a> Iterator for TermTokenizer<'a> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((index, c)) = self.char_indices.next() {
            if c == '+' || c == '-' {
                if self.seen_start {
                    let range = self.start_index..index;
                    self.start_index = index;
                    return Some(range);
                } else {
                    self.seen_start = true;
                }
            } else if !self.seen_start && !c.is_whitespace() {
                self.seen_start = true;
            }
        }

        if self.start_index != self.len {
            let range = self.start_index..self.len;
            self.start_index = self.len;
            Some(range)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use crate::Term;
    use core::str::FromStr;

    #[test]
    fn test_to_str_easy() {
        assert_eq!(Ok(Term::Term(5i32, 2)), Term::from_str("5x^2"));
    }

    #[test]
    fn test_to_str_harder() {
        assert_eq!(Ok(Term::Term(5i32, 2)), Term::from_str("+5 x ^ 2"));
    }

    #[test]
    fn test_to_str_dangling_caret() {
        assert!(Term::<i32>::from_str("+5 x ^ ").is_err());
    }

    #[test]
    fn test_to_str_multiple_carets() {
        assert!(Term::<i32>::from_str("5x^^2").is_err());
    }

    #[test]
    fn test_to_str_no_degree() {
        assert_eq!(Ok(Term::Term(5i32, 1)), Term::from_str("+5 x "));
    }

    #[test]
    fn test_to_str_no_x() {
        assert_eq!(Ok(Term::Term(5i32, 0)), Term::from_str("+5 "));
    }

    #[test]
    fn test_to_str_no_x_and_caret() {
        assert!(Term::<i32>::from_str("+5^ ").is_err());
    }

    #[test]
    fn test_to_str_negative() {
        assert_eq!(Ok(Term::Term(-1i32, 0)), Term::from_str("-1"));
    }

    #[test]
    fn x_no_coeff() {
        assert_eq!(Ok(Term::Term(1i32, 1)), Term::from_str("x"));
    }

    #[test]
    fn neg_x_no_coeff_neg() {
        assert_eq!(Ok(Term::Term(-1i32, 1)), Term::from_str("-x"));
    }

    #[test]
    fn neg_x_no_coeff_pos() {
        assert_eq!(Ok(Term::Term(1i32, 1)), Term::from_str("+x"));
    }

    #[test]
    fn x_no_coeff_float() {
        assert_eq!(Ok(Term::Term(1.0f32, 1)), Term::from_str("x"));
    }

    #[test]
    fn neg_x_no_coeff_neg_float() {
        assert_eq!(Ok(Term::Term(-1.0f32, 1)), Term::from_str("-x"));
    }

    #[test]
    fn neg_x_no_coeff_pos_float() {
        assert_eq!(Ok(Term::Term(1.0f32, 1)), Term::from_str("+x"));
    }

    #[test]
    fn term_from_zero() {
        assert_eq!(Ok(Term::ZeroTerm), Term::<i32>::from_str("000"));
    }

    #[test]
    fn sign_only() {
        assert!(Term::<i32>::from_str("+").is_err());
        assert!(Term::<i32>::from_str("-").is_err());
    }

    #[test]
    fn empty_str() {
        assert!(Term::<i32>::from_str("").is_err());
        assert!(Term::<i32>::from_str("    ").is_err());
    }
}
