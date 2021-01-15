use std::str::FromStr;

use num::{One, Zero};

use crate::err::TermFromStringError;

#[derive(Debug, Clone, PartialEq)]
pub enum Degree {
    NegInf,
    Num(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Term<N> {
    ZeroTerm,
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

pub(crate) struct TermTokenizer {
    chars: Vec<char>,
    start_index: usize,
    end_index: usize,
}

impl TermTokenizer {
    pub(crate) fn new(s: &str) -> Self {
        let chars: Vec<char> = s.chars().collect();
        let start_index = match chars.iter().position(|&x| !x.is_whitespace()) {
            Some(pos) => pos,
            None => chars.len(),
        };
        TermTokenizer {
            chars,
            start_index,
            end_index: start_index + 1,
        }
    }
}

impl Iterator for TermTokenizer {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_index >= self.chars.len() {
            return None;
        }

        let mut end_index = self.end_index;
        while end_index < self.chars.len() {
            if self.chars[end_index] == '+' || self.chars[end_index] == '-' {
                let s: String = self.chars[self.start_index..end_index].iter().collect();
                self.start_index = end_index;
                self.end_index = end_index + 1;
                return Some(s);
            }
            end_index += 1;
        }
        let s: String = self.chars[self.start_index..end_index].iter().collect();
        self.start_index = end_index;
        Some(s)
    }
}

#[cfg(test)]
mod test {
    use crate::Term;
    use std::str::FromStr;

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
