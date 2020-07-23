use num::{One, Zero};
use std::str::FromStr;

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
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut char_iter = s.chars();
        let mut num_vec = vec![];
        let mut coeff = N::one();
        let mut seen_x = false;
        let mut seen_caret = false;
        while let Some(c) = char_iter.next() {
            match c {
                ' ' | ',' | '_' => {}
                '+' | '-' | '.' | '0'..='9' => num_vec.push(c),
                'x' => {
                    if seen_x {
                        return Err("More than one occurance of 'x' in str".to_string());
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
                                let err_str = format!("Coeff {} could not be parsed.", str);
                                return Err(err_str);
                            }
                        };
                    }
                    seen_x = true;
                }
                '^' => {
                    if !seen_x {
                        return Err("^ seen without x in front".to_string());
                    }
                    seen_caret = true;
                }
                _ => {
                    let err_str = format!(
                        "Unexpected char ({}) (legal characters include +, -, ., x, ^, 0..9).",
                        c
                    );
                    return Err(err_str);
                }
            }
        }
        return if seen_x {
            if num_vec.is_empty() {
                if seen_caret {
                    return Err("^ without ensuing degree!".to_string());
                }
                return Ok(Term::new(coeff, 1));
            }
            let str: String = num_vec.iter().collect();
            if let Ok(degree) = usize::from_str(&str) {
                Ok(Term::new(coeff, degree))
            } else {
                Err("degree could not be parsed".to_string())
            }
        } else {
            let str: String = num_vec.iter().collect();
            return match N::from_str(&str) {
                Ok(val) => Ok(Term::new(val, 0)),
                Err(_) => {
                    let err_str = format!("Coeff {} could not be parsed.", str);
                    Err(err_str)
                }
            };
        };
    }
}

mod test {
    use std::str::FromStr;
    use Term;

    #[test]
    fn test_to_str_easy() {
        let a = Term::Term(5, 2);
        assert_eq!(Ok(a), Term::<i32>::from_str("5x^2"));
    }

    #[test]
    fn test_to_str_harder() {
        let a = Term::Term(5, 2);
        assert_eq!(Ok(a), Term::<i32>::from_str("+5 x ^ 2"));
    }

    #[test]
    fn test_to_str_dangling_caret() {
        assert!(Term::<i32>::from_str("+5 x ^ ").is_err());
    }

    #[test]
    fn test_to_str_no_degree() {
        let a = Term::Term(5, 1);
        assert_eq!(Ok(a), Term::<i32>::from_str("+5 x "));
    }

    #[test]
    fn test_to_str_no_x() {
        let a = Term::Term(5, 0);
        assert_eq!(Ok(a), Term::<i32>::from_str("+5 "));
    }

    #[test]
    fn test_to_str_no_x_and_caret() {
        assert!(Term::<i32>::from_str("+5^ ").is_err());
    }

    #[test]
    fn test_to_str_negative() {
        let a = Term::Term(-1, 0);
        assert_eq!(Ok(a), Term::<i32>::from_str("-1"));
    }

    #[test]
    fn x_no_coeff() {
        let a = Term::Term(1, 1);
        assert_eq!(Ok(a), Term::<i32>::from_str("x"));
    }

    #[test]
    fn neg_x_no_coeff_neg() {
        let a = Term::Term(-1, 1);
        assert_eq!(Ok(a), Term::<i32>::from_str("-x"));
    }

    #[test]
    fn neg_x_no_coeff_pos() {
        let a = Term::Term(1, 1);
        assert_eq!(Ok(a), Term::<i32>::from_str("+x"));
    }

    #[test]
    fn x_no_coeff_float() {
        let a = Term::Term(1.0, 1);
        assert_eq!(Ok(a), Term::<f32>::from_str("x"));
    }

    #[test]
    fn neg_x_no_coeff_neg_float() {
        let a = Term::Term(-1.0, 1);
        assert_eq!(Ok(a), Term::<f32>::from_str("-x"));
    }

    #[test]
    fn neg_x_no_coeff_pos_float() {
        let a = Term::Term(1.0, 1);
        assert_eq!(Ok(a), Term::<f32>::from_str("+x"));
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
