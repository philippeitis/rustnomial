use std::fmt;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TryAddError {
    DegreeOutOfBounds,
    TooManyTerms,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TermFromStringError{
    MoreThanOneX,
    CoeffCouldNotBeParsed,
    UnexpectedChar(char),
    CaretWithoutXInFront,
    CaretWithoutDegree,
    DegreeCouldNotBeParsed,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PolynomialFromStringError {
    TermFromString(TermFromStringError),
    AddingTerm(TryAddError),
    NoTermsFound,
}

impl fmt::Display for TermFromStringError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TermFromStringError::MoreThanOneX => {
                write!(f, "More than one occurance of 'x' in str")
            },
            TermFromStringError::CoeffCouldNotBeParsed => {
                write!(f, "Coeff could not be parsed.")
            },
            TermFromStringError::CaretWithoutXInFront => {
                write!(f, "^ seen without x in front")
            },
            TermFromStringError::UnexpectedChar(c) => {
                write!(f, "Unexpected char ({}) (legal characters include +, -, ., x, ^, 0..9).", c)
            },
            TermFromStringError::CaretWithoutDegree => {
                write!(f, "^ without ensuing degree!")
            },
            TermFromStringError::DegreeCouldNotBeParsed => {
                write!(f, "Degree could not be parsed")
            },
        }
    }
}
