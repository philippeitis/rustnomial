#[derive(Debug, Clone, PartialEq)]
pub enum Degree {
    NegInf,
    Num(usize)
}

pub enum Term<N> {
    ZeroTerm,
    Term(N, usize)
}