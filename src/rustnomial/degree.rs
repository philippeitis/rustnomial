#[derive(Debug, Clone, PartialEq)]
pub enum Degree {
    NegInf,
    Num(usize)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Term<N> {
    ZeroTerm,
    Term(N, usize)
}