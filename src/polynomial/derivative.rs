#[macro_export]
macro_rules! derivative {
    ( $( $x:expr ),* ) => {
        {
            use $crate::rustnomial::{polynomial, Derivable};
            polynomial!($($x),*).derivative()
        }
    };
}

pub trait Derivable<N> {
    fn derivative(&self) -> Self;
}
