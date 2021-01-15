#[macro_export]
macro_rules! derivative {
    ( $( $x:expr ),* ) => {
        {
            use $crate::{polynomial, Derivable};
            polynomial!($($x),*).derivative()
        }
    };
}

pub trait Derivable<N> {
    fn derivative(&self) -> Self;
}

#[cfg(test)]
mod test {
    use crate::{derivative, Polynomial};

    #[test]
    fn test_derivative_macro() {
        assert_eq!(derivative!(1, 2, 3), Polynomial::new(vec![2, 2]));
    }
}
