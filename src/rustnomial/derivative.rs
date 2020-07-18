#[macro_export]
macro_rules! derivative {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            Polynomial::new(temp_vec).derivative()
        }
    };
}

pub trait Derivable<N> {
    fn derivative(&self) -> Self;
}