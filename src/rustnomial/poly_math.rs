#[macro_export]
macro_rules! poly_add {
    ($lhs:expr, $rhs:expr) => {
        {
            let mut sink = Polynomial::zero();
            poly_add!($lhs, $rhs, sink)
        }
    };

    ($lhs:expr, $rhs:expr, $sink:expr) => {
        {
        for (coeff, deg) in $lhs.term_iter() {
            $sink.add_term(coeff, deg);
        }

        for (coeff, deg) in $rhs.term_iter() {
            $sink.add_term(coeff, deg);
        }

        $sink
        }
    };
}

#[macro_export]
macro_rules! poly_mul {
    ($lhs:expr, $rhs:expr) => {
        {
            let mut sink = Polynomial::zero();
            poly_mul!($lhs, $rhs, sink)
        }
    };

    ($lhs:expr, $rhs:expr, $sink:expr) => {
        {
        for (coeff_a, deg_a) in $lhs.term_iter() {
            for (coeff_b, deg_b) in $rhs.term_iter() {
                $sink.add_term(coeff_a * coeff_b, deg_a + deg_b);
            }
        }

        $sink
        }
    };
}

#[macro_export]
macro_rules! poly_sub {
    ($lhs:expr, $rhs:expr) => {
        {
            let mut sink = Polynomial::zero();
            poly_sub!($lhs, $rhs, sink)
        }
    };

    ($lhs:expr, $rhs:expr, $sink:expr) => {
        {
        for (coeff, deg) in $lhs.term_iter() {
            $sink.add_term(coeff, deg);
        }

        for (coeff, deg) in $rhs.term_iter() {
            $sink.add_term(-coeff, deg);
        }

        $sink
        }
    };
}
