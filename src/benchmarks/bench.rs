#![feature(test)]

extern crate rustnomial;

mod bench {
    extern crate test;

    use std::collections::BTreeMap;

    use self::test::{black_box, Bencher};

    use rustnomial::{
        Evaluable, FreeSizePolynomial, Polynomial, SizedPolynomial, SparsePolynomial,
    };

    #[bench]
    fn bench_init_dense(b: &mut Bencher) {
        b.iter(|| {
            Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        });
    }

    #[bench]
    fn bench_term_iter_dense(b: &mut Bencher) {
        let ap = Polynomial::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        b.iter(|| ap.ordered_term_iter().for_each(drop));
    }

    #[bench]
    fn bench_term_iter_sparse(b: &mut Bencher) {
        let ap = SparsePolynomial::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        b.iter(|| ap.ordered_term_iter().for_each(drop));
    }

    #[bench]
    fn bench_init_sparse(b: &mut Bencher) {
        b.iter(|| {
            SparsePolynomial::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        });
    }

    #[bench]
    fn bench_mul_dense(b: &mut Bencher) {
        b.iter(|| {
            let ap = Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            let bp = Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            ap * bp
        });
    }

    #[bench]
    fn bench_mul_sparse(b: &mut Bencher) {
        b.iter(|| {
            let ap = SparsePolynomial::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            let bp = SparsePolynomial::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            ap * bp
        });
    }

    #[bench]
    fn bench_scale_dense(b: &mut Bencher) {
        b.iter(|| {
            Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 5;
        });
    }

    #[bench]
    fn bench_scale_sparse(b: &mut Bencher) {
        b.iter(|| {
            SparsePolynomial::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 5;
        });
    }

    #[bench]
    fn bench_div_dense(b: &mut Bencher) {
        let a = &Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        b.iter(|| {
            a.clone() / 5;
        });
    }

    #[bench]
    fn bench_div_sparse(b: &mut Bencher) {
        let a = &SparsePolynomial::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        b.iter(|| a.clone() / 5);
    }

    #[bench]
    fn bench_degree_empty_dense(b: &mut Bencher) {
        let mut ap = black_box(Polynomial::new(vec![]));
        ap.terms = vec![0; 100000];

        b.iter(|| black_box(ap.degree()));
    }

    #[bench]
    fn bench_degree_dense(b: &mut Bencher) {
        let ap = black_box(Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        b.iter(|| black_box(ap.degree()));
    }

    #[bench]
    fn bench_degree_sparse(b: &mut Bencher) {
        let ap = black_box(SparsePolynomial::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        b.iter(|| black_box(ap.degree()));
    }

    #[bench]
    fn bench_trim_empty_dense(b: &mut Bencher) {
        let mut ap = Polynomial::zero();
        let terms = vec![0; 10000];
        b.iter(|| {
            black_box({
                ap.terms = terms.clone();
                ap.trim();
            })
        });
    }

    #[bench]
    fn bench_trim_dense(b: &mut Bencher) {
        let mut ap = Polynomial::new(vec![]);
        let terms = vec![0, 1, 2];
        b.iter(|| {
            black_box({
                ap.terms = terms.clone();
                ap.trim();
            })
        });
    }

    #[bench]
    fn bench_pow_dense(b: &mut Bencher) {
        let a = Polynomial::new(vec![1i32, 2, 3, 4, 5]);
        b.iter(|| {
            black_box({
                a.pow(37);
            })
        });
    }

    #[bench]
    fn bench_pow_sparse(b: &mut Bencher) {
        let a = SparsePolynomial::from(vec![1i32, 2, 3, 4, 5]);
        b.iter(|| {
            black_box({
                a.pow(37);
            })
        });
    }

    #[bench]
    fn bench_eval_dense(b: &mut Bencher) {
        let a = black_box(Polynomial::new(vec![
            1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.,
        ]));
        b.iter(|| black_box(a.eval(5.)));
    }

    #[bench]
    fn bench_eval_sparse(b: &mut Bencher) {
        let a = black_box(SparsePolynomial::from(vec![
            1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.,
        ]));
        b.iter(|| black_box(a.eval(5.)));
    }

    #[bench]
    fn bench_equal_sparse(b: &mut Bencher) {
        let a = SparsePolynomial::from(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let c = SparsePolynomial::from(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        b.iter(|| a == c);
    }

    #[bench]
    fn bench_equal_dense(b: &mut Bencher) {
        let a = Polynomial::new(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let c = Polynomial::new(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        b.iter(|| a == c);
    }

    #[bench]
    fn bench_add_term_dense(b: &mut Bencher) {
        b.iter(|| Polynomial::zero().add_term(1u32, 6));
    }

    #[bench]
    fn bench_from_terms_dense(b: &mut Bencher) {
        b.iter(|| Polynomial::from_terms(&[(4u32, 0), (4u32, 1), (1u32, 2), (2u32, 3)]));
    }
}
