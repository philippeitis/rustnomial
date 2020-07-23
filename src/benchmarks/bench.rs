#![feature(test)]

extern crate rustnomial;

mod bench {
    extern crate test;
    use self::test::{Bencher, black_box};
    use rustnomial::{Evaluable, Polynomial, SparsePolynomial};

    #[bench]
    fn bench_init(b: &mut Bencher) {
        b.iter(|| {
            Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        });
    }

    #[bench]
    fn bench_inits(b: &mut Bencher) {
        b.iter(|| {
            SparsePolynomial::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        });
    }

    #[bench]
    fn bench_mul(b: &mut Bencher) {
        b.iter(|| {
            let ap = Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            let bp = Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            ap * bp
        });
    }

    #[bench]
    fn bench_muls(b: &mut Bencher) {
        b.iter(|| {
            let ap = SparsePolynomial::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            let bp = SparsePolynomial::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            ap * bp
        });
    }

    #[bench]
    fn bench_scale(b: &mut Bencher) {
        b.iter(|| {
            Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 5;
        });
    }

    #[bench]
    fn bench_scales(b: &mut Bencher) {
        b.iter(|| {
            SparsePolynomial::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 5;
        });
    }

    #[bench]
    fn bench_div(b: &mut Bencher) {
        b.iter(|| {
            Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) / 5;
        });
    }

    #[bench]
    fn bench_divs(b: &mut Bencher) {
        b.iter(|| {
            SparsePolynomial::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) / 5;
        });
    }

    #[bench]
    fn bench_degree_empty(b: &mut Bencher) {
        let mut ap = black_box(Polynomial::new(vec![]));
        ap.terms = vec![0; 100000];

        b.iter(|| black_box(ap.degree()));
    }

    #[bench]
    fn bench_degree(b: &mut Bencher) {
        let ap = black_box(Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        b.iter(|| black_box(ap.degree()));
    }

    #[bench]
    fn bench_degrees(b: &mut Bencher) {
        let ap = black_box(SparsePolynomial::from_vec(vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        ]));
        b.iter(|| black_box(ap.degree()));
    }

    #[bench]
    fn bench_trim_empty(b: &mut Bencher) {
        let mut ap = Polynomial::new(vec![]);
        let terms = vec![0; 10000];
        let empty_vec = Vec::<i32>::new();
        b.iter(|| {
            black_box({
                ap.terms = terms.clone();
                ap.trim();
                assert_eq!(empty_vec, ap.terms)
            })
        });
    }

    #[bench]
    fn bench_trim(b: &mut Bencher) {
        let mut ap = Polynomial::new(vec![]);
        let terms = vec![0, 1, 2];
        let expected_vec = vec![1, 2];
        b.iter(|| {
            black_box({
                ap.terms = terms.clone();
                ap.trim();
                assert_eq!(expected_vec, ap.terms)
            })
        });
    }

    #[bench]
    fn bench_pow(b: &mut Bencher) {
        let a = Polynomial::new(vec![1i32, 2, 3, 4, 5]);
        b.iter(|| {
            black_box({
                a.pow(37);
            })
        });
    }

    #[bench]
    fn bench_pows(b: &mut Bencher) {
        let a = SparsePolynomial::from_vec(vec![1i32, 2, 3, 4, 5]);
        b.iter(|| {
            black_box({
                a.pow(37);
            })
        });
    }

    #[bench]
    fn bench_eval(b: &mut Bencher) {
        let a = Polynomial::new(vec![1i32, 2, 3, 4, 5]);
        b.iter(|| {
            black_box({
                a.eval(5);
            })
        });
    }

    #[bench]
    fn bench_evals(b: &mut Bencher) {
        let a = SparsePolynomial::from_vec(vec![1i32, 2, 3, 4, 5]);
        b.iter(|| {
            black_box({
                a.eval(5);
            })
        });
    }
}
