#![feature(test)]

extern crate rustnomial;

mod bench {
    extern crate test;
    use self::test::Bencher;
    use self::test::black_box;
    use rustnomial::Polynomial;

    #[bench]
    fn bench_init(b: &mut Bencher) {
        b.iter(|| {
            Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
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
    fn bench_scale(b: &mut Bencher) {
        b.iter(|| {
            Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 5;
        });
    }

    #[bench]
    fn bench_div(b: &mut Bencher) {
        b.iter(|| {
            Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) / 5;
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
        let mut ap = black_box(Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        b.iter(|| black_box(ap.degree()));
    }

    #[bench]
    fn bench_trim(b: &mut Bencher) {
        let mut ap = Polynomial::new(vec![]);
        let terms = vec![0; 10000];
        b.iter(|| black_box({
            ap.terms = terms.clone();
            ap.trim()
        }));
    }

    #[bench]
    fn bench_pow(b: &mut Bencher) {
        let mut a = Polynomial::new(vec![1i32, 2, 3, 4, 5]);
        b.iter(|| black_box({
            a.pow(37);
        }));
    }

    // #[bench]
    // fn bench_pow_manual(b: &mut Bencher) {
    //     let a = Polynomial::new(vec![1i32, 2, 3, 4, 5]);
    //     b.iter(|| black_box({
    //         a
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a)
    //         .borrow_mul(&a).borrow_mul(&a);
    //     }));
    // }

}