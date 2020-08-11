/// Applies `$macro` in the context of upscaling from one numerical type
/// to another, more permissive numerical type. Assumes `$macro` takes
/// the smaller type first, and the larger type second.
/// All possible upscalings will be applied.
macro_rules! upcast {
    ($macro:ident) => {
        $macro!(u8, u16);
        $macro!(u8, u32);
        $macro!(u8, u64);
        $macro!(u8, u128);

        $macro!(u16, u32);
        $macro!(u16, u64);
        $macro!(u16, u128);

        $macro!(u32, u64);
        $macro!(u32, u128);

        $macro!(u64, u128);

        $macro!(u8, i16);
        $macro!(u8, i32);
        $macro!(u8, i64);
        $macro!(u8, i128);

        $macro!(u16, i32);
        $macro!(u16, i64);
        $macro!(u16, i128);

        $macro!(u32, i64);
        $macro!(u32, i128);

        $macro!(u64, i128);

        $macro!(i8, i16);
        $macro!(i8, i32);
        $macro!(i8, i64);
        $macro!(i8, i128);

        $macro!(i16, i32);
        $macro!(i16, i64);
        $macro!(i16, i128);

        $macro!(i32, i64);
        $macro!(i32, i128);

        $macro!(i64, i128);

        $macro!(u8, f32);
        $macro!(u16, f32);
        $macro!(i8, f32);
        $macro!(i16, f32);

        $macro!(u8, f64);
        $macro!(u16, f64);
        $macro!(u32, f64);
        $macro!(i8, f64);
        $macro!(i16, f64);
        $macro!(i32, f64);
        $macro!(f32, f64);
    };
}
