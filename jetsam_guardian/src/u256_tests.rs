use super::*;

fn le(x: u128) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&x.to_le_bytes());
    out
}

#[test]
fn u256_cmp_sub_and_be_round_trip() {
    let a = le(1 << 100);
    let b = le((1 << 100) - 7);
    assert_eq!(cmp(&a, &b), Ordering::Greater);
    assert_eq!(sub(&a, &b), le(7));
    assert_eq!(sub(&b, &a), [0u8; 32], "saturates at 0");
    assert_eq!(from_be(&to_be(&a)), a);
    assert_eq!(to_be(&from_u64(1))[31], 1);
}

#[test]
fn u256_products_are_exact_on_320_bits() {
    let max = [0xFFu8; 32];
    let p = mul_u64(&max, u64::MAX);
    // (2^256 − 1)(2^64 − 1) = 2^320 − 2^256 − 2^64 + 1
    assert_eq!(p, [1, u64::MAX, u64::MAX, u64::MAX, u64::MAX - 1]);
    assert!(mul_ge(&le(10), 4, &le(5), 8));
    assert!(!mul_ge(&le(10), 3, &le(5), 7));
}
