from fractions import Fraction

def stage(L):
    cond = 2**L - 1
    soft = 0 if L < 4 else 2**(L-3) - 1
    return {
        "L": L,
        "frontier": 2**L,
        "lambda_min": Fraction(1, cond),
        "condition": cond,
        "soft_below_0_1": soft,
        "soft_fraction": Fraction(soft, cond),
    }

for L in range(1, 9):
    s = stage(L)
    print(
        f"L={s['L']}: frontier={s['frontier']} "
        f"lambda_min={s['lambda_min']} cond={s['condition']} "
        f"soft={s['soft_below_0_1']}/{s['condition']}={float(s['soft_fraction']):.6f}"
    )
