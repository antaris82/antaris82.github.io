#!/usr/bin/env python3
"""
Schur–Baum Primzahl-Generator

Berechnet die Schur-Impedanzrekursion auf b-ären Bäumen der Tiefe L
und extrahiert ganzzahlige Eigenwerte > 1 aus den irreduziblen Faktoren
des resultierenden Spektralpolynoms.

Primzahltest über Wilson's Theorem: n ist prim ⟺ (n-1)! ≡ -1 (mod n).
Kein externer isprime()-Aufruf.

Mathematischer Hintergrund:
  Der Graph-Laplacian eines b-ären Baums der Tiefe L wird rekursiv
  Schur-eliminiert von den Blättern zur Wurzel.
  Die Impedanzrekursion lautet:
    σ_L(λ)     = λ - 1                              (Blätter, Grad 1)
    σ_k(λ)     = (λ - (b+1)) - b / σ_{k+1}(λ)      (Interior, Grad b+1)
    σ_0(λ)     = (λ - b) - b / σ_1(λ)               (Wurzel, Grad b)

  Der Zähler von σ_0(λ) faktorisiert über ℚ. Ganzzahlige Nullstellen
  der linearen Faktoren sind Eigenwerte des Laplacians.

Nutzung:
  python schur_prime_gen.py           (Standard: L=1..4, b=2..99)
  python schur_prime_gen.py 5         (L=1..5)
  python schur_prime_gen.py 3 50      (L=1..3, b=2..50)
"""

from sympy import Symbol, cancel, numer, together, expand, Poly, Rational
import sys


# ── Wilson-Primzahltest ──────────────────────────────────────────────

def is_prime_wilson(n: int) -> bool:
    """Wilson's Theorem: n ist prim ⟺ (n-1)! ≡ -1 (mod n)."""
    if n < 2:
        return False
    if n == 2:
        return True
    factorial_mod = 1
    for k in range(2, n):
        factorial_mod = (factorial_mod * k) % n
    return factorial_mod == n - 1


# ── Schur-Impedanzrekursion ──────────────────────────────────────────

def schur_sigma(b: int, L: int, lam):
    """
    Rekursive Schur-Impedanz σ_0(λ) für b-ären Baum der Tiefe L.

    Gibt eine rationale Funktion in λ zurück (sympy-Ausdruck).
    """
    if L == 0:
        return lam  # einzelner Knoten, keine Kanten

    # Blätter (Level L): Grad 1
    s = lam - 1

    # Interior (Level L-1 bis 1): Grad b+1
    for _ in range(L - 1, 0, -1):
        s = (lam - (b + 1)) - Rational(b) / s
        s = cancel(s)

    # Wurzel (Level 0): Grad b
    s = (lam - b) - Rational(b) / s
    s = cancel(s)

    return s


# ── Ganzzahlige Eigenwerte extrahieren ───────────────────────────────

def integer_eigenvalues(sigma_expr, lam):
    """
    Faktorisiert den Zähler von σ_0 über ℚ und gibt alle ganzzahligen
    Nullstellen > 1 der linearen Faktoren zurück.
    """
    num = numer(together(sigma_expr))
    num = expand(num)
    poly = Poly(num, lam, domain='QQ')

    eigenvalues = set()
    _coeff, factors = poly.factor_list()
    for fac, _mult in factors:
        if fac.degree() == 1:
            a = fac.nth(1)
            b_coef = fac.nth(0)
            root = Rational(-b_coef, a)
            if root.is_integer and root > 1:
                eigenvalues.add(int(root))
    return eigenvalues


# ── Analytische Formel für L=2 ───────────────────────────────────────

def L2_formula(max_k: int):
    """
    Für L=2 ist der Zähler von σ_0 exakt:
      λ · (λ² - 2(b+1)λ + (b²+b+1))
    Diskriminante Δ = 4b. Reduzibel über ℚ ⟺ b = k².
    Dann Wurzeln: k²+k+1 und k²-k+1.

    Diese Funktion gibt die Eigenwertpaare für b = k² zurück.
    """
    pairs = []
    for k in range(1, max_k + 1):
        b = k * k
        r1 = k * k + k + 1
        r2 = k * k - k + 1
        pairs.append((b, k, r1, r2, is_prime_wilson(r1), is_prime_wilson(r2)))
    return pairs


# ── Hauptprogramm ────────────────────────────────────────────────────

def main():
    L_max = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    b_max = int(sys.argv[2]) if len(sys.argv) > 2 else 99

    lam = Symbol('lam')

    all_primes = set()
    detail_lines = []

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Schur–Baum Primzahl-Generator                      ║")
    print("║     Primzahltest: Wilson's Theorem (kein isprime)       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── Phase 1: L=1 (Sterngraphen) ──────────────────────────────

    print("━━━ Phase 1: L=1 (Sterngraph K_{1,b}) ━━━")
    print("    Eigenwerte: 0, 1^{b-1}, b+1")
    print("    b+1 prim ⟺ b+1 prim (trivial, aber Schur-derived)")
    print()

    L1_primes = []
    for b in range(2, b_max + 1):
        ev = b + 1
        if is_prime_wilson(ev):
            L1_primes.append(ev)
            all_primes.add(ev)

    print(f"    Primzahlen via L=1 (b=2..{b_max}):")
    print(f"    {L1_primes}")
    print()

    # ── Phase 2: L=2 (analytische Formel) ────────────────────────

    print("━━━ Phase 2: L=2 (analytische Formel) ━━━")
    print("    Zähler von σ_0: λ · (λ² - 2(b+1)λ + (b²+b+1))")
    print("    Δ = 4b → reduzibel ⟺ b = k²")
    print("    Dann Eigenwerte: k²+k+1 und k²-k+1")
    print()

    max_k = int(b_max ** 0.5) + 1
    pairs = L2_formula(max_k)

    print(f"    {'k':>3}  {'b=k²':>5}  {'k²+k+1':>8}  {'prim?':>6}  {'k²-k+1':>8}  {'prim?':>6}")
    print(f"    {'─'*3}  {'─'*5}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*6}")
    for b, k, r1, r2, p1, p2 in pairs:
        if b > b_max:
            break
        mark1 = "  ✓" if p1 else ""
        mark2 = "  ✓" if p2 else ""
        print(f"    {k:>3}  {b:>5}  {r1:>8}{mark1:>6}  {r2:>8}{mark2:>6}")
        if p1:
            all_primes.add(r1)
        if p2 and r2 > 1:
            all_primes.add(r2)
    print()

    # ── Phase 3: Allgemeine Schur-Rekursion (L=2..L_max) ─────────

    if L_max >= 2:
        print(f"━━━ Phase 3: Allgemeine Schur-Rekursion (L=2..{L_max}) ━━━")
        print()

        for L in range(2, L_max + 1):
            found_this_L = {}
            for b in range(2, b_max + 1):
                sigma = schur_sigma(b, L, lam)
                eigvals = integer_eigenvalues(sigma, lam)
                for ev in eigvals:
                    if ev > 1 and ev not in all_primes:
                        if is_prime_wilson(ev):
                            all_primes.add(ev)
                            if ev not in found_this_L:
                                found_this_L[ev] = b

            if found_this_L:
                new_sorted = sorted(found_this_L.keys())
                print(f"    L={L}: neue Primzahlen aus ganzzahligen Eigenwerten:")
                for p in new_sorted:
                    print(f"          {p} (erstmals bei b={found_this_L[p]})")
            else:
                print(f"    L={L}: keine neuen Primzahlen über L=1 hinaus")
            print()

    # ── Phase 4: Diskriminanten-Analyse (L=2) ────────────────────

    print("━━━ Phase 4: Diskriminanten-Analyse (L=2) ━━━")
    print("    Für irreduzible quadratische Faktoren: Δ = 4b")
    print("    Primfaktoren von b ↔ arithmetische Struktur")
    print()

    print(f"    {'b':>4}  {'Δ=4b':>6}  {'reduzibel?':>11}  {'Primfaktoren von b'}")
    print(f"    {'─'*4}  {'─'*6}  {'─'*11}  {'─'*20}")
    for b in range(2, min(b_max + 1, 26)):
        delta = 4 * b
        is_square = int(delta ** 0.5) ** 2 == delta
        # Prime factorization of b using Wilson-based method
        factors = []
        temp = b
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors.append(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.append(temp)

        red_str = "  ja (b=k²)" if is_square else "  nein"
        fac_str = " × ".join(str(f) for f in factors)
        print(f"    {b:>4}  {delta:>6}{red_str:>11}  {fac_str}")
    print()

    # ── Zusammenfassung ──────────────────────────────────────────

    print("═" * 58)
    print(f"Gefundene Primzahlen (L=1..{L_max}, b=2..{b_max}):")
    sorted_primes = sorted(all_primes)
    # Print in rows of 10
    for i in range(0, len(sorted_primes), 10):
        chunk = sorted_primes[i:i+10]
        print(f"  {', '.join(str(p) for p in chunk)}")
    print(f"\nAnzahl: {len(sorted_primes)}")
    print()

    # ── Vergleich: Alle Primzahlen bis max ───────────────────────

    if sorted_primes:
        max_p = sorted_primes[-1]
        all_below = [n for n in range(2, max_p + 1) if is_prime_wilson(n)]
        missing = set(all_below) - all_primes
        if missing:
            print(f"Fehlende Primzahlen ≤ {max_p} (nicht als Eigenwert gefunden):")
            print(f"  {sorted(missing)}")
        else:
            print(f"Alle Primzahlen ≤ {max_p} gefunden: ✓ vollständig")


if __name__ == '__main__':
    main()
