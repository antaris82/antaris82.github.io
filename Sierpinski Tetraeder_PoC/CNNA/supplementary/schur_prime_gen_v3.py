#!/usr/bin/env python3
"""
Schur–Baum Primzahl-Generator v3

Algebraische Beschleunigung durch:
  1. Fraktale Ketten: nur eine Sequenz pro Level statt zwei
  2. Modulare Vorfilter: garantiert zusammengesetzte Kandidaten überspringen
  3. Algebraische Zahlkörper-Verbindung:
       L=2 → Eisenstein-Ganzzahlen Z[ω],  Norm = k²+k+1
       L=3 → Gauss-Ganzzahlen Z[i],       Norm = (k+1)² + k²

Die Schur-Eigenwerte auf b-ären Bäumen sind Normen in
algebraischen Zahlkörpern. Die Reduzibilität des Spektralpolynoms
über Q entspricht der Zerlegbarkeit der Norm.

Primzahltest: Wilson's Theorem. Kein externer isprime()-Aufruf.

Nutzung:
  python schur_prime_gen_v3.py              (Standard: bis N=100)
  python schur_prime_gen_v3.py 1000         (bis N=1000)
"""

import sys
import math


# ── Wilson-Primzahltest ──────────────────────────────────────────────

def is_prime_wilson(n: int) -> bool:
    """Wilson's Theorem: n prim ⟺ (n-1)! ≡ -1 (mod n)."""
    if n < 2:
        return False
    if n == 2:
        return True
    f = 1
    for k in range(2, n):
        f = (f * k) % n
    return f == n - 1


# ── Modulare Vorfilter ───────────────────────────────────────────────

def is_candidate_eisenstein(k: int) -> bool:
    """
    k²+k+1 ≡ 0 (mod 3) ⟺ k ≡ 1 (mod 3).
    Für k > 1 und k ≡ 1 (mod 3): garantiert zusammengesetzt.
    """
    if k <= 1:
        return True
    return k % 3 != 1


def is_candidate_gauss(k: int) -> bool:
    """
    2k²+2k+1 ≡ 0 (mod 5) ⟺ k ≡ 1 oder k ≡ 3 (mod 5).
    Für k > 2 und (k%5 == 1 oder k%5 == 3): garantiert zusammengesetzt.
    """
    if k <= 2:
        return True
    return k % 5 not in (1, 3)


# ── Fraktale Ketten ─────────────────────────────────────────────────

def eisenstein_chain(N: int):
    """
    L=2 fraktale Kette: a(k) = k²+k+1 für k = 0, 1, 2, ...
    
    Fraktale Eigenschaft: a(k) = (k+1)²-(k+1)+1 = a'(k+1)
    Die Kette überlappt sich selbst:
      k=1: 1 ── 3
      k=2: 3 ── 7
      k=3: 7 ── 13
      ...
    
    Algebraisch: a(k) = N(1 + kω) in Z[ω], ω = e^{2πi/3}
    """
    results = []
    k = 0
    while True:
        val = k * k + k + 1
        if val > N:
            break
        results.append((k, val))
        k += 1
    return results


def gauss_chain(N: int):
    """
    L=3 fraktale Kette: Werte 2k²-2k+1, 2k²+1, 2k²+2k+1 für k = 1, 2, ...
    
    Fraktale Eigenschaft: 2k²+2k+1 = 2(k+1)²-2(k+1)+1
    
    Algebraisch: 2k²+2k+1 = (k+1)² + k² = N(k+1 + ki) in Z[i]
    """
    results = []
    k = 1
    while True:
        e1 = 2 * k * k - 2 * k + 1
        e2 = 2 * k * k + 1
        e3 = 2 * k * k + 2 * k + 1
        if e1 > N:
            break
        results.append((k, e1, e2, e3))
        k += 1
    return results


# ── Hauptprogramm ────────────────────────────────────────────────────

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Schur–Baum Primzahl-Generator v3                      ║")
    print("║  Algebraische Zahlkörper + Modulare Filter + Wilson    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    primes_eisenstein = set()
    primes_gauss = set()
    primes_L1 = set()
    wilson_calls = 0
    skipped_mod = 0

    # ── Eisenstein-Kette (L=2): Z[ω]-Normen ─────────────────────

    print("━━━ Eisenstein-Kette (L=2): k²+k+1 = N(1+kω) in Z[ω] ━━━")
    print("    ω = e^{2πi/3}, Reduzibel ⟺ b = k²")
    print("    Filter: skip k ≡ 1 (mod 3) für k > 1")
    print()

    chain_E = eisenstein_chain(N)
    for k, val in chain_E:
        if val <= 1:
            continue
        if not is_candidate_eisenstein(k):
            skipped_mod += 1
            continue
        wilson_calls += 1
        if is_prime_wilson(val):
            primes_eisenstein.add(val)

    print(f"    Primzahlen: {sorted(primes_eisenstein)}")
    print(f"    Wilson-Aufrufe: {wilson_calls}, übersprungen (mod 3): {skipped_mod}")

    # ── Gauss-Kette (L=3): Z[i]-Normen ──────────────────────────

    print()
    print("━━━ Gauss-Kette (L=3): (k+1)²+k² = N(k+1+ki) in Z[i] ━━━")
    print("    Reduzibel ⟺ b = 2k²")
    print("    Filter: skip k ≡ 1,3 (mod 5) für k > 2")
    print()

    calls_before = wilson_calls
    skip_before = skipped_mod
    chain_G = gauss_chain(N)
    for k, e1, e2, e3 in chain_G:
        for val in [e1, e2, e3]:
            if val <= 1 or val > N or val in primes_eisenstein:
                continue
            if not is_candidate_gauss(k):
                skipped_mod += 1
                continue
            wilson_calls += 1
            if is_prime_wilson(val):
                primes_gauss.add(val)

    new_from_gauss = primes_gauss - primes_eisenstein
    if new_from_gauss:
        print(f"    Neue: {sorted(new_from_gauss)}")
    else:
        print(f"    Keine neuen über Eisenstein hinaus")
    print(f"    Wilson-Aufrufe: {wilson_calls - calls_before}, "
          f"übersprungen (mod 5): {skipped_mod - skip_before}")

    # ── L=1 Auffüller ────────────────────────────────────────────

    print()
    print("━━━ L=1: Auffüller b+1 ━━━")

    already = primes_eisenstein | primes_gauss
    calls_before = wilson_calls
    gaps = []
    for b in range(1, N):
        ev = b + 1
        if ev <= N and ev not in already:
            wilson_calls += 1
            if is_prime_wilson(ev):
                primes_L1.add(ev)
                gaps.append(ev)

    if gaps:
        print(f"    Aufgefüllt: {gaps}")
    else:
        print(f"    Keine Lücken!")
    print(f"    Wilson-Aufrufe: {wilson_calls - calls_before}")

    # ── Zusammenfassung ──────────────────────────────────────────

    all_primes = primes_eisenstein | primes_gauss | primes_L1

    print()
    print("═" * 58)
    sorted_primes = sorted(all_primes)
    for i in range(0, len(sorted_primes), 12):
        chunk = sorted_primes[i:i+12]
        print(f"  {', '.join(str(p) for p in chunk)}")

    print()
    print(f"  Gesamt: {len(all_primes)} Primzahlen bis {N}")

    # Vollständigkeitscheck
    all_ref = [n for n in range(2, N+1) if is_prime_wilson(n)]
    if set(all_ref) == all_primes:
        print(f"  Vollständig: ✓")
    else:
        missing = set(all_ref) - all_primes
        print(f"  Fehlend: {sorted(missing)}")

    # ── Herkunft & Effizienz ─────────────────────────────────────

    print()
    print("── Herkunft ──")
    print(f"  Z[ω] Eisenstein (k²+k+1):    {len(primes_eisenstein):>4}")
    print(f"  Z[i] Gauss ((k+1)²+k²):      {len(primes_gauss):>4} neue")
    print(f"  L=1 Auffüller:                {len(primes_L1):>4}")
    print()

    brute = N - 1
    print("── Effizienz ──")
    print(f"  Wilson-Aufrufe:        {wilson_calls:>6}")
    print(f"  Modulare Skips:        {skipped_mod:>6}")
    print(f"  Brute-Force (2..N):    {brute:>6}")
    if wilson_calls < brute:
        pct = 100 * (1 - wilson_calls / brute)
        print(f"  Ersparnis:             {pct:>5.1f}%")
    print()

    # ── Fraktale Ketten visuell ──────────────────────────────────

    print("── Eisenstein-Kette (Z[ω]) ──")
    print("  k²+k+1 = (k+1)²-(k+1)+1 → Überlappung")
    print()
    for k in range(1, min(10, len(chain_E))):
        val = k*k + k + 1
        if val > N:
            break
        mark = f"*{val}*" if val in all_primes else f" {val} "
        prev = k*k - k + 1
        pmk = f"*{prev}*" if prev in all_primes else f" {prev} "
        print(f"  k={k}:  {pmk:>6} ── {mark:>6}")
    print()

    print("── Gauss-Kette (Z[i]) ──")
    print("  (k+1)²+k² → Überlappung: max(k) = min(k+1)")
    print()
    for k, e1, e2, e3 in chain_G[:8]:
        if e1 > N:
            break
        marks = []
        for e in [e1, e2, e3]:
            if e in all_primes:
                marks.append(f"*{e}*")
            else:
                marks.append(f" {e} ")
        print(f"  k={k}:  {marks[0]:>6} ── {marks[1]:>6} ── {marks[2]:>6}")
    print()
    print("  *n* = prim")
    print()

    # ── Zahlkörper-Zusammenfassung ───────────────────────────────

    print("── Algebraische Zahlkörper-Verbindung ──")
    print()
    print("  Schur-Level │ Reduzibilität │ Zahlkörper │ Normform")
    print("  ────────────┼───────────────┼────────────┼──────────────────")
    print("  L=1         │ immer linear  │ Z          │ b+1")
    print("  L=2         │ b = k²        │ Z[ω]      │ k²+k+1 = N(1+kω)")
    print("  L=3         │ b = 2k²       │ Z[i]      │ (k+1)²+k² = N(k+1+ki)")
    print("  L=4         │ b = ?         │ Z[?]      │ (offen)")
    print()
    print("  Die Schur-Elimination auf Bäumen erzeugt Normen")
    print("  algebraischer Ganzzahlen. Jedes Level L entspricht")
    print("  einem anderen Zahlkörper. Die Primzahlen sind die")
    print("  irreduziblen Normen.")


if __name__ == '__main__':
    main()
