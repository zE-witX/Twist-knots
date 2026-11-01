# perturbative_series - algorithm (step list shared by the Python, Sage and Wolfram engines)

Conventions: q = e^h, h = 2 pi i tau, x = e^u; stored coefficients are h-coefficients.
Sources are cited by short labels (GGMW = Garoufalidis-Gu-Marino-Wheeler, GGM = Garoufalidis-Gu-Marino).

## A. Trivial sector 0 (every twist knot K_p)
1. Cyclotomic coefficients C^p_n(q) of K_p (Masbaum; label "Marino 5.4"), exact Laurent polynomials.
   Check: N = 2, 3 colored Jones polynomials; p = -1 gives the 4_1 Habiro form.
2. x = 1: phi_0(1;h) = sum_n C^p_n(e^h) (e^h;e^h)_n^2, exact to the requested order (term n is O(h^{2n})).
3. Generic x (MMR): write (xq;q)_n (q/x;q)_n = q^{n(n+1)/2} prod_j (eps + s_j), eps = 2 - x - 1/x,
   s_j = 2cosh(jh) - 2; expand in (h, eps) to h^M, eps^{2M+6}; multiply the h^m coefficient by
   Delta^{2m+1}, Delta = 1 - p eps = 1 + p z^2, and read the polynomial P_m(eps) of degree <= 2m
   (the higher eps-coefficients must vanish: asserted).  Cancel common factors to get the minimal
   denominator power (3d+1 at h^{2d} for 4_1).
4. Cross-checks (python): the notebook route (coefficient extraction, Lagrange interpolation in n,
   resummation in t = log x); the residue routes for 4_1 (label "Marino 3.4" at x = 1 with hbar^2 = -h^2;
   the x-dependent residue routine with phi_0 = -sum_n res_{n+1} (-h^2)^n); the formal solution of the
   inhomogeneous q-difference equation of the 4_1 C-blocks (GGMW Thm 11).

## B. Sectors g, c of 4_1 (formal Gaussian expansion of the new state integral, GGMW (139),(148))
1. Exponent coefficients p_j (j = 1..2n) of the integrand around the saddle, from the classical part and the
   small-b expansion of the Faddeev dilogarithms; Li_{-m} via Stirling numbers; regular factor e^{-h/24}.
2. Exponentiate: e_j = (1/j) sum_k k p_k e_{j-k}.
3. Gaussian average with <z^{2k}> = (2k-1)!! (-1/delta)^k, delta = S_vv.
4. Generic x: exact in Q(x)[y]/(x y^2 + (x^2-x-1) y + 1); the scaled coefficients (y - 1/(xy))^{3n} a_n
   are y-free (asserted) and reproduce GGMW (86).
5. Fixed x: numerics at the requested precision (ball radii dropped after each step; validated against
   the 240-digit reference data and by precision repeats).

## C. Non-trivial sectors of K_p (python engine)
1. Integrand: Marino's double contour integral I_{K_p}(z, w) (Faddeev dilogarithms, Gaussian in w, two tanh
   factors), rescaled Z = 2 pi b z, W = 2 pi b w; each factor becomes a term of the generic engine
   (kdb.series.saddle): log Phi_b ~ sum_k h^{2k-1} B_{2k}(1/2)/(2k)! Li_{2-2k}(-e^X).
2. Critical points: the exponentiated critical equations are reduced to one polynomial in B = e^W (exact,
   rational x), factored over Q; spurious factors B, B +- 1 are removed; roots are refined by Newton;
   B <-> 1/B pairs are identified (one representative kept).
3. Formal Gaussian expansion in two variables (Kronecker-substituted polynomials, Wick moments with
   Sigma = -H^{-1}); prefactor e^{Z+W} - e^{Z-W} included; output a_n with a_0 = 1.
4. Sector labels: Bloch-Wigner volume -D(A) + D(AB) + D(A/B) - D(xA) - D(A/x) (lift independent);
   geometric = largest volume, then its conjugate, ..., real sectors last.
5. Normalisation: the series agree with the published/notebook series up to a sector-independent factor
   exp(c h) (c recorded per knot); comparisons use log-series beyond h^1, or differences between sectors.
