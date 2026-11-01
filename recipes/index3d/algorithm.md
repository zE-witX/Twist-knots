# index3d - algorithm

1. Gluing equations (SnapPy convention: rows = edges, meridian, longitude; columns (z, z', z'') per tetrahedron),
   frozen in `src/kdb/knots/nz_data.yaml`.  Eliminate z': A = a - b, B = c - b, nu = 2 - sum b (edges), - sum b (cusp).
2. Tetrahedron index I_Delta(m, e)(q) = sum_n (-1)^n q^{n(n+1)/2 - (n + e/2) m} / ((q)_n (q)_{n+e}), in q^{1/2} units.
3. State sum over the edge charges k in Z^{N-1} (one edge equation is redundant):
   I(m, e) = sum_k (-q^{1/2})^{nu.k + e nu_mu + m nu_lam/2} prod_j I_Delta(-B_j.k - e B_mu,j - m B_lam,j/2, A_j.k + e A_mu,j + m A_lam,j/2).
4. Summation region: steepest descent of the degree function from k = 0, then flood fill of {degree <= order}
   (with a safety margin); every factor is expanded to the order needed given the degrees of the other factors.
5. Ind^rot(m, x; q) = sum_e I(m, e) x^e, computed two orders deeper than requested and truncated.
