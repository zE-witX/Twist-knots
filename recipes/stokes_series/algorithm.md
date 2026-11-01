# stokes_series - algorithm (4_1)

Conventions: qt = e^{-2 pi i/tau}, xt = e^{u/tau}; cones I, II (|qt| < 1), III, IV (|qt| > 1) as in GGMW Fig. 5.

1. Plain blocks C_m, A_m, B_m (GGMW (94)) as truncated qt-series with coefficients in Q(xt) (or Q for a
   rational xt); for |qt| > 1 use p = 1/qt and GGMW (98)-(99).
2. Decorations alpha, beta with the *squared* theta (GGMW (109)-(110)); for |qt| > 1 the products are continued
   by (a;q)_inf = 1/(a/q;1/q)_inf, which is consistent with GGMW (112).
3. calJ_{-1} = diag(1, alpha, beta) J_{-1};  M_R = L_R calJ_{-1} P_R (GGMW (118a-d)).
   P_III, P_IV: the entry printed as xt^{-1} - 1 in GGMW (118c,d) is replaced by 1 (variant 'corrected',
   default).  Reason: only then do S_{IV->I} and S_{II->III} reproduce GGMW (131) and (135), and the
   x -> 1 limit reproduces GGMW (41c,d), (54).  The printed variant is available as 'ggmw118'.
4. S_{R->R'} = M_{R'}^{-1} M_R for I -> II and III -> IV (exact series inversion);
   IV -> I and II -> III are formal (the calJ factors cancel), with the branch-cut rule GGMW (134).
5. Gauss factorisation: LDU for the upper sweep (I -> II), UDL for the lower sweep (III -> IV); the latter
   reproduces GGMW (130) and the mirror rules GGM (66).
6. P-frame: S^P = E S E^{-1}, E = D Delta^{-1}; tau drops out: 0g -> (xt - 1), 0c -> (1 - xt^{-1}),
   gc -> xt^{-1}, cg -> xt (multipliers), diagonal unchanged.
7. Assertions: every coefficient is in Z[xt^{+-1}]; S -> 1 as qt -> 0; Euler (plethystic) exponents of S_gg
   are integral.
