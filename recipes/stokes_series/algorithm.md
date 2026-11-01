# stokes_series - algorithm (4_1)

Conventions: qt = e^{-2 pi i/tau}, xt = e^{u/tau}; cones I, II (|qt| < 1), III, IV (|qt| > 1) as in GGMW Fig. 5.

1. Plain blocks C_m, A_m, B_m (GGMW (94)) as truncated qt-series with coefficients in Q(xt) (or Q for a
   rational xt); for |qt| > 1 use p = 1/qt and GGMW (98)-(99).
2. Decorations alpha, beta with the *squared* theta (GGMW (109)-(110)); for |qt| > 1 the products are continued
   by (a;q)_inf = 1/(a/q;1/q)_inf, which is consistent with GGMW (112).
3. calJ_{-1} = diag(1, alpha, beta) J_{-1};  M_R = L_R calJ_{-1} P_R (GGMW (118a-d)).
   P_III, P_IV as corrected in the Phi0 note (corrections to GGMW); S_{IV->I}, S_{II->III} then reproduce
   GGMW (131), (135).
4. S_{R->R'} = M_{R'}^{-1} M_R for I -> II and III -> IV (exact series inversion);
   IV -> I and II -> III are formal (the calJ factors cancel), with the branch-cut rule GGMW (134).
5. Gauss factorisation: LDU for the upper sweep (I -> II), UDL for the lower sweep (III -> IV); the latter
   reproduces GGMW (130) and the mirror rules GGM (66).
6. Assertions: every coefficient is in Z[xt^{+-1}]; S -> 1 as qt -> 0; Euler (plethystic) exponents of S_gg
   are integral.

## 7_2 (upper sweep)
1. q~-blocks G^eps_j(xt; qt; n) of the two-dimensional factorisation (series in s = qt^{1/4}), for the
   descendant basis B = {(0,0),(0,1),(1,0),(1,1),(2,1)} and the six columns (j, eps).
2. Conjugation factors r_j (theta quotients) and T = Ytilde_B(1/xt) R E (Ytilde_B E)^{-1} (E: five columns).
3. S^+ = diag(-1,-1,-1,-1,-xt^2) Pi N_I(1/xt) T N_I(xt)^{-1}, Pi = (s1 s2)(s3 s4).
4. Assertions: only integer powers of qt occur; coefficients are integral Laurent polynomials; xt symbolic
   (the columns j = 0, 2 coincide at xt = 1, so xt = 1 is reached by specialisation).

## 5_2
1. Blocks A_m, B_m, C_m (GGM (241)-(242)) at nome q and, for |q| > 1, at nome q^{-1} with
   theta(x; q^{-1}) = 1/theta(x; q) and H(.; q^{-1}) = H_- (GGM sec. 4); check: orthogonality GGM (256).
2. W_{-1}(x; q) (GGM (244)) and M_R = W_{-1}(x^{+-1}; q^{+-1})^T R_R (GGM (304a-d)).
3. Reduced S_{I->II} = M_II^{-1} M_I (only |q| < 1 blocks); check: S_{I->II} S_{IV->I} = GGM (305a) (which uses |q| > 1
   blocks), S^+_{11} = GGM (316).  Lower sweep: S_{III->IV} = M_IV^{-1} M_III, and GGM (305b).
4. Trivial row: Campbell's D_m (GGMW (219)), M_R = (qt D_0, qt^2 D_1, qt^3 D_2) M^{(s0)}_R (GGMW (235)-(236)) and
   S_{0,j} = M_{I,j} - sum_k M_{II,k} S_{k,j} (GGMW (237)); check: GGMW (238).
