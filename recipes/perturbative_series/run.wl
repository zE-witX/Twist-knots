(* kdb recipe perturbative_series -- Wolfram Language engine (independent implementation of algorithm.md).
   Usage:  wolframscript -file run.wl '{"knot":"4_1","sector":"0","order":4,"x":"symbol"}'
   Prints one JSON object {"coefficients": [...]} (InputForm strings).
   Implemented: phi_0 of any twist knot K_p (Habiro/MMR eps-route; exact at x = 1), sectors g, c of 4_1
   (formal Gaussian expansion; exact generic x via reduction modulo the saddle curve, numeric fixed x).
   Not ported: the 2d I_{K_p} saddle engine (python only). *)

params = If[ValueQ[PARAMS], PARAMS, ImportString[$ScriptCommandLine[[2]], "RawJSON"]];
twistOf = <|"3_1" -> 1, "4_1" -> -1, "5_2" -> 2, "6_1" -> -2, "7_2" -> 3, "8_1" -> -3, "9_2" -> 4, "10_1" -> -4|>;
p = If[KeyExistsQ[params, "twist"], params["twist"], twistOf[params["knot"]]];
sector = ToString[Lookup[params, "sector", "0"]];
order = Lookup[params, "order", 4];
xspec = ToString[Lookup[params, "x", "symbol"]];

qp[n_] := Product[1 - q^j, {j, 1, n}];
(* Masbaum's cyclotomic coefficients, Marino 5.4 / CLAUDE.md 3.0 *)
masbaumC[pp_, n_] := Cancel[-q^n Sum[(-1)^k q^(pp k (k + 1) + k (k - 1)/2) (q^(2 k + 1) - 1) qp[n]/(qp[n + k + 1] qp[n - k]), {k, 0, n}]];
hCoeffs[f_, L_] := PadRight[CoefficientList[Normal[Series[f /. q -> Exp[h], {h, 0, L}]], h], L + 1];

phi0x1[pp_, L_] := Total[Table[hCoeffs[Expand[masbaumC[pp, n] qp[n]^2], L], {n, 0, Floor[L/2]}]];

(* eps-route: prod_{j<=n} (eps + s_j), s_j = 2 cosh(j h) - 2, H_n = C_n q^(n(n+1)/2) *)
mmrEps[pp_, M_] := Module[{A = 2 M + 6, nmax, prod, total, s, new, hc, out, f, trunc},
  trunc[g_] := Normal[Series[g, {eps, 0, A}]];
  nmax = A + Floor[M/2] + 1;
  prod = Join[{1}, ConstantArray[0, M]];
  total = ConstantArray[0, M + 1];
  Do[
   If[n >= 1,
    s = Table[If[EvenQ[m] && m >= 2, 2 n^m/m!, 0], {m, 0, M}];
    new = Table[eps prod[[m + 1]] + Sum[s[[k + 1]] prod[[m - k + 1]], {k, 2, m, 2}], {m, 0, M}];
    prod = trunc /@ Expand[new]];
   hc = hCoeffs[Expand[masbaumC[pp, n] q^(n (n + 1)/2)], M];
   total += Table[Sum[hc[[k + 1]] prod[[m - k + 1]], {k, 0, m}], {m, 0, M}],
   {n, 0, nmax}];
  Table[f = trunc[Expand[total[[m + 1]] (1 - pp eps)^(2 m + 1)]];
   If[Exponent[f, eps] > 2 m, Print["MMR form violated"]; Abort[]];
   f, {m, 0, M}]];

phi0Symbolic[pp_, M_] := Module[{P = mmrEps[pp, M]},
  Table[Factor[Together[(P[[m + 1]]/(1 - pp eps)^(2 m + 1)) /. eps -> 2 - x - 1/x]], {m, 0, M}]];

(* ---------------- 4_1, sectors g and c ---------------- *)
liNeg[m_, t_] := Sum[k! StirlingS2[m + 1, k + 1] (t/(1 - t))^(k + 1), {k, 0, m}];
expTerms[xx_, yy_, n_] := Table[
   (liNeg[j, yy] + liNeg[j, xx yy]) zz^(j + 2)/(j + 2)! +
    Sum[BernoulliB[2 r, 1/2] (liNeg[2 r + (j - 4 r + 2) - 2, yy] + liNeg[2 r + (j - 4 r + 2) - 2, xx yy])/(2 r)! zz^(j - 4 r + 2)/(j - 4 r + 2)!,
     {r, 1, Floor[(j + 2)/4]}] - If[j == 2, 1/24, 0], {j, 1, 2 n}];
expCoeffs[ps_] := Module[{e = {1}}, Do[AppendTo[e, Expand[Sum[k ps[[k]] e[[j - k + 1]], {k, 1, j}]/j]], {j, 1, Length[ps]}]; e];
gAvg[f_, d_] := With[{c = CoefficientList[f, zz]}, Sum[c[[k + 1]] If[OddQ[k], 0, (k - 1)!! (-1/d)^(k/2)], {k, 0, Length[c] - 1}]];
hess[xx_, yy_] := 1 + yy/(1 - yy) + xx yy/(1 - xx yy);
gauss41[xx_, yy_, n_] := Module[{e = expCoeffs[expTerms[xx, yy, n]], d = hess[xx, yy]}, Table[gAvg[e[[2 k + 1]], d], {k, 0, n}]];
curve = x y^2 + (x^2 - x - 1) y + 1;
reduceY[f_] := With[{r = Together[f]}, Cancel[PolynomialRemainder[Numerator[r], curve, y]/PolynomialRemainder[Denominator[r], curve, y]]];
gcSymbolic[n_, sgn_] := Module[{a = gauss41[x, y, n], dd = y - 1/(x y)},
  Table[If[k == 0, "1",
    ToString[Factor[reduceY[a[[k + 1]] dd^(3 k)]], InputForm] <> " / (" <> If[sgn > 0, "+", "-"] <> "(y_g - y_c))^" <> ToString[3 k]], {k, 0, n}]];
gcNumeric[xv_, n_, sgn_, digits_] := Module[{wp = digits + 40 + 4 n, xx, yy},
  xx = N[xv, wp];
  yy = (1 + xx - xx^2 + sgn Sqrt[(xx^2 - 3 xx + 1) (xx^2 + xx + 1)])/(2 xx);
  ToString[N[#, digits], InputForm] & /@ gauss41[xx, yy, n]];

res = <|"engine" -> "wolfram", "twist_p" -> p, "sector" -> sector|>;
Which[
  sector == "0" && xspec == "1", res["coefficients"] = ToString[#, InputForm] & /@ phi0x1[p, order],
  sector == "0" && xspec == "symbol", res["coefficients"] = ToString[#, InputForm] & /@ phi0Symbolic[p, order],
  sector == "0", res["coefficients"] = ToString[#, InputForm] & /@ (phi0Symbolic[p, order] /. x -> ToExpression[xspec]),
  p == -1 && MemberQ[{"g", "c"}, sector],
  res["coefficients"] = If[xspec == "symbol", gcSymbolic[order, If[sector == "g", 1, -1]],
    gcNumeric[ToExpression[xspec], order, If[sector == "g", 1, -1], Lookup[params, "digits", 40]]],
  True, res["error"] = "not implemented in the Wolfram engine (python only)"];
Print[ExportString[res, "RawJSON", "Compact" -> True]];
