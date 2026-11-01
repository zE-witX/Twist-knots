(* kdb recipe stokes_series -- Wolfram Language engine (independent implementation of algorithm.md).
   Usage:  wolframscript -file run.wl '{"knot":"4_1","from_cone":"I","to_cone":"II","order":4,"x":"symbol"}'
   Prints {"entries": {"gg": [...], ...}}: q~-coefficients (strings in xt) of the standard-form entries of
   S_{I->II} = M_II^-1 M_I, M_R = L_R calJ_{-1} P_R, decorated blocks with squared theta (CLAUDE.md 3.3-3.5). *)

params = If[ValueQ[PARAMS], PARAMS, ImportString[$ScriptCommandLine[[2]], "RawJSON"]];
nn = Lookup[params, "order", 4];
xspec = ToString[Lookup[params, "x", "symbol"]];
X = If[xspec == "symbol", xt, ToExpression[xspec]];

tr[f_] := Normal[Series[f, {qt, 0, nn}]];
poch[a_, s_, n_] := Product[1 - a qt^(s + j), {j, 0, Min[n - 1, nn - s]}];
pochInf[a_, s_] := poch[a, s, nn + 1 - s];
kList[m_] := Select[Table[{k, k (k + 1)/2 + k m}, {k, 0, nn + 2 + Abs[m]}], #[[2]] <= nn &];
cB[m_] := Sum[(-1)^kk[[1]] qt^kk[[2]]/(poch[1/X, 0, kk[[1]] + 1] poch[X, 0, kk[[1]] + 1]), {kk, kList[m]}];
aB[m_, x_] := Sum[(-1)^kk[[1]] qt^kk[[2]] x^(kk[[1]] + m)/(poch[1, 1, kk[[1]]] poch[x^2, 1, kk[[1]]]), {kk, kList[m]}];
thp = pochInf[X, 1] pochInf[1/X, 0];
thm = pochInf[1/X, 1] pochInf[X, 0];
alpha = pochInf[X^2, 1]/thp^2;
beta = X pochInf[X^-2, 1]/thm^2;
ser[f_] := Series[f, {qt, 0, nn}];
J = {{ser[1], ser[cB[-1]], ser[cB[0]]},
     {0, ser[alpha aB[-1, X]], ser[alpha aB[0, X]]},
     {0, ser[beta aB[-1, 1/X]], ser[beta aB[0, 1/X]]}};
PI = {{1, 0, 0}, {0, 0, -1}, {0, 1, -1}};
PII = {{1, 0, 0}, {0, 1, 0}, {0, 1, -1}};
LII = {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}};
MI = J.PI;
MII = LII.J.PII;
S = Inverse[MII].MI;
s11 = S[[2, 2]];
ent = <|"gg" -> s11, "gc" -> S[[2, 3]]/s11, "0g" -> S[[1, 2]]/s11, "cg" -> S[[3, 2]]/s11|>;
ent["0c"] = S[[1, 3]] - ent["0g"] S[[2, 3]];
ent["cc"] = S[[3, 3]] - ent["cg"] S[[2, 3]];
ent["00"] = S[[1, 1]];
If[Lookup[params, "frame", "standard"] == "P",
  ent = <|"0g" -> (X - 1) ent["0g"], "0c" -> (1 - 1/X) ent["0c"], "gc" -> ent["gc"]/X, "cg" -> X ent["cg"],
     "gg" -> ent["gg"], "cc" -> ent["cc"], "00" -> ent["00"]|>];
coeffs[s_] := Table[ToString[Factor[Together[SeriesCoefficient[s, {qt, 0, k}]]], InputForm], {k, 0, nn}];
Print[ExportString[<|"engine" -> "wolfram", "entries" -> (coeffs /@ ent)|>, "RawJSON", "Compact" -> True]];
