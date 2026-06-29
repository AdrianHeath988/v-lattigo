package ring

// Traced clone of ModDownQPtoQNTT (basis_extension.go:235) for the vFHE
// key-switch (relin/rotate) proof. Single special prime (levelP==0) — the
// provable config. It performs the identical mod-down of prod from basis QP to
// Q (rounded division by P), but emits the proof regions and routes the re-NTT
// through the stage-snapshotting clone:
//
//   md_advice   : iNTT_P(prod's P-limb) in coeff form (= c_P)        [comp]
//   md_divround : per kept limb, c_P base-extended into q_i (centered) [comp][limb]
//   md_ntt      : per kept limb, per-stage NTT of md_divround          [comp][limb][stage]
//   ks          : key-switch output (P^-1 * (prod_i - NTT(divround)))  [comp][limb]
//
// Faithfulness: c_P and the base-extended divround come from the real INTTLazy /
// ModUpPtoQ (forward from prod), and ks is the real SubThenMul output — nothing
// is derived from the final ciphertext. (md_intt, the producing-side iNTT
// stages, is intentionally not emitted: the shared gadget binds c_P as advice,
// matching the existing layout.)
func (be *BasisExtender) ModDownQPtoQNTTTraced(comp, levelQ, levelP int, p1Q, p1P, p2Q Poly, L int, sink TraceSink) {

	ringQ := be.ringQ.AtLevel(levelQ)
	poolQ := be.poolQ.AtLevel(levelQ)
	ringP := be.ringP.AtLevel(levelP)
	poolP := be.poolP.AtLevel(levelP)
	modDownConstants := be.modDownConstantsPtoQ[levelP]
	buffP := poolP.GetBuffPoly()
	defer poolP.RecycleBuffPoly(buffP)
	buffQ := poolQ.GetBuffPoly()
	defer poolQ.RecycleBuffPoly(buffQ)
	N := ringQ.N()

	// c_P = iNTT of the special-prime limb of prod (real intermediate).
	ringP.INTTLazy(p1P, *buffP)
	qP := ringP.SubRings[0].Modulus
	adv := make([]uint64, N)
	for k := 0; k < N; k++ {
		adv[k] = reduceCanonical((*buffP).Coeffs[0][k], qP)
	}
	sink.Poly("md_advice", comp, adv)

	// Base-extend c_P into the Q basis (real ModUpPtoQ); buffQ.Coeffs[i] is the
	// centered divide-and-round coeff for limb i.
	be.ModUpPtoQ(levelP, levelQ, *buffP, *buffQ)

	for i, s := range ringQ.SubRings[:levelQ+1] {
		dr := make([]uint64, N)
		for k := 0; k < N; k++ {
			dr[k] = reduceCanonical((*buffQ).Coeffs[i][k], s.Modulus)
		}
		sink.Poly("md_divround", comp*L+i, dr)

		ntt := make([]uint64, N)
		ForwardNTTTraced(dr, ntt, N, s.Modulus, s.MRedConstant, s.RootsForward, "md_ntt", comp*L+i, sink)

		// ks_i = (prod_i - NTT(divround_i)) * P^-1 mod q_i (real mod-down relation).
		s.SubThenMulScalarMontgomeryTwoModulus(ntt, p1Q.Coeffs[i], s.Modulus-modDownConstants[i], p2Q.Coeffs[i])
		sink.Poly("ks", comp*L+i, p2Q.Coeffs[i])
	}
}

// ModDownQPtoQNTTMultiPTraced is the MULTI-special-prime (nP>=1) variant of
// ModDownQPtoQNTTTraced. The mod-down ARITHMETIC already generalizes — ModUpPtoQ
// reconstructs the single integer [a mod P] (P = ∏ Pᵥ) via CRT over all nP special
// primes, and modDownConstantsPtoQ[levelP] is the multi-P constant — so divround /
// md_ntt / ks stay one-poly-per-Q-limb (the reconstructed [a mod P] mod q_i). The
// ONLY multi-P change is capturing all nP residues of c_P as advice (md_advice idx
// comp*nP+v), which the prover binds to the CRT reconstruction of divround.
func (be *BasisExtender) ModDownQPtoQNTTMultiPTraced(comp, levelQ, levelP int, p1Q, p1P, p2Q Poly, L, nP int, sink TraceSink) {

	ringQ := be.ringQ.AtLevel(levelQ)
	poolQ := be.poolQ.AtLevel(levelQ)
	ringP := be.ringP.AtLevel(levelP)
	poolP := be.poolP.AtLevel(levelP)
	modDownConstants := be.modDownConstantsPtoQ[levelP]
	buffP := poolP.GetBuffPoly()
	defer poolP.RecycleBuffPoly(buffP)
	buffQ := poolQ.GetBuffPoly()
	defer poolQ.RecycleBuffPoly(buffQ)
	N := ringQ.N()

	// c_P = iNTT of the nP special-prime limbs of prod (real intermediate).
	ringP.INTTLazy(p1P, *buffP)
	for v, sP := range ringP.SubRings[:levelP+1] {
		adv := make([]uint64, N)
		for k := 0; k < N; k++ {
			adv[k] = reduceCanonical((*buffP).Coeffs[v][k], sP.Modulus)
		}
		sink.Poly("md_advice", comp*nP+v, adv)
	}

	// Base-extend c_P (all nP residues) into the Q basis (real ModUpPtoQ, CRT);
	// buffQ.Coeffs[i] is the centered divide-and-round coeff for limb i.
	be.ModUpPtoQ(levelP, levelQ, *buffP, *buffQ)

	for i, s := range ringQ.SubRings[:levelQ+1] {
		dr := make([]uint64, N)
		for k := 0; k < N; k++ {
			dr[k] = reduceCanonical((*buffQ).Coeffs[i][k], s.Modulus)
		}
		sink.Poly("md_divround", comp*L+i, dr)

		ntt := make([]uint64, N)
		ForwardNTTTraced(dr, ntt, N, s.Modulus, s.MRedConstant, s.RootsForward, "md_ntt", comp*L+i, sink)

		// ks_i = (prod_i - NTT(divround_i)) * P^-1 mod q_i (P = ∏ Pᵥ).
		s.SubThenMulScalarMontgomeryTwoModulus(ntt, p1Q.Coeffs[i], s.Modulus-modDownConstants[i], p2Q.Coeffs[i])
		sink.Poly("ks", comp*L+i, p2Q.Coeffs[i])
	}
}
