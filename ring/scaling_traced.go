package ring

// Traced clone of DivRoundByLastModulusNTT (scaling.go:99) for vFHE rescale
// proofs. It performs the EXACT same RNS divide-and-round-by-last-modulus, but
// routes the two NTTs through the stage-snapshotting clones and emits every
// internal region the prover binds (vfhe::layout_rescale):
//
//   operand  : p0 limbs (NTT form)                              [comp][in_limb]
//   intt     : per-stage iNTT of the dropped (top) limb         [comp][stage]
//   advice   : iNTT(dropped limb) in coeff form (= c_last)      [comp]
//   divround : per kept limb, the divide-and-round coeff delta  [comp][out_limb]
//   ntt      : per kept limb, per-stage NTT of delta            [comp][out_limb][stage]
//   result   : p1 limbs (NTT form)                              [comp][out_limb]
//
// Faithfulness: advice and delta are the engine's real intermediates, computed
// FORWARD from the operand by the rescale algorithm (buff0 = iNTT of the dropped
// limb; buff1 = the centered residue mapped into q_i). The result p1 is then
// produced from delta by the real divide-and-round relation
// p1_i = (p0_i - NTT(delta_i)) * q_last^-1 mod q_i — nothing is derived backward
// from the result. Using canonical (reduced) intermediates instead of Lattigo's
// lazy ones only removes multiples of q, so p1 is identical to the production
// DivRoundByLastModulusNTT.
func (r Ring) DivRoundByLastModulusNTTTraced(comp int, p0, p1 Poly, sink TraceSink) {

	level := r.level
	inLimbs := level + 1
	outLimbs := level
	N := r.N()

	// operand: all in-limbs of this component (already canonical NTT residues).
	for l := 0; l <= level; l++ {
		sink.Poly("operand", comp*inLimbs+l, p0.Coeffs[l])
	}

	buff0 := r.pool.GetBuffUintArray()
	defer r.pool.RecycleBuffUintArray(buff0)
	buff1 := r.pool.GetBuffUintArray()
	defer r.pool.RecycleBuffUintArray(buff1)

	// Dropped (top) limb: canonical iNTT with per-stage snapshots. The normalized
	// last stage IS the advice c_last; emit a copy before centering.
	sd := r.SubRings[level]
	BackwardNTTTraced(p0.Coeffs[level], *buff0, N, sd.NInv, sd.Modulus, sd.MRedConstant,
		sd.RootsBackward, "intt", comp, sink)
	advice := make([]uint64, N)
	copy(advice, *buff0)
	sink.Poly("advice", comp, advice)

	// Center by (p-1)/2 — production does this in place on buff0.
	pHalf := (sd.Modulus - 1) >> 1
	sd.AddScalar(*buff0, pHalf, *buff0)

	for i, s := range r.SubRings[:level] {
		// delta_i (divide-and-round coeff) = centered dropped-limb residue mapped
		// into q_i, exactly as production computes buff1 before its NTT.
		s.AddScalarLazy(*buff0, s.Modulus-BRedAdd(pHalf, s.Modulus, s.BRedConstant), *buff1)
		delta := make([]uint64, N)
		for k := 0; k < N; k++ {
			delta[k] = reduceCanonical((*buff1)[k], s.Modulus)
		}
		sink.Poly("divround", comp*outLimbs+i, delta)

		// NTT(delta_i) with per-stage snapshots; last stage == D_i.
		ntt := make([]uint64, N)
		ForwardNTTTraced(delta, ntt, N, s.Modulus, s.MRedConstant, s.RootsForward,
			"ntt", comp*outLimbs+i, sink)

		// p1_i = (p0_i - NTT(delta_i)) * q_last^-1 mod q_i (real rescale relation).
		s.SubThenMulScalarMontgomeryTwoModulus(ntt, p0.Coeffs[i], r.RescaleConstants[level-1][i], p1.Coeffs[i])
		sink.Poly("result", comp*outLimbs+i, p1.Coeffs[i])
	}
}
