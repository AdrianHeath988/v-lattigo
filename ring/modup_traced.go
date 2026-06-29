package ring

// vFHE trace capture for the bootstrap MOD-UP (modulus raise q -> Q).
//
// Bootstrap's ModUp (circuits/ckks/bootstrapping/evaluator.go ModUp) raises a
// level-0 ciphertext from the bottom prime q0 to the full chain Q by a CENTERED
// CRT basis extension, coefficient by coefficient:
//
//   a   = the q0-limb coefficient (coeff domain), in [0, q0)
//   x   = the centered representative of a: x = a       if a <  q0/2
//                                          x = a - q0   if a >= q0/2   (so |x| < q0/2)
//   lifted_i = x mod q_i  for every higher prime q_i  (in [0, q_i))
//
// i.e. the bottom-limb integer is re-expressed (centered) across the whole RNS
// basis. The lifted limbs are then NTT'd and the message is scaled. This file
// captures the centered-lift half — the genuinely new proof primitive; the NTT
// and scaling reuse the existing traced-NTT / scalar-mul machinery.
//
// We reproduce Lattigo's EXACT centering arithmetic (BRedAdd, the same pos/neg
// branch as ModUp) so the captured lift is bit-identical to what the engine
// computes, while exposing it over canonical residues for the prover. The proof
// binds a -> lifted_i via the shared centered integer x with the real range
// bound |x| < q0/2 (a non-vacuous cross-field binding, like the rescale dropped
// limb but in the lift direction).

// ModUpCentered performs the centered CRT lift of a single q0-limb coefficient
// array `a` (coeff domain, residues mod q0) into every prime of the level-`levelQ`
// chain of `ringQ`, writing the lifted coeff-domain limbs into `out` (out.Coeffs[i]
// for i in 0..levelQ). out.Coeffs[0] is set to `a` (the q0 limb is unchanged).
// When sink != nil it emits the proof trace:
//
//   modup_operand[comp]      : a (the bound q0 coefficients)
//   modup_sign[comp]         : per-coeff centering flag (1 if a>=q0/2 i.e. negative x)
//   modup_lift[comp*L + i]   : lifted_i (coeff domain) for i in 1..levelQ
//
// This mirrors ModUp's lines for ctIn.Value[0] (centered around q, BRedAdd into
// each higher prime). `comp` selects the ciphertext component for region indexing.
func (r Ring) ModUpCentered(comp, levelQ int, a []uint64, out Poly, sink TraceSink) {
	q0 := r.SubRings[0].Modulus
	half := q0 >> 1
	N := r.N()

	// Operand: the q0 limb (also out.Coeffs[0]).
	op := make([]uint64, N)
	copy(op, a)
	copy(out.Coeffs[0], a)
	if sink != nil {
		sink.Poly("modup_operand", comp, op)
	}

	sign := make([]uint64, N) // 0 => x = a (>=0), 1 => x = a - q0 (<0)
	for j := 0; j < N; j++ {
		coeff := a[j]
		var pos, neg uint64 = 1, 0
		if coeff >= half {
			coeff = q0 - coeff
			pos, neg = 0, 1
		}
		sign[j] = neg
		for i := 1; i <= levelQ; i++ {
			qi := r.SubRings[i].Modulus
			tmp := BRedAdd(coeff, qi, r.SubRings[i].BRedConstant)
			out.Coeffs[i][j] = tmp*pos + (qi-tmp)*neg
		}
	}

	if sink != nil {
		sink.Poly("modup_sign", comp, sign)
		for i := 1; i <= levelQ; i++ {
			lifted := make([]uint64, N)
			copy(lifted, out.Coeffs[i])
			sink.Poly("modup_lift", comp*(levelQ+1)+i, lifted)
		}
	}
}

// NTTLiftTraced captures the per-stage forward NTT of the centered-lift limbs of
// the coeff-domain poly `in` (limbs lo..hi), emitting region<idx,stage> via
// sink.Stage with idx = idxBase+i — so the modup_ntt region for limb i aligns
// with that limb's modup_lift index. This binds the ModUp output eval_i =
// NTT(lift_i): the prover feeds these stages to the CT_NTT gadget (input = the
// lift already bound by RnsDigitBinding, output = the canonical last stage).
// `in` is NOT modified (the NTT is recomputed into a scratch); reuses
// ForwardNTTTraced so the captured stages are bit-identical to the engine's NTT.
// No-op when sink == nil (zero overhead in production).
func (r Ring) NTTLiftTraced(in Poly, lo, hi int, region string, idxBase int, sink TraceSink) {
	if sink == nil {
		return
	}
	N := r.N()
	scratch := make([]uint64, N)
	for i := lo; i <= hi; i++ {
		sr := r.SubRings[i]
		ForwardNTTTraced(in.Coeffs[i], scratch, N, sr.Modulus, sr.MRedConstant,
			sr.RootsForward, region, idxBase+i, sink)
	}
}
