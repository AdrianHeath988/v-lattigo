package rlwe

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
	"github.com/tuneinsight/lattigo/v6/utils"
)

// GadgetProductTraced is the vFHE-instrumented clone of GadgetProduct
// (evaluator_gadget_product.go:16) for the single-special-prime, no-bit-decomp
// key-switch (relin/rotate) — the provable config. It computes the identical
// ct = <decomp(cx), gadget>/P mod Q, but emits every key-switch region the prover
// binds (the mod-up / MAC half of vfhe::layout_relin; the mod-down half is
// emitted by ring.ModDownQPtoQNTTTraced):
//
//   modup_coeff : digit J base-extended into factor fi (coeff)   [fi][J]
//   modup_ntt   : per-stage NTT of modup_coeff (off-diagonal)    [fi][J][stage]
//   evk         : de-Montgomerised relin/galois key slice        [(J,k),fi]
//   prod        : key MAC accumulator (mod QP, canonical)        [k][fi]
//
// Factor indexing matches the layout: fi in [0,L) are the Q primes, fi==L is the
// single special prime P. Faithfulness: digits come from the real
// DecomposeAndSplit, the MAC uses the real keys, and the NTT stages come from the
// engine's own butterflies (traced clone) — no reconstruction. The diagonal
// (fi==J) reuses the NTT-form target in the gadget, so its modup_ntt stages are
// not emitted, matching the layout.
func (eval Evaluator) GadgetProductTraced(levelQ int, cx ring.Poly, gadgetCt *GadgetCiphertext, ct *Ciphertext, sink ring.TraceSink) error {

	levelP := gadgetCt.LevelP()
	if levelP != 0 || gadgetCt.BaseTwoDecomposition != 0 {
		return fmt.Errorf("GadgetProductTraced requires single special prime (levelP=0) and BaseTwoDecomposition=0; got levelP=%d pw2=%d",
			levelP, gadgetCt.BaseTwoDecomposition)
	}
	levelQ = utils.Min(levelQ, gadgetCt.LevelQ())

	ringQP := eval.params.RingQP().AtLevel(levelQ, levelP)
	ringQ := ringQP.RingQ
	ringP := ringQP.RingP
	N := ringQ.N()
	L := levelQ + 1
	F := L + 1 // L Q-factors + 1 special prime P
	const ctOut = 2

	poolQP := eval.pool.AtLevel(levelQ, levelP)
	buffQP1 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP1)
	buffQP2 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP2)

	// ctQP = MAC result mod QP: Q part in ct.Value, P part in the QP buffers.
	ctQP := &Element[ringqp.Poly]{}
	ctQP.Value = []ringqp.Poly{{Q: ct.Value[0], P: buffQP1.P}, {Q: ct.Value[1], P: buffQP2.P}}
	ctQP.MetaData = ct.MetaData

	buffInv := poolQP.GetBuffPoly()
	defer poolQP.RecycleBuffPoly(buffInv)
	cxInvNTT := *buffInv
	ringQ.INTT(cx, cxInvNTT)

	c2QP := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(c2QP)

	QiOverF := eval.params.QiOverflowMargin(levelQ) >> 1
	PiOverF := eval.params.PiOverflowMargin(levelP) >> 1

	el := gadgetCt.Value

	emitEvk := func(src []uint64, q, mred uint64, J, k, fi int) {
		evk := make([]uint64, N)
		for x := 0; x < N; x++ {
			evk[x] = ring.IMForm(src[x], q, mred)
		}
		sink.Poly("evk", (J*ctOut+k)*F+fi, evk)
	}

	var reduce int
	for i := 0; i < L; i++ {
		eval.Decomposer.DecomposeAndSplit(levelQ, levelP, levelP+1, i, cxInvNTT, c2QP.Q, c2QP.P)

		for u, s := range ringQ.SubRings[:levelQ+1] {
			mc := make([]uint64, N)
			for x := 0; x < N; x++ {
				mc[x] = c2QP.Q.Coeffs[u][x] % s.Modulus
			}
			sink.Poly("modup_coeff", u*L+i, mc)

			// NTT of the digit in factor u; the diagonal (u==i) reuses the target
			// in the gadget, so emit its stages only off-diagonal.
			cw := make([]uint64, N)
			var stageSink ring.TraceSink
			if u != i {
				stageSink = sink
			}
			ring.ForwardNTTTraced(mc, cw, N, s.Modulus, s.MRedConstant, s.RootsForward, "modup_ntt", u*L+i, stageSink)

			if i == 0 {
				s.MulCoeffsMontgomeryLazy(el[i][0][0].Q.Coeffs[u], cw, ctQP.Value[0].Q.Coeffs[u])
				s.MulCoeffsMontgomeryLazy(el[i][0][1].Q.Coeffs[u], cw, ctQP.Value[1].Q.Coeffs[u])
			} else {
				s.MulCoeffsMontgomeryLazyThenAddLazy(el[i][0][0].Q.Coeffs[u], cw, ctQP.Value[0].Q.Coeffs[u])
				s.MulCoeffsMontgomeryLazyThenAddLazy(el[i][0][1].Q.Coeffs[u], cw, ctQP.Value[1].Q.Coeffs[u])
			}
			for k := 0; k < ctOut; k++ {
				emitEvk(el[i][0][k].Q.Coeffs[u], s.Modulus, s.MRedConstant, i, k, u)
			}
		}

		// Single special prime P (factor index L).
		sP := ringP.SubRings[0]
		mcP := make([]uint64, N)
		for x := 0; x < N; x++ {
			mcP[x] = c2QP.P.Coeffs[0][x] % sP.Modulus
		}
		sink.Poly("modup_coeff", L*L+i, mcP)
		cwP := make([]uint64, N)
		ring.ForwardNTTTraced(mcP, cwP, N, sP.Modulus, sP.MRedConstant, sP.RootsForward, "modup_ntt", L*L+i, sink)
		if i == 0 {
			sP.MulCoeffsMontgomeryLazy(el[i][0][0].P.Coeffs[0], cwP, ctQP.Value[0].P.Coeffs[0])
			sP.MulCoeffsMontgomeryLazy(el[i][0][1].P.Coeffs[0], cwP, ctQP.Value[1].P.Coeffs[0])
		} else {
			sP.MulCoeffsMontgomeryLazyThenAddLazy(el[i][0][0].P.Coeffs[0], cwP, ctQP.Value[0].P.Coeffs[0])
			sP.MulCoeffsMontgomeryLazyThenAddLazy(el[i][0][1].P.Coeffs[0], cwP, ctQP.Value[1].P.Coeffs[0])
		}
		for k := 0; k < ctOut; k++ {
			emitEvk(el[i][0][k].P.Coeffs[0], sP.Modulus, sP.MRedConstant, i, k, L)
		}

		// Periodic reduction to keep the lazy MAC within uint64 (essential at the
		// 60-bit / 10-limb regime), matching production's overflow management.
		if reduce%QiOverF == QiOverF-1 {
			ringQ.Reduce(ctQP.Value[0].Q, ctQP.Value[0].Q)
			ringQ.Reduce(ctQP.Value[1].Q, ctQP.Value[1].Q)
		}
		if reduce%PiOverF == PiOverF-1 {
			ringP.Reduce(ctQP.Value[0].P, ctQP.Value[0].P)
			ringP.Reduce(ctQP.Value[1].P, ctQP.Value[1].P)
		}
		reduce++
	}
	if reduce%QiOverF != 0 {
		ringQ.Reduce(ctQP.Value[0].Q, ctQP.Value[0].Q)
		ringQ.Reduce(ctQP.Value[1].Q, ctQP.Value[1].Q)
	}
	if reduce%PiOverF != 0 {
		ringP.Reduce(ctQP.Value[0].P, ctQP.Value[0].P)
		ringP.Reduce(ctQP.Value[1].P, ctQP.Value[1].P)
	}

	// prod[k][fi] (canonical) — emit BEFORE mod-down overwrites ct.Value (Q part).
	for k := 0; k < ctOut; k++ {
		for u, s := range ringQ.SubRings[:levelQ+1] {
			pv := make([]uint64, N)
			for x := 0; x < N; x++ {
				pv[x] = ctQP.Value[k].Q.Coeffs[u][x] % s.Modulus
			}
			sink.Poly("prod", k*F+u, pv)
		}
		sP := ringP.SubRings[0]
		pvP := make([]uint64, N)
		for x := 0; x < N; x++ {
			pvP[x] = ctQP.Value[k].P.Coeffs[0][x] % sP.Modulus
		}
		sink.Poly("prod", k*F+L, pvP)
	}

	// Mod-down each component to basis Q (real internals + md_* / ks emission).
	for k := 0; k < ctOut; k++ {
		eval.BasisExtender.ModDownQPtoQNTTTraced(k, levelQ, levelP, ctQP.Value[k].Q, ctQP.Value[k].P, ct.Value[k], L, sink)
	}
	return nil
}
