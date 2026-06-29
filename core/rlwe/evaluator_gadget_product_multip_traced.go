package rlwe

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
	"github.com/tuneinsight/lattigo/v6/utils"
)

// GadgetProductMultiPTraced is the MULTI-special-prime vFHE-instrumented clone of
// gadgetProductMultiplePLazy (evaluator_gadget_product.go:129) — the key-switch
// config the bootstrap CoeffsToSlots actually uses (levelP = nP-1 ≥ 1). Unlike the
// single-P path (GadgetProductTraced) there is NO diagonal target-reuse: the real
// engine decomposes via a GROUPED RNS decomposition (d = BaseRNSDecompositionVectorSize
// digit-groups, each spanning up to nP Q-primes) and base-extends every digit into
// EVERY RNS factor with a real NTT. So the trace is structurally simpler: every
// (factor fi, digit i) pair carries a real modup_coeff + modup_ntt.
//
// Emits the multi-P layout_relin regions (relin_multip_trace.go):
//   modup_coeff : digit i base-extended into factor fi (coeff)   idx fi*d+i
//   modup_ntt   : per-stage NTT of modup_coeff                    idx fi*d+i
//   evk         : de-Montgomerised key slice                      idx (i*ctOut+k)*F+fi
//   prod        : key MAC accumulator (mod QP, canonical)         idx k*F+fi
// Factor indexing: fi in [0,L) are Q primes, fi in [L,L+nP) the nP special primes.
// The mod-down half is emitted by ring.ModDownQPtoQNTTMultiPTraced.
func (eval Evaluator) GadgetProductMultiPTraced(levelQ int, cx ring.Poly, gadgetCt *GadgetCiphertext, ct *Ciphertext, sink ring.TraceSink) error {

	levelP := gadgetCt.LevelP()
	if levelP < 1 || gadgetCt.BaseTwoDecomposition != 0 {
		return fmt.Errorf("GadgetProductMultiPTraced requires levelP>=1 and BaseTwoDecomposition=0; got levelP=%d pw2=%d",
			levelP, gadgetCt.BaseTwoDecomposition)
	}
	levelQ = utils.Min(levelQ, gadgetCt.LevelQ())

	ringQP := eval.params.RingQP().AtLevel(levelQ, levelP)
	ringQ := ringQP.RingQ
	ringP := ringQP.RingP
	N := ringQ.N()
	L := levelQ + 1
	nP := levelP + 1
	F := L + nP
	d := eval.params.BaseRNSDecompositionVectorSize(levelQ, levelP)
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

	emitEvk := func(src []uint64, q, mred uint64, i, k, fi int) {
		evk := make([]uint64, N)
		for x := 0; x < N; x++ {
			evk[x] = ring.IMForm(src[x], q, mred)
		}
		sink.Poly("evk", (i*ctOut+k)*F+fi, evk)
	}

	var reduce int
	for i := 0; i < d; i++ {

		// Grouped base-extension of digit-group i into the full QP basis (coeff).
		eval.Decomposer.DecomposeAndSplit(levelQ, levelP, levelP+1, i, cxInvNTT, c2QP.Q, c2QP.P)

		// Q factors. For the in-group factors [i*nP, (i+1)*nP) the engine's
		// DecomposeSingleNTT COPIES the input (cxNTT) rather than using the grouped
		// base-conversion output (which differs from cx mod x for nP>1). We mirror
		// that by taking the digit's coeff there as the input's own coeff residue
		// (cxInvNTT); its forward NTT is exactly cxNTT, matching the copy, while
		// keeping the capture self-consistent (NTT(modup_coeff)==cw).
		for u, s := range ringQ.SubRings[:levelQ+1] {
			inGroup := u >= i*nP && u < (i+1)*nP
			mc := make([]uint64, N)
			if inGroup {
				for x := 0; x < N; x++ {
					mc[x] = cxInvNTT.Coeffs[u][x] % s.Modulus
				}
			} else {
				for x := 0; x < N; x++ {
					mc[x] = c2QP.Q.Coeffs[u][x] % s.Modulus
				}
			}
			sink.Poly("modup_coeff", u*d+i, mc)
			cw := make([]uint64, N)
			ring.ForwardNTTTraced(mc, cw, N, s.Modulus, s.MRedConstant, s.RootsForward, "modup_ntt", u*d+i, sink)
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

		// nP special primes (factor indices L..L+nP-1).
		for v, sP := range ringP.SubRings[:levelP+1] {
			fi := L + v
			mcP := make([]uint64, N)
			for x := 0; x < N; x++ {
				mcP[x] = c2QP.P.Coeffs[v][x] % sP.Modulus
			}
			sink.Poly("modup_coeff", fi*d+i, mcP)
			cwP := make([]uint64, N)
			ring.ForwardNTTTraced(mcP, cwP, N, sP.Modulus, sP.MRedConstant, sP.RootsForward, "modup_ntt", fi*d+i, sink)
			if i == 0 {
				sP.MulCoeffsMontgomeryLazy(el[i][0][0].P.Coeffs[v], cwP, ctQP.Value[0].P.Coeffs[v])
				sP.MulCoeffsMontgomeryLazy(el[i][0][1].P.Coeffs[v], cwP, ctQP.Value[1].P.Coeffs[v])
			} else {
				sP.MulCoeffsMontgomeryLazyThenAddLazy(el[i][0][0].P.Coeffs[v], cwP, ctQP.Value[0].P.Coeffs[v])
				sP.MulCoeffsMontgomeryLazyThenAddLazy(el[i][0][1].P.Coeffs[v], cwP, ctQP.Value[1].P.Coeffs[v])
			}
			for k := 0; k < ctOut; k++ {
				emitEvk(el[i][0][k].P.Coeffs[v], sP.Modulus, sP.MRedConstant, i, k, fi)
			}
		}

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
		for v, sP := range ringP.SubRings[:levelP+1] {
			pvP := make([]uint64, N)
			for x := 0; x < N; x++ {
				pvP[x] = ctQP.Value[k].P.Coeffs[v][x] % sP.Modulus
			}
			sink.Poly("prod", k*F+L+v, pvP)
		}
	}

	// Multi-P mod-down each component to basis Q (real internals + md_* / ks).
	for k := 0; k < ctOut; k++ {
		eval.BasisExtender.ModDownQPtoQNTTMultiPTraced(k, levelQ, levelP, ctQP.Value[k].Q, ctQP.Value[k].P, ct.Value[k], L, nP, sink)
	}
	return nil
}
