package rlwe

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
	"github.com/tuneinsight/lattigo/v6/utils"
)

// GadgetProductHoistedMultiPTraced is GadgetProductHoisted with the vFHE proof
// trace, so the HOISTED key-switch lands on the same multi-P board every other
// key-switch already uses.
//
// WHY THIS COULD NOT JUST CALL THE NON-HOISTED TRACED FORM
// -------------------------------------------------------
// GadgetProductMultiPTraced derives the decomposition itself, so it holds each
// digit in COEFFICIENT form and emits `modup_coeff` plus the forward-NTT stages
// the board's CT_NTT gadget binds (NTT(modup_coeff) == the digit it MACs with).
// The hoisted form is handed digits that are already in the EVALUATION domain --
// and at ModUp's call site they have since been scaled by the message ratio, so
// the coefficient form of what actually gets multiplied is never materialised
// anywhere. There was nothing for the gadget to bind against.
//
// So the coefficient form is recovered here, by an inverse NTT of the digit. That
// is OBSERVATION, not recomputation: the NTT is an exact bijection on canonical
// residues, so the recovered value is *the* coefficient form of the digit the
// key-switch uses, and re-running the forward transform reproduces that digit
// exactly. The board then binds a true statement about the digit actually
// multiplied, which is the whole point -- unlike a capture that re-derives the
// operation on a copy and proves something adjacent to what ran.
//
// Everything else mirrors GadgetProductMultiPTraced: same regions, same index
// packing, same mod-down, so the runtime drains it with the existing flushKS and
// it assembles into the existing layout_relin_multip buffer.
func (eval Evaluator) GadgetProductHoistedMultiPTraced(levelQ int,
	BuffQPDecompQP []ringqp.Poly, gadgetCt *GadgetCiphertext, ct *Ciphertext,
	sink ring.TraceSink) error {

	levelP := gadgetCt.LevelP()
	if levelP < 1 || gadgetCt.BaseTwoDecomposition != 0 {
		return fmt.Errorf("GadgetProductHoistedMultiPTraced requires levelP>=1 and BaseTwoDecomposition=0; got levelP=%d pw2=%d",
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

	if len(BuffQPDecompQP) < d {
		return fmt.Errorf("GadgetProductHoistedMultiPTraced: got %d hoisted digits, need %d",
			len(BuffQPDecompQP), d)
	}

	poolQP := eval.pool.AtLevel(levelQ, levelP)
	buffQP1 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP1)
	buffQP2 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP2)

	// ctQP = MAC result mod QP: Q part in ct.Value, P part in the QP buffers.
	ctQP := &Element[ringqp.Poly]{}
	ctQP.Value = []ringqp.Poly{{Q: ct.Value[0], P: buffQP1.P}, {Q: ct.Value[1], P: buffQP2.P}}
	ctQP.MetaData = ct.MetaData

	// Scratch for the recovered coefficient form of one digit.
	coeffQP := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(coeffQP)

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

		// Recover the digit's coefficient form. The digit may be lazy (unreduced)
		// from the NTT and the scaling that produced it, so reduce first: the
		// forward transform below emits canonical residues, and the two have to be
		// the same value for NTT(modup_coeff) == the MAC operand to hold.
		ringQP.Reduce(BuffQPDecompQP[i], *coeffQP)
		ringQP.INTT(*coeffQP, *coeffQP)

		// Q factors.
		for u, s := range ringQ.SubRings[:levelQ+1] {
			mc := make([]uint64, N)
			for x := 0; x < N; x++ {
				mc[x] = coeffQP.Q.Coeffs[u][x] % s.Modulus
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
				mcP[x] = coeffQP.P.Coeffs[v][x] % sP.Modulus
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
