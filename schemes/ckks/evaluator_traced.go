package ckks

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/utils"
)

// RescaleTraced is the vFHE-instrumented clone of Rescale (evaluator.go:477). It
// performs the identical single-prime rescale but routes each component through
// ring.DivRoundByLastModulusNTTTraced, emitting the divide-and-round proof trace
// to sink. Only the single-rescale case (LevelsConsumedPerRescaling == 1) is
// instrumented — the provable CKKS configs use one prime per rescaling; callers
// fall back to the untraced Rescale otherwise.
func (eval Evaluator) RescaleTraced(op0, opOut *rlwe.Ciphertext, sink ring.TraceSink) (err error) {

	if op0.MetaData == nil || opOut.MetaData == nil {
		return fmt.Errorf("cannot RescaleTraced: op0.MetaData or opOut.MetaData is nil")
	}

	params := eval.GetParameters()

	if nbRescales := params.LevelsConsumedPerRescaling(); nbRescales != 1 {
		return fmt.Errorf("RescaleTraced supports only LevelsConsumedPerRescaling==1, got %d", nbRescales)
	}

	if op0.Level() < 1 {
		return fmt.Errorf("cannot RescaleTraced: input Ciphertext level is too low")
	}

	if op0 != opOut {
		opOut.Resize(op0.Degree(), op0.Level()-1)
	}

	*opOut.MetaData = *op0.MetaData

	ringQ := params.RingQ().AtLevel(op0.Level())

	opOut.Scale = opOut.Scale.Div(rlwe.NewScale(ringQ.SubRings[op0.Level()].Modulus))

	for i := range opOut.Value {
		ringQ.DivRoundByLastModulusNTTTraced(i, op0.Value[i], opOut.Value[i], sink)
	}

	if op0 == opOut {
		opOut.Resize(op0.Degree(), op0.Level()-1)
	}

	return
}

// RelinearizeTraced is the vFHE-instrumented clone of rlwe Relinearize
// (evaluator_evaluationkey.go:120). It relinearizes the size-3 ctIn into the
// size-2 opOut while emitting the relin proof regions: target (c2), c_in
// (c0,c1), the key-switch internals (via GadgetProductTraced), and result.
func (eval Evaluator) RelinearizeTraced(ctIn, opOut *rlwe.Ciphertext, sink ring.TraceSink) error {

	if ctIn.Degree() != 2 {
		return fmt.Errorf("cannot RelinearizeTraced: ctIn.Degree() must be 2, got %d", ctIn.Degree())
	}
	rlk, err := eval.CheckAndGetRelinearizationKey()
	if err != nil {
		return fmt.Errorf("cannot RelinearizeTraced: %w", err)
	}

	level := utils.Min(ctIn.Level(), opOut.Level())
	ringQ := eval.GetParameters().RingQ().AtLevel(level)
	L := level + 1

	// target = c2, c_in = (c0, c1).
	for l := 0; l < L; l++ {
		sink.Poly("target", l, ctIn.Value[2].Coeffs[l])
		sink.Poly("c_in", 0*L+l, ctIn.Value[0].Coeffs[l])
		sink.Poly("c_in", 1*L+l, ctIn.Value[1].Coeffs[l])
	}

	ctTmp := NewCiphertext(*eval.GetParameters(), 1, level)
	ctTmp.MetaData = ctIn.MetaData
	if err := eval.GadgetProductTraced(level, ctIn.Value[2], &rlk.GadgetCiphertext, ctTmp, sink); err != nil {
		return err
	}

	ringQ.Add(ctIn.Value[0], ctTmp.Value[0], opOut.Value[0])
	ringQ.Add(ctIn.Value[1], ctTmp.Value[1], opOut.Value[1])
	opOut.Resize(1, level)
	*opOut.MetaData = *ctIn.MetaData

	for k := 0; k < 2; k++ {
		for l := 0; l < L; l++ {
			sink.Poly("result", k*L+l, opOut.Value[k].Coeffs[l])
		}
	}
	return nil
}

// RelinearizeMultiPTraced is the MULTI-special-prime (nP>=2) vFHE-instrumented
// relinearization: it key-switches the degree-2 term c2 of the size-3 ctIn back
// into the size-2 opOut while emitting the multi-P key-switch proof regions
// (relin_multip layout, drained by the runtime's flushRelinMultiP). It is
// structurally identical to AutomorphismMultiPTraced (GadgetProductMultiPTraced +
// ÷P mod-down) EXCEPT: the key-switch target is c2 (not c1), the recombination adds
// BOTH c0 and c1 (so c_in = (c0,c1), not (c0,0)), and there is NO output permutation
// (the runtime supplies an IDENTITY automorphism index, so result = c_in + ks).
// Returns (L, nP, d) so the caller can emit the trailing "rl_meta" trigger.
func (eval Evaluator) RelinearizeMultiPTraced(ctIn, opOut *rlwe.Ciphertext, sink ring.TraceSink) (L, nP, d int, err error) {

	if ctIn.Degree() != 2 {
		return 0, 0, 0, fmt.Errorf("cannot RelinearizeMultiPTraced: ctIn.Degree() must be 2, got %d", ctIn.Degree())
	}
	rlk, err := eval.CheckAndGetRelinearizationKey()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("cannot RelinearizeMultiPTraced: %w", err)
	}
	levelP := rlk.GadgetCiphertext.LevelP()
	if levelP < 1 {
		return 0, 0, 0, fmt.Errorf("cannot RelinearizeMultiPTraced: requires nP>=2 (levelP>=1), got levelP=%d", levelP)
	}

	level := utils.Min(ctIn.Level(), opOut.Level())
	ringQ := eval.GetParameters().RingQ().AtLevel(level)
	L = level + 1
	nP = levelP + 1
	d = eval.GetParameters().BaseRNSDecompositionVectorSize(level, levelP)

	// c_in = (c0, c1) — both components (relin recombines into both, unlike rotation).
	for l := 0; l < L; l++ {
		sink.Poly("c_in", 0*L+l, ctIn.Value[0].Coeffs[l])
		sink.Poly("c_in", 1*L+l, ctIn.Value[1].Coeffs[l])
	}

	ctTmp := NewCiphertext(*eval.GetParameters(), 1, level)
	ctTmp.MetaData = ctIn.MetaData
	if err = eval.GadgetProductMultiPTraced(level, ctIn.Value[2], &rlk.GadgetCiphertext, ctTmp, sink); err != nil {
		return 0, 0, 0, err
	}

	ringQ.Add(ctIn.Value[0], ctTmp.Value[0], opOut.Value[0])
	ringQ.Add(ctIn.Value[1], ctTmp.Value[1], opOut.Value[1])
	opOut.Resize(1, level)
	*opOut.MetaData = *ctIn.MetaData

	for k := 0; k < 2; k++ {
		for l := 0; l < L; l++ {
			sink.Poly("result", k*L+l, opOut.Value[k].Coeffs[l])
		}
	}
	return L, nP, d, nil
}

// AutomorphismTraced is the vFHE-instrumented clone of rlwe Automorphism
// (evaluator_automorphism.go:13). It key-switches c1 with the Galois key,
// recombines (c0+ks0, ks1), and permutes the OUTPUT by galEl — emitting the same
// relin-shaped regions (target=c1, c_in=(c0,0), key-switch internals, result =
// the permuted output). The caller appends the automorphism index so the prover
// can bind result = gamma(c_in + ks).
func (eval Evaluator) AutomorphismTraced(ctIn *rlwe.Ciphertext, galEl uint64, opOut *rlwe.Ciphertext, sink ring.TraceSink) error {

	if ctIn.Degree() != 1 || opOut.Degree() != 1 {
		return fmt.Errorf("cannot AutomorphismTraced: input and output must be degree 1")
	}
	evk, err := eval.CheckAndGetGaloisKey(galEl)
	if err != nil {
		return fmt.Errorf("cannot AutomorphismTraced: %w", err)
	}

	level := utils.Min(ctIn.Level(), opOut.Level())
	opOut.Resize(opOut.Degree(), level)
	ringQ := eval.GetParameters().RingQ().AtLevel(level)
	N := ringQ.N()
	L := level + 1

	zero := make([]uint64, N)
	for l := 0; l < L; l++ {
		sink.Poly("target", l, ctIn.Value[1].Coeffs[l])
		sink.Poly("c_in", 0*L+l, ctIn.Value[0].Coeffs[l])
		sink.Poly("c_in", 1*L+l, zero)
	}

	ctTmp := NewCiphertext(*eval.GetParameters(), 1, level)
	ctTmp.MetaData = ctIn.MetaData
	if err := eval.GadgetProductTraced(level, ctIn.Value[1], &evk.GadgetCiphertext, ctTmp, sink); err != nil {
		return err
	}

	// inter = (c0 + ks0, ks1), then permute the output by galEl.
	ringQ.Add(ctTmp.Value[0], ctIn.Value[0], ctTmp.Value[0])
	index, err := ring.AutomorphismNTTIndex(N, eval.GetParameters().RingQ().NthRoot(), galEl)
	if err != nil {
		return fmt.Errorf("AutomorphismTraced index: %w", err)
	}
	ringQ.AutomorphismNTTWithIndex(ctTmp.Value[0], index, opOut.Value[0])
	ringQ.AutomorphismNTTWithIndex(ctTmp.Value[1], index, opOut.Value[1])
	*opOut.MetaData = *ctIn.MetaData

	for k := 0; k < 2; k++ {
		for l := 0; l < L; l++ {
			sink.Poly("result", k*L+l, opOut.Value[k].Coeffs[l])
		}
	}
	return nil
}

// AutomorphismMultiPTraced is the MULTI-special-prime (nP>=2) variant of
// AutomorphismTraced: identical execution (key-switch c1, inter=(c0+ks0,ks1),
// output=γ(inter)) but routes through GadgetProductMultiPTraced (grouped RNS
// decomposition over nP special primes, multi-P mod-down). Emits c_in + result;
// no `target` region (the multi-P gadget has no diagonal reuse).
func (eval Evaluator) AutomorphismMultiPTraced(ctIn *rlwe.Ciphertext, galEl uint64, opOut *rlwe.Ciphertext, sink ring.TraceSink) error {

	if ctIn.Degree() != 1 || opOut.Degree() != 1 {
		return fmt.Errorf("cannot AutomorphismMultiPTraced: input and output must be degree 1")
	}
	evk, err := eval.CheckAndGetGaloisKey(galEl)
	if err != nil {
		return fmt.Errorf("cannot AutomorphismMultiPTraced: %w", err)
	}

	level := utils.Min(ctIn.Level(), opOut.Level())
	opOut.Resize(opOut.Degree(), level)
	ringQ := eval.GetParameters().RingQ().AtLevel(level)
	N := ringQ.N()
	L := level + 1

	zero := make([]uint64, N)
	for l := 0; l < L; l++ {
		sink.Poly("c_in", 0*L+l, ctIn.Value[0].Coeffs[l])
		sink.Poly("c_in", 1*L+l, zero)
	}

	ctTmp := NewCiphertext(*eval.GetParameters(), 1, level)
	ctTmp.MetaData = ctIn.MetaData
	if err := eval.GadgetProductMultiPTraced(level, ctIn.Value[1], &evk.GadgetCiphertext, ctTmp, sink); err != nil {
		return err
	}

	ringQ.Add(ctTmp.Value[0], ctIn.Value[0], ctTmp.Value[0])
	idx, err := ring.AutomorphismNTTIndex(N, eval.GetParameters().RingQ().NthRoot(), galEl)
	if err != nil {
		return fmt.Errorf("AutomorphismMultiPTraced index: %w", err)
	}
	ringQ.AutomorphismNTTWithIndex(ctTmp.Value[0], idx, opOut.Value[0])
	ringQ.AutomorphismNTTWithIndex(ctTmp.Value[1], idx, opOut.Value[1])
	*opOut.MetaData = *ctIn.MetaData

	for k := 0; k < 2; k++ {
		for l := 0; l < L; l++ {
			sink.Poly("result", k*L+l, opOut.Value[k].Coeffs[l])
		}
	}
	return nil
}
