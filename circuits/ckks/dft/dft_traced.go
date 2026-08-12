package dft

// vFHE trace capture for the homomorphic DFT (CoeffsToSlots). dft() emits, for
// each linear-transform matrix step, the operand and result ciphertexts (both
// components, every limb at the step's level). This records the C2S op stream's
// intermediate ciphertext state — the foundation for a later linear-transform
// proof. No proof gadget yet; the per-step I/O is just captured.

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// c2sIdx packs (step, component, limb) into one trace index. step < 256,
// component < 8, limb < 64 (ample for any CoeffsToSlots depth / chain).
func c2sIdx(step, comp, limb int) int { return (step*8+comp)*64 + limb }

// dftPrefixSink wraps a TraceSink with a region prefix, so the per-step rescale
// (RescaleTraced emits operand/intt/advice/divround/ntt/result) lands under "rs_*"
// keys disjoint from the DFT's other regions.
type dftPrefixSink struct {
	inner  ring.TraceSink
	prefix string
}

func (p dftPrefixSink) Poly(region string, idx int, vals []uint64) {
	p.inner.Poly(p.prefix+region, idx, vals)
}
func (p dftPrefixSink) Stage(region string, idx, stage int, vals []uint64) {
	p.inner.Stage(p.prefix+region, idx, stage, vals)
}

// emitDFTStep emits one DFT matrix step under region prefix `pfx` ("c2s" for
// CoeffsToSlots, "s2c" for SlotsToCoeffs): a meta record [level, ctSize] plus the
// operand and result limbs.
func emitDFTStep(sink ring.TraceSink, pfx string, step int, operand, result *rlwe.Ciphertext) {
	level := result.Level()
	ctSize := len(result.Value)
	sink.Poly(pfx+"_meta", step, []uint64{uint64(level), uint64(ctSize)})
	for k := 0; k < ctSize; k++ {
		for i := 0; i <= level; i++ {
			sink.Poly(pfx+"_operand", c2sIdx(step, k, i), append([]uint64{}, operand.Value[k].Coeffs[i]...))
			sink.Poly(pfx+"_result", c2sIdx(step, k, i), append([]uint64{}, result.Value[k].Coeffs[i]...))
		}
	}
}

// ckksEval exposes the underlying *ckks.Evaluator, which the scalar-op capture
// needs to encode a public constant to its reference plaintext. Returns nil when
// tracing is off or the evaluator is some other implementation, so every call
// site degrades to "no reference" rather than to a panic.
func (eval *Evaluator) ckksEval() *ckks.Evaluator {
	if eval.TraceSink == nil {
		return nil
	}
	return eval.Evaluator
}

// ksTraced runs one automorphism through the traced path and closes it with the
// "ks_meta" trigger the runtime drains on. AutomorphismMultiPTraced emits the
// whole board (c_in, the grouped key-switch, the mod-down, result, idx); only the
// meta -- which carries the shape the assembler needs -- is the caller's.
//
// BEST-EFFORT: capture must never change the result, so any failure falls back to
// the production call. That keeps a missing Galois key or an unsupported levelP
// from turning a working bootstrap into a broken one; the op then shows up as an
// absent proof rather than a wrong answer.
func (eval *Evaluator) ksTraced(ctIn *rlwe.Ciphertext, galEl uint64,
	opOut *rlwe.Ciphertext) error {
	params := eval.parameters
	levelP := params.MaxLevelP()
	if eval.TraceSink == nil || levelP < 1 {
		return eval.Automorphism(ctIn, galEl, opOut)
	}
	sink := dftPrefixSink{inner: eval.TraceSink, prefix: "ks_"}
	if err := eval.AutomorphismMultiPTraced(ctIn, galEl, opOut, sink); err != nil {
		return eval.Automorphism(ctIn, galEl, opOut)
	}
	level := opOut.Level()
	eval.TraceSink.Poly("ks_meta", 0, []uint64{
		uint64(params.N()), uint64(level + 1), uint64(levelP + 1),
		uint64(params.BaseRNSDecompositionVectorSize(level, levelP)), galEl})
	return nil
}

// conjugateTraced is Conjugate with the key-switch captured.
func (eval *Evaluator) conjugateTraced(ctIn, opOut *rlwe.Ciphertext) error {
	if eval.TraceSink == nil {
		return eval.Conjugate(ctIn, opOut)
	}
	return eval.ksTraced(ctIn, eval.parameters.GaloisElementOrderTwoOrthogonalSubgroup(), opOut)
}

// rotateTraced is an IN-PLACE Rotate with the key-switch captured. The
// automorphism permutes, so it cannot be done in place: the traced path writes to
// a fresh ciphertext and the result is copied back, matching what Rotate does
// internally.
func (eval *Evaluator) rotateTraced(ct *rlwe.Ciphertext, k int) error {
	if eval.TraceSink == nil {
		return eval.Rotate(ct, k, ct)
	}
	out := ckks.NewCiphertext(eval.parameters, 1, ct.Level())
	if err := eval.ksTraced(ct, eval.parameters.GaloisElement(k), out); err != nil {
		return err
	}
	ct.Copy(out)
	return nil
}
