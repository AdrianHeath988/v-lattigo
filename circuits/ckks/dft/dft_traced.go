package dft

// vFHE trace capture for the homomorphic DFT (CoeffsToSlots). dft() emits, for
// each linear-transform matrix step, the operand and result ciphertexts (both
// components, every limb at the step's level). This records the C2S op stream's
// intermediate ciphertext state — the foundation for a later linear-transform
// proof. No proof gadget yet; the per-step I/O is just captured.

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
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
