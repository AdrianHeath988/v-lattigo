package ring

// vFHE proof-trace instrumentation.
//
// This file (and the *_traced.go siblings) add SEAL-style trace capture to the
// fork: the production hot paths are left untouched, and parallel "_traced"
// clones reproduce the exact same math (non-unrolled, so each butterfly stage is
// well defined) while snapshotting the real internal scratch into a TraceSink.
//
// The proof gadgets verify over CANONICAL residues (fully reduced mod q, not
// Montgomery, not lazy). Montgomery form and lazy reduction are representation
// details of how Lattigo computes, not part of the FHE math, so every snapshot
// is reduced to its canonical residue before being emitted. Reducing a lazy
// value mod q yields exactly the canonical per-stage value the gadget expects,
// so the captured trace is bit-identical to a canonical recomputation while
// being sourced from the engine's actual execution.

// TraceSink receives the canonical-residue scratch produced by the traced
// variants. The fork stays free of any vFHE buffer-layout knowledge: the sink
// implementation (in the Lattigo runtime package) maps (region, idx, stage) to
// the byte offsets of vfhe::layout_rescale / layout_relin.
type TraceSink interface {
	// Poly emits one canonical polynomial for region stream idx (operand,
	// advice, divround coeff, decomposition digit, evk slice, key product,
	// key-switch output, result, ...).
	Poly(region string, idx int, vals []uint64)

	// Stage emits the canonical full-array state after one (i)NTT butterfly
	// stage. idx selects the sub-stream within the region (e.g. a limb, or
	// factor*L+digit); stage is the 0-based stage index. For the forward NTT the
	// last stage equals NTT(input); for the inverse NTT the last stage carries
	// the N^-1 normalization (see BackwardNTTTraced).
	Stage(region string, idx, stage int, vals []uint64)
}

// reduceCanonical returns v reduced to its canonical residue in [0, q). v may be
// a lazy value up to a few multiples of q (forward NTT output is < 6q, inverse
// < 2q); for q up to ~60 bits these stay well within uint64, so a plain modulo
// is exact and cheap on the (cold) trace path.
func reduceCanonical(v, q uint64) uint64 { return v % q }

// Rescale ORIGIN TAGS, the last element of an "rs_meta".
//
// A bootstrap rescales at four different places and they were indistinguishable
// once captured, so an operand that resolved to no producing proof named no stage
// and the only way to find its producer was to guess. Here so every emitter and the
// runtime read one definition.
const (
	RsOriginScaleDown   = 0 // ScaleDown's RescaleTo ladder
	RsOriginDFTStep     = 1 // the per-level-group rescale inside CoeffsToSlots/SlotsToCoeffs
	RsOriginDoubleAngle = 2 // EvalMod's double-angle rescale
	RsOriginPolyEval    = 3 // the Chebyshev/Paterson-Stockmeyer evaluator's rescale
)
