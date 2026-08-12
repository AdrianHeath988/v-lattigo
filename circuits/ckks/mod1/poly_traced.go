package mod1

// vFHE trace capture for the EvalMod homomorphic polynomial evaluation (the Chebyshev
// mod1 poly, kind 1, and the optional arcsine poly, kind 3). The generic Paterson-
// Stockmeyer evaluator (circuits/common/polynomial) drives its primitive ops through
// a schemes.Evaluator interface; tracingPolyEvaluator DECORATES that interface to ALSO
// capture the poly-eval's HEAVY ops — relinearizations (multi-P key-switch) and rescales
// — reusing the exact streamed-proof paths the double-angle already validates
// (flushRelinMultiP / flushRescale). No new gadget: a poly-eval is just squarings
// (power basis) + plaintext-muls + adds + rescales over the same primitives.
//
// Capture is by RECOMPUTE on copies (never mutating the operands the production op
// consumes), so the numerically-sensitive EvalMod path is byte-for-byte unchanged —
// the decorator delegates every method to the real evaluator and only adds tracing.

import (
	"fmt"
	"os"

	"github.com/tuneinsight/lattigo/v6/circuits/common/vfhetrace"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Linear-op type tags (must match lib/Runtime/lattigo/evalmod_trace.go).
const (
	loADD  = 0 // ct + ct
	loMUL  = 1 // ct × ct (size-3 tensor)
	loPADD = 2 // ct + scalar/plaintext
	loMTA  = 3 // MulThenAdd: r = b + coeff·a
	loSUB  = 4 // ct - ct (the Chebyshev power-basis recurrence)
)

// emitLinOp emits one linear op (ciphertexts a, b[, nil], r) under "lo_*" regions +
// the "lo_meta" trigger the runtime drains via flushLinOp. Polys are dumped at the
// RESULT level L = r.Level()+1 (operands truncated to those limbs; the relation holds
// mod the first L primes). No-op when sink is nil. For SCALAR ops (PADD/MTA) refReal/
// refImag carry the INDEPENDENT plaintext reference (per-limb constants the public
// coefficient encodes to — hole-1 binding); nil for ct-only ops.
func emitLinOp(sink ring.TraceSink, opType int, a, b, r *rlwe.Ciphertext, refPt [][]uint64) {
	if sink == nil || a == nil || r == nil || len(a.Value) == 0 {
		return
	}
	L := r.Level() + 1
	put := func(region string, ct *rlwe.Ciphertext) {
		if ct == nil {
			return
		}
		for p := 0; p < len(ct.Value); p++ {
			for l := 0; l < L && l < len(ct.Value[p].Coeffs); l++ {
				sink.Poly(region, p*L+l, append([]uint64{}, ct.Value[p].Coeffs[l]...))
			}
		}
	}
	put("lo_a", a)
	put("lo_b", b)
	put("lo_r", r)
	hasRef := 0
	if refPt != nil { // full encoded-plaintext reference (one poly per limb)
		hasRef = 1
		for l := 0; l < L && l < len(refPt); l++ {
			if refPt[l] != nil {
				sink.Poly("lo_ptref", l, append([]uint64{}, refPt[l]...))
			}
		}
	}
	bSize := 0
	if b != nil {
		bSize = len(b.Value)
	}
	N := len(a.Value[0].Coeffs[0])
	sink.Poly("lo_meta", 0, []uint64{uint64(N), uint64(L), uint64(opType),
		uint64(len(a.Value)), uint64(bSize), uint64(len(r.Value)), uint64(hasRef)})
}

// scalarRefPlain computes the INDEPENDENT reference plaintext for a scalar op: it encodes
// the coefficient `op1` into a fresh ZERO ciphertext's component-0 via the engine's own
// scalar-add at the given scale/level — i.e. exactly the plaintext the real op adds, but
// derived from the PUBLIC coefficient (op1) + PUBLIC scale, NOT from the prover's I/O. The
// hole-1 anchor. Returns one full poly per limb (the engine's exact encoding, any layout).
func scalarRefPlain(eval *ckks.Evaluator, level int, scale rlwe.Scale, op1 rlwe.Operand) [][]uint64 {
	ref := ckks.NewCiphertext(*eval.GetParameters(), 1, level)
	ref.Scale = scale
	if err := eval.Add(ref, op1, ref); err != nil {
		return nil
	}
	L := level + 1
	out := make([][]uint64, L)
	for l := 0; l < L && l < len(ref.Value[0].Coeffs); l++ {
		out[l] = append([]uint64{}, ref.Value[0].Coeffs[l]...)
	}
	return out
}

// asCT returns op as a *rlwe.Ciphertext, or nil if op is a scalar/plaintext.
func asCT(op rlwe.Operand) *rlwe.Ciphertext {
	if ct, ok := op.(*rlwe.Ciphertext); ok {
		return ct
	}
	return nil
}

// tracingPolyEvaluator wraps the real schemes.Evaluator (a *ckks.Evaluator). All
// methods delegate via the embedded interface; Relinearize / Rescale / MulRelinNew
// additionally capture their key-switch / rescale trace, and Add / Mul / MulNew /
// MulThenAdd capture the linear (add / plaintext-mul / tensor) op stream.
type tracingPolyEvaluator struct {
	schemes.Evaluator                 // the real evaluator (delegate target)
	ckksEval          *ckks.Evaluator // same object, typed for the traced clones
	sink              ring.TraceSink
}

// Add captures ct+ct (loADD) or ct+scalar (loPADD). The op is in-place (opOut==op0 in
// the poly-eval), so the first operand is snapshotted before delegating.
func (t *tracingPolyEvaluator) Add(op0 *rlwe.Ciphertext, op1 rlwe.Operand, opOut *rlwe.Ciphertext) error {
	var before *rlwe.Ciphertext
	if t.sink != nil {
		before = op0.CopyNew()
	}
	err := t.Evaluator.Add(op0, op1, opOut)
	if err == nil && t.sink != nil {
		if op1ct := asCT(op1); op1ct != nil {
			b := op1ct
			if op1ct == op0 { // doubling: op1 aliases the (now mutated) op0
				b = before
			}
			emitLinOp(t.sink, loADD, before, b, opOut, nil)
		} else {
			// PADD: the scalar is encoded at op0.Scale (unchanged by the add). Bind the
			// plaintext to the public coefficient via the engine-encoded reference (hole 1).
			ref := scalarRefPlain(t.ckksEval, before.Level(), before.Scale, op1)
			emitLinOp(t.sink, loPADD, before, nil, opOut, ref)
		}
	}
	return err
}

// Sub captures ct-ct: the Chebyshev power-basis recurrence's subtraction,
// T_{a+b} = 2*T_a*T_b - T_{a-b}.
//
// It was the one arithmetic method of the polynomial evaluator this decorator did
// NOT override, so the power-basis element it produces had no board and no trace at
// all -- and the rescale that follows it (power_basis.go: GenPower) then had an
// operand no proof produced. Measured as the last two unbound hand-offs in the
// bootstrap, and the reason they appeared in no other captured region: nothing
// emitted the value.
//
// The board already existed (loSUB -> the SUB board); only the interception was
// missing. In-place like Add, so op0 is snapshotted before delegating.
func (t *tracingPolyEvaluator) Sub(op0 *rlwe.Ciphertext, op1 rlwe.Operand, opOut *rlwe.Ciphertext) error {
	var before *rlwe.Ciphertext
	if t.sink != nil {
		before = op0.CopyNew()
	}
	// SCALE MATCHING. The SUB board proves r = a - b, and that is only what ckks.Sub
	// computes when the operand scales already agree. When they differ it scales ONE
	// side by an integer ratio first (evaluator.go, the `cmp` branches), so the real
	// relation is r = k*a - b or r = a - k*b. Emitting a plain SUB there would prove
	// a relation the engine never performed -- the capture self-check caught exactly
	// that.
	//
	// One side scaled by a public integer, then a plain subtraction, is a PMUL
	// followed by a SUB, and both boards already exist. The intermediate is a
	// captured region on both halves (the PMUL's result and the SUB's operand), so
	// the edge between them falls out of the ordinary fingerprint discovery.
	op1ct := asCT(op1)
	var beforeB *rlwe.Ciphertext
	if t.sink != nil && op1ct != nil {
		beforeB = op1ct.CopyNew()
	}
	err := t.Evaluator.Sub(op0, op1, opOut)
	if err != nil || t.sink == nil {
		return err
	}
	if op1ct == nil {
		// A scalar subtraction would be a PADD by the negated constant; the power
		// basis never does one, so it is deliberately not modelled here.
		return nil
	}
	if opOut != op0 {
		// ckks.Sub's scale matching differs by which operand aliases the output, and
		// only the opOut==op0 form occurs in the power basis. Guessing at the others
		// is how a board comes to prove the wrong relation.
		fmt.Fprintf(os.Stderr, "[vfhe] WARNING Sub NOT captured (opOut is not op0, a "+
			"form the capture does not model): its result will be bound to nothing\n")
		return nil
	}
	switch before.Scale.Cmp(beforeB.Scale) {
	case 0:
		emitLinOp(t.sink, loSUB, before, beforeB, opOut, nil)
	case 1: // a has the larger scale: b is scaled up to match
		t.emitScaledSub(before, beforeB, opOut, before.Scale.Div(beforeB.Scale), false)
	default: // b has the larger scale: a is scaled up
		t.emitScaledSub(before, beforeB, opOut, beforeB.Scale.Div(before.Scale), true)
	}
	return nil
}

// emitScaledSub emits the two boards of a scale-matched subtraction: the integer
// PMUL that brings one side to the other's scale, then the plain SUB.
//
// THE PLAINTEXT REFERENCE COMES FROM THE ENGINE, via MulRefPlain. ckks.Mul by a
// *big.Int is NOT a raw modular scalar multiply: it converts the scalar to a
// bignum.Complex, derives an RNS scalar pair through bigComplexToRNSScalar (which
// rounds), and applies MulDoubleRNSScalar. A hand-rolled "ratio mod q_l broadcast"
// -- which is the right model for ring.MulScalar, and is what ModUp's scalings
// use -- is therefore the WRONG reference here, and the coefficient binding
// rejected it. MulRefPlain instead multiplies the engine's own Mul over an
// all-ones ciphertext, so the reference is the engine's semantics rather than a
// model of them, while still being derived independently of the trace.
//
// A non-positive ratio means the engine did NOT scale (it guards on ratioInt > 0),
// so the plain relation still holds and one board is enough.
func (t *tracingPolyEvaluator) emitScaledSub(a, b, r *rlwe.Ciphertext,
	ratio rlwe.Scale, scaleA bool) {
	ratioInt, _ := ratio.Value.Int(nil)
	if ratioInt == nil || ratioInt.Sign() <= 0 {
		emitLinOp(t.sink, loSUB, a, b, r, nil)
		return
	}
	params := t.ckksEval.GetParameters()
	src := b
	if scaleA {
		src = a
	}
	scaled := ckks.NewCiphertext(*params, src.Degree(), r.Level())
	// COPY the metadata, do not alias it: Mul writes opOut.Scale from op0.Scale, so
	// a shared pointer changes the operand's scale underneath the operation.
	*scaled.MetaData = *src.MetaData
	if err := t.ckksEval.Mul(src, ratioInt, scaled); err != nil {
		fmt.Fprintf(os.Stderr, "[vfhe] WARNING scale-matched Sub NOT captured "+
			"(%v): its result will be bound to nothing\n", err)
		return
	}
	ref := vfhetrace.MulRefPlain(t.ckksEval, r.Level(), rlwe.NewScale(1), ratioInt)
	vfhetrace.EmitLinOp(t.sink, vfhetrace.OpPMUL, src, nil, scaled, ref)
	if scaleA {
		emitLinOp(t.sink, loSUB, scaled, b, r, nil)
	} else {
		emitLinOp(t.sink, loSUB, a, scaled, r, nil)
	}
}

// Mul captures both forms: ct×ct is the tensor product (loMUL), ct×scalar is a
// multiply by a PUBLIC coefficient (loPMUL). The scalar form used to delegate
// without capture, which left an arithmetic step of the polynomial evaluation
// with no board — and, being a coefficient application, one where substituting a
// different constant changes what EvalMod computes.
func (t *tracingPolyEvaluator) Mul(op0 *rlwe.Ciphertext, op1 rlwe.Operand, opOut *rlwe.Ciphertext) error {
	var before *rlwe.Ciphertext
	op1ct := asCT(op1)
	if t.sink != nil {
		before = op0.CopyNew()
	}
	err := t.Evaluator.Mul(op0, op1, opOut)
	if err != nil || t.sink == nil || before == nil {
		return err
	}
	if op1ct != nil {
		b := op1ct
		if op1ct == op0 {
			b = before
		}
		emitLinOp(t.sink, loMUL, before, b, opOut, nil)
		return nil
	}
	// Scalar: bind the plaintext to the public coefficient, encoded at the scale
	// the multiply consumes (opOut.Scale / op0.Scale), exactly as MulThenAdd does.
	ref := vfhetrace.MulRefPlain(t.ckksEval, opOut.Level(),
		opOut.Scale.Div(before.Scale), op1)
	vfhetrace.EmitLinOp(t.sink, vfhetrace.OpPMUL, before, nil, opOut, ref)
	return nil
}

// MulNew captures the (degree-2, unrelinearized) ct×ct product from the power basis.
func (t *tracingPolyEvaluator) MulNew(op0 *rlwe.Ciphertext, op1 rlwe.Operand) (*rlwe.Ciphertext, error) {
	out, err := t.Evaluator.MulNew(op0, op1)
	if err == nil && t.sink != nil {
		if op1ct := asCT(op1); op1ct != nil {
			emitLinOp(t.sink, loMUL, op0, op1ct, out, nil)
		}
	}
	return out, err
}

// MulThenAdd captures res += coeff·X (loMTA) — the baby-step coefficient application.
func (t *tracingPolyEvaluator) MulThenAdd(op0 *rlwe.Ciphertext, op1 rlwe.Operand, opOut *rlwe.Ciphertext) error {
	var resOld *rlwe.Ciphertext
	scalar := asCT(op1) == nil
	if t.sink != nil && scalar {
		resOld = opOut.CopyNew()
	}
	err := t.Evaluator.MulThenAdd(op0, op1, opOut)
	if err == nil && t.sink != nil && scalar && resOld != nil {
		// MTA: coeff encoded at scale res.Scale/X.Scale (so coeff·X lands at res.Scale).
		// Bind the recovered plaintext to the public coefficient via the engine-encoded ref.
		ref := scalarRefPlain(t.ckksEval, opOut.Level(), opOut.Scale.Div(op0.Scale), op1)
		emitLinOp(t.sink, loMTA, op0, resOld, opOut, ref) // a=X, b=res_old, r=res_new
	}
	return err
}

// captureRelin records the multi-P relinearization of a degree-2 ciphertext `in`
// (a copy is taken; `in` is not modified) and emits the "rl_meta" trigger.
func (t *tracingPolyEvaluator) captureRelin(in *rlwe.Ciphertext) {
	if t.sink == nil || in == nil || in.Degree() != 2 || in.Level() < 1 {
		return
	}
	if len(t.ckksEval.GetParameters().P()) < 2 {
		return
	}
	cp := in.CopyNew()
	tmp := ckks.NewCiphertext(*t.ckksEval.GetParameters(), 1, cp.Level())
	if L, nP, d, err := t.ckksEval.RelinearizeMultiPTraced(cp, tmp,
		mod1PrefixSink{inner: t.sink, prefix: "rl_"}); err == nil {
		t.sink.Poly("rl_meta", 0, []uint64{
			uint64(t.ckksEval.GetParameters().N()), uint64(L), uint64(nP), uint64(d)})
	}
}

// Relinearize captures the in-place relinearization of the degree-2 op0, then delegates.
func (t *tracingPolyEvaluator) Relinearize(op0, op1 *rlwe.Ciphertext) error {
	t.captureRelin(op0) // op0 is the degree-2 input, intact until the delegate runs
	return t.Evaluator.Relinearize(op0, op1)
}

// Rescale captures the single-prime rescale (on a copy of op0), then delegates.
func (t *tracingPolyEvaluator) Rescale(op0, op1 *rlwe.Ciphertext) error {
	if t.sink != nil && t.ckksEval.GetParameters().LevelsConsumedPerRescaling() == 1 && op0.Level() >= 1 {
		cp := op0.CopyNew()
		tmp := cp.CopyNew()
		inLimbs := op0.Level() + 1
		if err := t.ckksEval.RescaleTraced(cp, tmp,
			mod1PrefixSink{inner: t.sink, prefix: "rs_"}); err == nil {
			t.sink.Poly("rs_meta", 0, []uint64{uint64(t.ckksEval.GetParameters().N()),
				uint64(inLimbs), 0, 0, ring.RsOriginPolyEval})
		}
	}
	return t.Evaluator.Rescale(op0, op1)
}

// MulRelinNew delegates the production fused mul+relin, then captures the relin half
// by recomputing the (unrelinearized) product and relinearizing the copy.
func (t *tracingPolyEvaluator) MulRelinNew(op0 *rlwe.Ciphertext, op1 rlwe.Operand) (*rlwe.Ciphertext, error) {
	out, err := t.Evaluator.MulRelinNew(op0, op1)
	if err == nil && t.sink != nil {
		if prod, e := t.ckksEval.MulNew(op0, op1); e == nil {
			if op1ct := asCT(op1); op1ct != nil {
				emitLinOp(t.sink, loMUL, op0, op1ct, prod, nil) // tensor
			}
			t.captureRelin(prod) // relin (key-switch)
		}
	}
	return out, err
}

// withTracedPolyEval installs the tracing decorator on the polynomial evaluator for
// the duration of f (the homomorphic poly evaluation), then restores the real
// evaluator. No-op when tracing is off or the evaluator isn't a *ckks.Evaluator.
func (eval Evaluator) withTracedPolyEval(f func() error) error {
	if eval.TraceSink == nil {
		return f()
	}
	real := eval.PolynomialEvaluator.Evaluator.Evaluator
	ck, ok := real.(*ckks.Evaluator)
	if !ok {
		return f()
	}
	eval.PolynomialEvaluator.Evaluator.Evaluator = &tracingPolyEvaluator{
		Evaluator: real, ckksEval: ck, sink: eval.TraceSink,
	}
	defer func() { eval.PolynomialEvaluator.Evaluator.Evaluator = real }()
	return f()
}
