// Package vfhetrace holds the vFHE capture helpers that more than one circuit
// package needs.
//
// The linear-op capture (emitLinOp + scalarRefPlain) started life private to the
// EvalMod poly-eval, because that was the only place emitting an op stream. It
// is not specific to EvalMod: an add is an add, and every remaining untraced
// bootstrap operation that is not a key-switch turned out to be an add, a
// subtract, or a multiply by a public constant. The homomorphic DFT's real/imag
// split and recombination, ModUp's message-ratio scaling and its post-key-switch
// add, and the poly-eval's scalar multiply all emit through here, into the SAME
// "lo_*" regions the runtime already drains into add / sub / plainop boards.
//
// It lives in circuits/common rather than in ring because it speaks in
// ciphertexts: ring cannot import rlwe.
package vfhetrace

import (
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Linear-op type tags. Must match lib/Runtime/lattigo/evalmod_trace.go.
const (
	OpADD  = 0 // ct + ct
	OpMUL  = 1 // ct x ct (size-3 tensor)
	OpPADD = 2 // ct + scalar/plaintext
	OpMTA  = 3 // MulThenAdd: r = b + coeff*a
	OpSUB  = 4 // ct - ct
	OpPMUL = 5 // ct * plaintext/scalar
)

// EmitLinOp emits one linear op (ciphertexts a, b[, nil], r) under "lo_*" regions
// plus the "lo_meta" trigger the runtime drains via flushLinOp. Polys are dumped
// at the RESULT level L = r.Level()+1 (operands truncated to those limbs; the
// relation holds mod the first L primes). No-op when sink is nil.
//
// For SCALAR ops (PADD / PMUL / MTA) refPt carries the INDEPENDENT plaintext
// reference: the per-limb polynomials the PUBLIC coefficient encodes to, derived
// from the coefficient and the scale rather than from the prover's I/O. nil for
// ct-only ops.
func EmitLinOp(sink ring.TraceSink, opType int, a, b, r *rlwe.Ciphertext, refPt [][]uint64) {
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
	if refPt != nil {
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

// ScalarRefPlain computes the INDEPENDENT reference plaintext for a scalar op: it
// encodes the coefficient `op1` into a fresh ZERO ciphertext's component-0 via the
// engine's own scalar-add at the given scale and level — i.e. exactly the
// plaintext the real op applies, but derived from the PUBLIC coefficient and the
// PUBLIC scale, never from the prover's I/O. Returns one full poly per limb.
//
// This is what lets a scalar op be checked against what the program SAYS the
// constant is, rather than against whatever the trace happened to contain.
func ScalarRefPlain(eval *ckks.Evaluator, level int, scale rlwe.Scale, op1 rlwe.Operand) [][]uint64 {
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

// MulRefPlain is ScalarRefPlain for a MULTIPLICATIVE constant: the plaintext a
// `Mul(ct, c)` applies. Lattigo lowers a complex scalar multiply to a
// piecewise-constant vector (the real part on the first N/2 coefficients, the
// imaginary part on the second), so the reference is obtained by multiplying a
// ciphertext of ONES rather than by adding to zeros.
func MulRefPlain(eval *ckks.Evaluator, level int, scale rlwe.Scale, op1 rlwe.Operand) [][]uint64 {
	params := *eval.GetParameters()
	ref := ckks.NewCiphertext(params, 1, level)
	ref.Scale = scale
	// c0 = 1 in the evaluation domain: every slot one, so the product IS the
	// encoded constant.
	ringQ := params.RingQ().AtLevel(level)
	for l := 0; l <= level && l < len(ref.Value[0].Coeffs); l++ {
		q := ringQ.SubRings[l].Modulus
		for x := range ref.Value[0].Coeffs[l] {
			ref.Value[0].Coeffs[l][x] = 1 % q
		}
	}
	ref.IsNTT = true
	if err := eval.Mul(ref, op1, ref); err != nil {
		return nil
	}
	L := level + 1
	out := make([][]uint64, L)
	for l := 0; l < L && l < len(ref.Value[0].Coeffs); l++ {
		out[l] = append([]uint64{}, ref.Value[0].Coeffs[l]...)
	}
	return out
}

// EmitPolyOp is EmitLinOp for operations the engine performs on raw polynomials
// rather than on ciphertexts.
//
// Several bootstrap steps never build an rlwe.Ciphertext around what they touch:
// ModUp scales `ctIn.Value[0]` and the hoisted decomposition buffers directly,
// and adds the key-switch output back into a bare poly. They are the same adds
// and plaintext multiplies as everywhere else and land on the same boards, so
// this emits into the same "lo_*" regions rather than inventing a second capture
// path — the runtime cannot tell the difference and does not need to.
//
// `a`, `b`, `r` are the operation's components (1 or 2 polys each; b may be nil).
// `limbs` is how many limbs of each to emit, which the caller knows and the polys
// do not: a poly allocated at the ring's top level is still only meaningful up to
// the ciphertext's current level.
func EmitPolyOp(sink ring.TraceSink, opType, limbs int, a, b, r []ring.Poly,
	refPt [][]uint64) {
	if sink == nil || len(a) == 0 || len(r) == 0 || limbs <= 0 {
		return
	}
	put := func(region string, polys []ring.Poly) {
		for p := 0; p < len(polys); p++ {
			for l := 0; l < limbs && l < len(polys[p].Coeffs); l++ {
				sink.Poly(region, p*limbs+l, append([]uint64{}, polys[p].Coeffs[l]...))
			}
		}
	}
	put("lo_a", a)
	if b != nil {
		put("lo_b", b)
	}
	put("lo_r", r)
	hasRef := 0
	if refPt != nil {
		hasRef = 1
		for l := 0; l < limbs && l < len(refPt); l++ {
			if refPt[l] != nil {
				sink.Poly("lo_ptref", l, append([]uint64{}, refPt[l]...))
			}
		}
	}
	N := len(a[0].Coeffs[0])
	sink.Poly("lo_meta", 0, []uint64{uint64(N), uint64(limbs), uint64(opType),
		uint64(len(a)), uint64(len(b)), uint64(len(r)), uint64(hasRef)})
}

// ScalarBroadcastRef is the plaintext reference for a multiply by an INTEGER
// scalar: the constant reduced into each limb, at every coefficient. Lattigo's
// ring.MulScalar is pointwise in the evaluation domain, so that is exactly the
// plaintext the plainop board multiplies by.
//
// Derived from the scalar and the moduli — both public — so it is the same kind
// of independent reference ScalarRefPlain provides for an encoded coefficient,
// not a value read back out of the trace.
// ScalarBigintBroadcastRef is ScalarBroadcastRef for a scalar too wide for a
// uint64.
//
// The one that matters is P = prod(P_v), the key-switch special-prime product: at
// nP=2 with 61-bit primes it is already ~122 bits, so the uint64 form cannot
// express it at all. It is the constant the BSGS inner sum scales its step input
// by (ctInTmp = P*c), and that multiply is a real op of the transform -- see the
// emission in MultiplyByDiagMatrixBSGS.
//
// Derived from the scalar and the moduli, both public, so it is an INDEPENDENT
// reference: the coefficient binding compares the prover's plaintext against this,
// not against something read back out of the trace.
func ScalarBigintBroadcastRef(scalar *big.Int, moduli []uint64, limbs, N int) [][]uint64 {
	if scalar == nil {
		return nil
	}
	out := make([][]uint64, limbs)
	m := new(big.Int)
	r := new(big.Int)
	for l := 0; l < limbs && l < len(moduli); l++ {
		v := r.Mod(scalar, m.SetUint64(moduli[l])).Uint64()
		row := make([]uint64, N)
		for x := range row {
			row[x] = v
		}
		out[l] = row
	}
	return out
}

func ScalarBroadcastRef(scalar uint64, moduli []uint64, limbs, N int) [][]uint64 {
	out := make([][]uint64, limbs)
	for l := 0; l < limbs && l < len(moduli); l++ {
		v := scalar % moduli[l]
		row := make([]uint64, N)
		for x := range row {
			row[x] = v
		}
		out[l] = row
	}
	return out
}
