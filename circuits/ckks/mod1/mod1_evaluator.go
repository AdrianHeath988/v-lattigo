// Package mod1 implements a homomorphic mod1 circuit for the CKKS scheme.
package mod1

import (
	"fmt"
	"math/big"
	"math/cmplx"
	"os"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// Evaluator is an evaluator providing an API for homomorphic evaluations of scaled x mod 1.
// All fields of this struct are public, enabling custom instantiations.
type Evaluator struct {
	*ckks.Evaluator
	PolynomialEvaluator *polynomial.Evaluator
	Parameters          Parameters

	// vFHE trace capture (set by the bootstrapping evaluator only during EvalMod):
	// when non-nil, EvaluateAndScaleNew emits each top-level step's operand/result.
	TraceSink ring.TraceSink
	TraceStep int
}

// mod1PrefixSink wraps a TraceSink with a region prefix, so the double-angle
// rescale (RescaleTraced emits operand/intt/advice/divround/ntt/result) lands
// under "rs_*" keys the runtime drains via flushRescale — the same proven
// streamed-rescale path the homomorphic DFT (CoeffsToSlots) per-step rescale uses.
type mod1PrefixSink struct {
	inner  ring.TraceSink
	prefix string
}

func (p mod1PrefixSink) Poly(region string, idx int, vals []uint64) {
	p.inner.Poly(p.prefix+region, idx, vals)
}
func (p mod1PrefixSink) Stage(region string, idx, stage int, vals []uint64) {
	p.inner.Stage(p.prefix+region, idx, stage, vals)
}

// emitEvalModStep emits one EvalMod step (operand + result ciphertexts) under
// region prefix "evalmod_". `kind` tags the op (offset/poly/dbl-angle/arcsine).
func emitEvalModStep(sink ring.TraceSink, step int, kind uint64, operand, result *rlwe.Ciphertext) {
	if sink == nil || operand == nil {
		return
	}
	level := result.Level()
	ctSize := len(result.Value)
	opLevel := operand.Level()
	sink.Poly("evalmod_meta", step, []uint64{uint64(level), uint64(ctSize), kind, uint64(opLevel)})
	for k := 0; k < ctSize; k++ {
		for i := 0; i <= opLevel; i++ {
			sink.Poly("evalmod_operand", (step*8+k)*64+i, append([]uint64{}, operand.Value[k].Coeffs[i]...))
		}
		for i := 0; i <= level; i++ {
			sink.Poly("evalmod_result", (step*8+k)*64+i, append([]uint64{}, result.Value[k].Coeffs[i]...))
		}
	}
}

// NewEvaluator instantiates a new [Evaluator] evaluator from [ckks.Evaluator].
// This method is allocation free.
func NewEvaluator(eval *ckks.Evaluator, evalPoly *polynomial.Evaluator, Mod1Parameters Parameters) *Evaluator {
	return &Evaluator{Evaluator: eval, PolynomialEvaluator: evalPoly, Parameters: Mod1Parameters}
}

// EvaluateAndScaleNew calls [EvaluateNew] and scales the output values by `scaling` (without consuming additional depth).
// If `scaling` set to 1, then this is equivalent to simply calling [EvaluateNew].
func (eval Evaluator) EvaluateAndScaleNew(ct *rlwe.Ciphertext, scaling complex128) (res *rlwe.Ciphertext, err error) {

	evm := eval.Parameters

	if ct.Level() < evm.LevelQ {
		return nil, fmt.Errorf("cannot Evaluate: ct.Level() < Mod1Parameters.LevelQ")
	}

	if ct.Level() > evm.LevelQ {
		eval.DropLevel(ct, ct.Level()-evm.LevelQ)
	}

	res = ct.CopyNew()

	// Normalize the modular reduction to mod by 1 (division by Q)
	res.Scale = evm.ScalingFactor()

	// vFHE: per-step EvalMod trace capture. snap() copies the current operand;
	// emit() records (operand, res) for the just-completed step. kind tags the op:
	// 0=offset-add, 1=Chebyshev poly-eval, 2=double-angle iteration, 3=arcsine.
	tstep := 0
	snap := func() *rlwe.Ciphertext {
		if eval.TraceSink != nil {
			return res.CopyNew()
		}
		return nil
	}
	emit := func(kind uint64, operand *rlwe.Ciphertext) {
		if eval.TraceSink != nil {
			// eval.TraceStep is a per-call base offset (set by the bootstrapping
			// EvalMod) so the ctReal/ctImag EvalMod calls don't collide.
			emitEvalModStep(eval.TraceSink, eval.TraceStep+tstep, kind, operand, res)
			tstep++
		}
	}

	// Compute the scales that the ciphertext should have before the double angle
	// formula such that after it it has the scale it had before the polynomial
	// evaluation

	Qi := eval.GetParameters().Q()

	targetScale := res.Scale
	for i := 0; i < evm.DoubleAngle; i++ {
		targetScale = targetScale.Mul(rlwe.NewScale(Qi[ct.Level()-evm.Mod1Poly.Depth()-evm.DoubleAngle+i+1]))
		targetScale.Value.Sqrt(&targetScale.Value)
	}

	// Division by 1/2^r and change of variable for the Chebyshev evaluation
	if evm.Mod1Type == CosDiscrete || evm.Mod1Type == CosContinuous {
		offset := new(big.Float).Sub(&evm.Mod1Poly.B, &evm.Mod1Poly.A)
		offset.Mul(offset, new(big.Float).SetFloat64(evm.IntervalShrinkFactor()))
		offset.Quo(new(big.Float).SetFloat64(-0.5), offset)
		o := snap()
		if err = eval.Add(res, offset, res); err != nil {
			return nil, fmt.Errorf("cannot Evaluate: %w", err)
		}
		var offRef [][]uint64
		if eval.TraceSink != nil && o != nil { // bind offset to the public constant (hole 1)
			offRef = scalarRefPlain(eval.Evaluator, o.Level(), o.Scale, offset)
		}
		emitLinOp(eval.TraceSink, loPADD, o, nil, res, offRef) // offset change-of-variable
		emit(0, o)
	}

	// Double angle
	sqrt2pi := complex(evm.Sqrt2Pi, 0)

	var mod1Poly bignum.Polynomial
	if evm.Mod1InvPoly == nil {

		scaling := cmplx.Pow(scaling, complex(1/evm.IntervalShrinkFactor(), 0))

		mul := bignum.NewComplexMultiplier().Mul

		mod1Poly = evm.Mod1Poly.Clone()

		scalingPowBig := bignum.NewComplex().SetComplex128(scaling)

		for i := range mod1Poly.Coeffs {
			if mod1Poly.Coeffs[i] != nil {
				mul(mod1Poly.Coeffs[i], scalingPowBig, mod1Poly.Coeffs[i])
			}
		}

		sqrt2pi *= scaling

	} else {
		mod1Poly = evm.Mod1Poly
	}

	// Chebyshev evaluation
	pIn := snap()
	// vFHE: capture the Chebyshev poly-eval's internal key-switches + rescales via the
	// tracing decorator (no-op when not tracing); production path is unchanged.
	if err = eval.withTracedPolyEval(func() (e error) {
		res, e = eval.PolynomialEvaluator.Evaluate(res, mod1Poly, rlwe.NewScale(targetScale))
		return
	}); err != nil {
		return nil, fmt.Errorf("cannot Evaluate: %w", err)
	}
	emit(1, pIn)

	for i := 0; i < evm.DoubleAngle; i++ {
		dIn := snap()

		// vFHE: capture the double-angle square+relin as a streamed multi-P
		// key-switch proof, WITHOUT perturbing the production MulRelin below. We
		// RECOMPUTE the identical square (dIn·dIn, the same product MulRelin forms)
		// and relinearize it through the instrumented multi-P path into a throwaway
		// ciphertext; the runtime drains the "rl_*" regions on the trailing "rl_meta".
		// Recompute (not split) keeps the numerically-sensitive bootstrap path exact.
		// A SKIPPED capture here leaves the double-angle relin with no board and its
		// result bound to nothing -- the same silent gap the ModUp key-switch had.
		if eval.TraceSink != nil &&
			!(len(eval.GetParameters().P()) >= 2 && dIn != nil && dIn.Level() >= 1) {
			fmt.Fprintf(os.Stderr, "[vfhe] WARNING double-angle relin NOT captured "+
				"(nP=%d level=%d): its result will be bound to nothing\n",
				len(eval.GetParameters().P()), func() int {
					if dIn == nil {
						return -1
					}
					return dIn.Level()
				}())
		}
		if eval.TraceSink != nil && len(eval.GetParameters().P()) >= 2 && dIn != nil && dIn.Level() >= 1 {
			if prod, e := eval.MulNew(dIn, dIn); e == nil && prod.Degree() == 2 {
				// square tensor (MUL: prod = dIn⊗dIn) + its relin (multi-P key-switch).
				emitLinOp(eval.TraceSink, loMUL, dIn, dIn, prod, nil)
				tmp := ckks.NewCiphertext(*eval.GetParameters(), 1, dIn.Level())
				if L, nP, d, e2 := eval.RelinearizeMultiPTraced(prod, tmp,
					mod1PrefixSink{inner: eval.TraceSink, prefix: "rl_"}); e2 == nil {
					eval.TraceSink.Poly("rl_meta", 0, []uint64{
						uint64(eval.GetParameters().N()), uint64(L), uint64(nP), uint64(d)})
				}
			}
		}

		sqrt2pi *= sqrt2pi

		if err = eval.MulRelin(res, res, res); err != nil {
			return nil, fmt.Errorf("cannot Evaluate: %w", err)
		}
		stA := snap() // after square+relin (before doubling)

		if err = eval.Add(res, res, res); err != nil {
			return nil, fmt.Errorf("cannot Evaluate: %w", err)
		}
		stB := snap() // after doubling (= 2·stA)
		emitLinOp(eval.TraceSink, loADD, stA, stA, stB, nil)

		if err = eval.Add(res, -sqrt2pi, res); err != nil {
			return nil, fmt.Errorf("cannot Evaluate: %w", err)
		}
		var saRef [][]uint64
		if eval.TraceSink != nil && stB != nil { // bind −sqrt2pi to the public constant (hole 1)
			saRef = scalarRefPlain(eval.Evaluator, stB.Level(), stB.Scale, complex128(-sqrt2pi))
		}
		emitLinOp(eval.TraceSink, loPADD, stB, nil, res, saRef) // const-add (= stB − sqrt2pi)

		// vFHE: capture the double-angle rescale as a streamed, SNARK-provable
		// RESCALE op (reuses the proven RescaleTraced + flushRescale path). The
		// runtime drains the "rs_*" regions on the trailing "rs_meta"; the 2-element
		// meta [N, inLimbs] selects the plain rescale dump (no BSGS-step edge fields).
		// Falls back to the untraced Rescale outside capture / multi-prime rescaling.
		if eval.TraceSink != nil &&
			!(eval.GetParameters().LevelsConsumedPerRescaling() == 1 && res.Level() >= 1) {
			fmt.Fprintf(os.Stderr, "[vfhe] WARNING double-angle rescale NOT captured "+
				"(levelsPerRescale=%d level=%d): its result will be bound to nothing\n",
				eval.GetParameters().LevelsConsumedPerRescaling(), res.Level())
		}
		if eval.TraceSink != nil && eval.GetParameters().LevelsConsumedPerRescaling() == 1 && res.Level() >= 1 {
			inLimbs := res.Level() + 1
			if err = eval.RescaleTraced(res, res, mod1PrefixSink{inner: eval.TraceSink, prefix: "rs_"}); err != nil {
				return nil, fmt.Errorf("cannot Evaluate (rescale traced): %w", err)
			}
			eval.TraceSink.Poly("rs_meta", 0, []uint64{uint64(eval.GetParameters().N()),
				uint64(inLimbs), 0, 0, ring.RsOriginDoubleAngle})
		} else if err = eval.Rescale(res, res); err != nil {
			return nil, fmt.Errorf("cannot Evaluate: %w", err)
		}
		emit(2, dIn)
	}

	// ArcSine
	if evm.Mod1InvPoly != nil {

		mul := bignum.NewComplexMultiplier().Mul

		mod1InvPoly := evm.Mod1InvPoly.Clone()

		scalingBig := bignum.NewComplex().SetComplex128(scaling)

		for i := range mod1InvPoly.Coeffs {
			if mod1InvPoly.Coeffs[i] != nil {
				mul(mod1InvPoly.Coeffs[i], scalingBig, mod1InvPoly.Coeffs[i])
			}
		}

		aIn := snap()
		if err = eval.withTracedPolyEval(func() (e error) {
			res, e = eval.PolynomialEvaluator.Evaluate(res, mod1InvPoly, res.Scale)
			return
		}); err != nil {
			return nil, fmt.Errorf("cannot Evaluate: %w", err)
		}
		emit(3, aIn)
	}

	// Multiplies back by q
	res.Scale = ct.Scale
	return res, nil
}

// EvaluateNew applies an homomorphic mod Q on a vector scaled by Delta, scaled down to mod 1:
//
//  1. Delta * (Q/Delta * I(X) + m(X)) (Delta = scaling factor, I(X) integer poly, m(X) message)
//  2. Delta * (I(X) + Delta/Q * m(X)) (divide by Q/Delta)
//  3. Delta * (Delta/Q * m(X)) (x mod 1)
//  4. Delta * (m(X)) (multiply back by Q/Delta)
//
// Since Q is not a power of two, but Delta is, then does an approximate division by the closest
// power of two to Q instead. Hence, it assumes that the input plaintext is already scaled by
// the correcting factor Q/2^{round(log(Q))}.
//
// !! Assumes that the input is normalized by 1/K for K the range of the approximation.
//
// Scaling back error correction by 2^{round(log(Q))}/Q afterward is included in the polynomial
func (eval Evaluator) EvaluateNew(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	return eval.EvaluateAndScaleNew(ct, 1)
}
