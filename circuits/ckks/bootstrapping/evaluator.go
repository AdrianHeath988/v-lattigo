package bootstrapping

import (
	"fmt"
	"math"
	"math/big"
	"math/bits"
	"os"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/mod1"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/circuits/common/vfhetrace"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// Evaluator is a struct to store a memory buffer with the plaintext matrices,
// the polynomial approximation, and the keys for the bootstrapping.
// It is used to evaluate the bootstrapping circuit on single ciphertexts.
type Evaluator struct {
	Parameters
	*ckks.Evaluator
	DFTEvaluator  *dft.Evaluator
	Mod1Evaluator *mod1.Evaluator
	*EvaluationKeys

	ckks.DomainSwitcher

	// [1, x, x^2, x^4, ..., x^N1/2] / (X^N1 +1)
	xPow2N1 []ring.Poly
	// [1, x, x^2, x^4, ..., x^N2/2] / (X^N2 +1)
	xPow2N2 []ring.Poly
	// [1, x^-1, x^-2, x^-4, ..., x^-N1/2] / (X^N1 +1)
	xPow2InvN1 []ring.Poly
	// [1, x^-1, x^-2, x^-4, ..., x^-N2/2] / (X^N2 +1)
	xPow2InvN2 []ring.Poly

	Mod1Parameters mod1.Parameters
	S2CDFTMatrix   dft.Matrix
	C2SDFTMatrix   dft.Matrix

	SkDebug *rlwe.SecretKey

	// TraceSink, when set (vFHE proof capture), receives the per-stage proof
	// trace of the bootstrap (currently the ModUp centered CRT lift). nil in
	// normal operation -> zero overhead, behaviour unchanged.
	TraceSink ring.TraceSink

	pool *rlwe.BufferPool
}

// NewEvaluator creates a new [Evaluator].
func NewEvaluator(btpParams Parameters, evk *EvaluationKeys) (eval *Evaluator, err error) {

	eval = &Evaluator{}

	paramsN1 := btpParams.ResidualParameters
	paramsN2 := btpParams.BootstrappingParameters

	switch paramsN1.RingType() {
	case ring.Standard:
		if paramsN1.N() != paramsN2.N() && (evk.EvkN1ToN2 == nil || evk.EvkN2ToN1 == nil) {
			return nil, fmt.Errorf("cannot NewBootstrapper: evk.(BootstrappingKeys) is missing EvkN1ToN2 and EvkN2ToN1")
		}
	case ring.ConjugateInvariant:
		if evk.EvkCmplxToReal == nil || evk.EvkRealToCmplx == nil {
			return nil, fmt.Errorf("cannot NewBootstrapper: evk.(BootstrappingKeys) is missing EvkN1ToN2 and EvkN2ToN1")
		}

		var err error
		if eval.DomainSwitcher, err = ckks.NewDomainSwitcher(paramsN2, evk.EvkCmplxToReal, evk.EvkRealToCmplx); err != nil {
			return nil, fmt.Errorf("cannot NewBootstrapper: ckks.NewDomainSwitcher: %w", err)
		}

		// The switch to standard to conjugate invariant multiplies the scale by 2
		btpParams.SlotsToCoeffsParameters.Scaling = new(big.Float).SetFloat64(0.5)
	}

	eval.Parameters = btpParams

	if paramsN1.N() != paramsN2.N() {
		eval.xPow2N1 = rlwe.GenXPow2NTT(paramsN1.RingQ().AtLevel(0), paramsN2.LogN(), false)
		eval.xPow2InvN1 = rlwe.GenXPow2NTT(paramsN1.RingQ(), paramsN1.LogN(), true)
	}
	eval.xPow2N2 = rlwe.GenXPow2NTT(paramsN2.RingQ().AtLevel(0), paramsN2.LogN(), false)
	eval.xPow2InvN2 = rlwe.GenXPow2NTT(paramsN2.RingQ(), paramsN2.LogN(), true)

	if btpParams.Mod1ParametersLiteral.Mod1Type == mod1.SinContinuous && btpParams.Mod1ParametersLiteral.DoubleAngle != 0 {
		return nil, fmt.Errorf("cannot use double angle formula for Mod1Type = Sin -> must use Mod1Type = Cos")
	}

	if btpParams.Mod1ParametersLiteral.Mod1Type == mod1.CosDiscrete && btpParams.Mod1ParametersLiteral.Mod1Degree < 2*(btpParams.Mod1ParametersLiteral.K-1) {
		return nil, fmt.Errorf("Mod1Type 'mod1.CosDiscrete' uses a minimum degree of 2*(K-1) but EvalMod degree is smaller")
	}

	switch btpParams.CircuitOrder {
	case ModUpThenEncode:
		if btpParams.CoeffsToSlotsParameters.LevelQ-btpParams.CoeffsToSlotsParameters.Depth(true) != btpParams.Mod1ParametersLiteral.LevelQ {
			return nil, fmt.Errorf("starting level and depth of CoeffsToSlotsParameters inconsistent starting level of Mod1ParametersLiteral")
		}

		if btpParams.Mod1ParametersLiteral.LevelQ-btpParams.Mod1ParametersLiteral.Depth() != btpParams.SlotsToCoeffsParameters.LevelQ {
			return nil, fmt.Errorf("starting level and depth of Mod1ParametersLiteral inconsistent starting level of CoeffsToSlotsParameters")
		}
	case DecodeThenModUp:
		if btpParams.BootstrappingParameters.MaxLevel()-btpParams.CoeffsToSlotsParameters.Depth(true) != btpParams.Mod1ParametersLiteral.LevelQ {
			return nil, fmt.Errorf("starting level and depth of Mod1ParametersLiteral inconsistent starting level of CoeffsToSlotsParameters")
		}
	case Custom:
	default:
		return nil, fmt.Errorf("invalid CircuitOrder value")
	}

	if err = eval.initialize(btpParams); err != nil {
		return
	}

	if err = eval.checkKeys(evk); err != nil {
		return
	}

	params := btpParams.BootstrappingParameters

	eval.EvaluationKeys = evk

	eval.Evaluator = ckks.NewEvaluator(params, evk)

	eval.DFTEvaluator = dft.NewEvaluator(params, eval.Evaluator)

	eval.Mod1Evaluator = mod1.NewEvaluator(eval.Evaluator, polynomial.NewEvaluator(params, eval.Evaluator), eval.Mod1Parameters)

	eval.pool = rlwe.NewPool(eval.BootstrappingParameters.RingQP())
	return
}

// CheckKeys checks if all the necessary keys are present in the instantiated [Evaluator]
func (eval Evaluator) checkKeys(evk *EvaluationKeys) (err error) {

	if _, err = evk.GetRelinearizationKey(); err != nil {
		return
	}

	for _, galEl := range eval.GaloisElements(eval.BootstrappingParameters) {
		if _, err = evk.GetGaloisKey(galEl); err != nil {
			return
		}
	}

	if evk.EvkDenseToSparse == nil && eval.EphemeralSecretWeight != 0 {
		return fmt.Errorf("rlwe.EvaluationKey key dense to sparse is nil")
	}

	if evk.EvkSparseToDense == nil && eval.EphemeralSecretWeight != 0 {
		return fmt.Errorf("rlwe.EvaluationKey key sparse to dense is nil")
	}

	return
}

func (eval *Evaluator) initialize(btpParams Parameters) (err error) {
	eval.Parameters = btpParams
	params := btpParams.BootstrappingParameters

	if eval.Mod1Parameters, err = mod1.NewParametersFromLiteral(params, btpParams.Mod1ParametersLiteral); err != nil {
		return
	}

	// [-K, K]
	K := eval.Mod1Parameters.K

	// Correcting factor for approximate division by Q
	// The second correcting factor for approximate multiplication by Q is included in the coefficients of the EvalMod polynomials
	qDiff := eval.Mod1Parameters.QDiff

	// If the scale used during the EvalMod step is smaller than Q0, then we cannot increase the scale during
	// the EvalMod step to get a free division by MessageRatio, and we need to do this division (totally or partly)
	// during the CoeffstoSlots step
	qDiv := eval.Mod1Parameters.ScalingFactor().Float64() / math.Exp2(math.Round(math.Log2(float64(params.Q()[0]))))

	// Sets qDiv to 1 if there is enough room for the division to happen using scale manipulation.
	if qDiv > 1 {
		qDiv = 1
	}

	encoder := ckks.NewEncoder(params)

	// CoeffsToSlots vectors
	// Change of variable for the evaluation of the Chebyshev polynomial + cancelling factor for the DFT and SubSum + eventual scaling factor for the double angle formula

	scale := eval.BootstrappingParameters.DefaultScale().Float64()
	offset := eval.Mod1Parameters.ScalingFactor().Float64() / eval.Mod1Parameters.MessageRatio()

	C2SScaling := new(big.Float).SetFloat64(qDiv / (K * qDiff))
	StCScaling := new(big.Float).SetFloat64(scale / offset)

	if btpParams.CoeffsToSlotsParameters.Scaling == nil {
		eval.CoeffsToSlotsParameters.Scaling = C2SScaling
	} else {
		eval.CoeffsToSlotsParameters.Scaling = new(big.Float).Mul(btpParams.CoeffsToSlotsParameters.Scaling, C2SScaling)
	}

	if btpParams.SlotsToCoeffsParameters.Scaling == nil {
		eval.SlotsToCoeffsParameters.Scaling = StCScaling
	} else {
		eval.SlotsToCoeffsParameters.Scaling = new(big.Float).Mul(btpParams.SlotsToCoeffsParameters.Scaling, StCScaling)
	}

	if eval.C2SDFTMatrix, err = dft.NewMatrixFromLiteral(params, eval.CoeffsToSlotsParameters, encoder); err != nil {
		return
	}

	if eval.S2CDFTMatrix, err = dft.NewMatrixFromLiteral(params, eval.SlotsToCoeffsParameters, encoder); err != nil {
		return
	}

	encoder = nil // For the GC

	return
}

// Bootstrap bootstraps a single ciphertext and returns the bootstrapped ciphertext.
func (eval Evaluator) Bootstrap(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	cts := []rlwe.Ciphertext{*ct}
	cts, err := eval.BootstrapMany(cts)
	if err != nil {
		return nil, err
	}
	return &cts[0], nil
}

// BootstrapMany bootstraps a list of ciphertexts and returns the list of bootstrapped ciphertexts.
func (eval Evaluator) BootstrapMany(cts []rlwe.Ciphertext) ([]rlwe.Ciphertext, error) {

	var err error

	switch eval.ResidualParameters.RingType() {
	case ring.ConjugateInvariant:

		for i := 0; i < len(cts); i = i + 2 {

			even, odd := i, i+1

			ct0 := &cts[even]

			var ct1 *rlwe.Ciphertext
			if odd < len(cts) {
				ct1 = &cts[odd]
			}

			if ct0, ct1, err = eval.EvaluateConjugateInvariant(ct0, ct1); err != nil {
				return nil, fmt.Errorf("cannot BootstrapMany: %w", err)
			}

			cts[even] = *ct0

			if ct1 != nil {
				cts[odd] = *ct1
			}
		}

	default:

		ctsPacked, ctxt1, ctxt2, err := eval.PackAndSwitchN1ToN2(cts)
		if err != nil {
			return nil, fmt.Errorf("cannot BootstrapMany: %w", err)
		}

		cts = ctsPacked
		for i := range cts {
			var ct *rlwe.Ciphertext
			if ct, err = eval.Evaluate(&cts[i]); err != nil {
				return nil, fmt.Errorf("cannot BootstrapMany: %w", err)
			}
			cts[i] = *ct
		}

		if cts, err = eval.UnpackAndSwitchN2ToN1(cts, ctxt1, ctxt2); err != nil {
			return nil, fmt.Errorf("cannot BootstrapMany: %w", err)
		}
	}

	for i := range cts {
		cts[i].Scale = eval.ResidualParameters.DefaultScale()
	}

	return cts, err
}

// Depth returns the multiplicative depth (number of levels consumed) of the bootstrapping circuit.
func (eval Evaluator) Depth() int {
	return eval.BootstrappingParameters.MaxLevel() - eval.ResidualParameters.MaxLevel()
}

// OutputLevel returns the output level after the evaluation of the bootstrapping circuit.
func (eval Evaluator) OutputLevel() int {
	return eval.ResidualParameters.MaxLevel()
}

// MinimumInputLevel returns the minimum level at which a ciphertext must be to be bootstrapped.
func (eval Evaluator) MinimumInputLevel() int {
	return eval.BootstrappingParameters.LevelsConsumedPerRescaling()
}

// Evaluate re-encrypts a ciphertext to a ciphertext at MaxLevel - k where k is the depth of the bootstrapping circuit.
// If the input ciphertext level is zero, the input scale must be an exact power of two smaller than Q[0]/MessageRatio
// (it can't be equal since Q[0] is not a power of two).
// The message ratio is an optional field in the bootstrapping parameters, by default it set to 2^{LogMessageRatio = 8}.
// See the bootstrapping parameters for more information about the message ratio or other parameters related to the bootstrapping.
// If the input ciphertext is at level one or more, the input scale does not need to be an exact power of two as one level
// can be used to do a scale matching.
//
// The circuit consists in 5 steps.
//  1. ScaleDown: scales the ciphertext to q/|m| and bringing it down to q
//  2. ModUp: brings the modulus from q to Q
//  3. CoeffsToSlots: homomorphic encoding
//  4. EvalMod: homomorphic modular reduction
//  5. SlotsToCoeffs: homomorphic decoding
func (eval Evaluator) Evaluate(ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext, err error) {

	if eval.IterationsParameters == nil && eval.ResidualParameters.PrecisionMode() != ckks.PREC128 {
		ctOut, _, err = eval.bootstrap(ctIn)
		return

	} else {

		var errScale *rlwe.Scale
		// [M^{d}/q1 + e^{d-logprec}]
		if ctOut, errScale, err = eval.bootstrap(ctIn.CopyNew()); err != nil {
			return nil, err
		}

		// Stores by how much a ciphertext must be scaled to get back
		// to the input scale
		// Error correcting factor of the approximate division by q1
		// diffScale = ctIn.Scale / (ctOut.Scale * errScale)
		diffScale := ctIn.Scale.Div(ctOut.Scale)
		diffScale = diffScale.Div(*errScale)

		// [M^{d} + e^{d-logprec}]
		// vFHE: the post-bootstrap scale correction. It runs OUTSIDE bootstrap(),
		// after every stage proof is done, so untraced it was a multiply by an
		// arbitrary constant applied to the finished result -- the last thing the
		// bootstrap does and the easiest place to change the answer.
		var scBefore *rlwe.Ciphertext
		if eval.TraceSink != nil {
			scBefore = ctOut.CopyNew()
		}
		if err = eval.Evaluator.Mul(ctOut, diffScale.BigInt(), ctOut); err != nil {
			return nil, err
		}
		if scBefore != nil {
			vfhetrace.EmitLinOp(eval.TraceSink, vfhetrace.OpPMUL, scBefore, nil, ctOut,
				vfhetrace.MulRefPlain(eval.Evaluator, ctOut.Level(),
					rlwe.NewScale(1), diffScale.BigInt()))
		}
		ctOut.Scale = ctIn.Scale

		if eval.IterationsParameters != nil {

			QiReserved := eval.BootstrappingParameters.Q()[eval.ResidualParameters.MaxLevel()+1]

			var totLogPrec float64

			for i := 0; i < len(eval.IterationsParameters.BootstrappingPrecision); i++ {

				logPrec := eval.IterationsParameters.BootstrappingPrecision[i]

				totLogPrec += logPrec

				// prec = round(2^{logprec})
				log2 := bignum.Log(new(big.Float).SetPrec(256).SetUint64(2))
				log2TimesLogPrec := log2.Mul(log2, new(big.Float).SetFloat64(totLogPrec))
				prec := new(big.Int)
				log2TimesLogPrec.Add(bignum.Exp(log2TimesLogPrec), new(big.Float).SetFloat64(0.5)).Int(prec)

				// Corrects the last iteration 2^{logprec} such that diffScale / prec * QReserved is as close to an integer as possible.
				// This is necessary to not lose bits of precision during the last iteration is a reserved prime is used.
				// If this correct is not done, what can happen is that there is a loss of up to 2^{logprec/2} bits from the last iteration.
				if eval.IterationsParameters.ReservedPrimeBitSize != 0 && i == len(eval.IterationsParameters.BootstrappingPrecision)-1 {

					// 1) Computes the scale = diffScale / prec * QReserved
					scale := new(big.Float).Quo(&diffScale.Value, new(big.Float).SetInt(prec))
					scale.Mul(scale, new(big.Float).SetUint64(QiReserved))

					// 2) Finds the closest integer to scale with scale = round(scale)
					scale.Add(scale, new(big.Float).SetFloat64(0.5))
					tmp := new(big.Int)
					scale.Int(tmp)
					scale.SetInt(tmp)

					// 3) Computes the corrected precision = diffScale * QReserved / round(scale)
					preccorrected := new(big.Float).Quo(&diffScale.Value, scale)
					preccorrected.Mul(preccorrected, new(big.Float).SetUint64(QiReserved))
					preccorrected.Add(preccorrected, new(big.Float).SetFloat64(0.5))

					// 4) Updates with the corrected precision
					preccorrected.Int(prec)
				}

				// round(q1/logprec)
				scale := new(big.Int).Set(diffScale.BigInt())
				bignum.DivRound(scale, prec, scale)

				// Checks that round(q1/logprec) >= 2^{logprec}
				requiresReservedPrime := scale.Cmp(new(big.Int).SetUint64(1)) < 0

				if requiresReservedPrime && eval.IterationsParameters.ReservedPrimeBitSize == 0 {
					return ctOut, fmt.Errorf("warning: early stopping at iteration k=%d: reason: round(q1/2^{logprec}) < 1 and no reserverd prime was provided", i+1)
				}

				// [M^{d} + e^{d-logprec}] - [M^{d}] -> [e^{d-logprec}]
				tmp, err := eval.Evaluator.SubNew(ctOut, ctIn)

				if err != nil {
					return nil, err
				}

				// prec * [e^{d-logprec}] -> [e^{d}]
				if err = eval.Evaluator.Mul(tmp, prec, tmp); err != nil {
					return nil, err
				}

				tmp.Scale = ctOut.Scale

				// [e^{d}] -> [e^{d}/q1] -> [e^{d}/q1 + e'^{d-logprec}]
				if tmp, errScale, err = eval.bootstrap(tmp); err != nil {
					return nil, err
				}

				tmp.Scale = tmp.Scale.Mul(*errScale)

				// [[e^{d}/q1 + e'^{d-logprec}] * q1/logprec -> [e^{d-logprec} + e'^{d-2logprec}*q1]
				if eval.IterationsParameters.ReservedPrimeBitSize == 0 {
					if err = eval.Evaluator.Mul(tmp, scale, tmp); err != nil {
						return nil, err
					}
				} else {

					// Else we compute the floating point ratio
					scale := new(big.Float).SetInt(diffScale.BigInt())
					scale.Quo(scale, new(big.Float).SetInt(prec))

					if new(big.Float).Mul(scale, new(big.Float).SetUint64(QiReserved)).Cmp(new(big.Float).SetUint64(1)) == -1 {
						return ctOut, fmt.Errorf("warning: early stopping at iteration k=%d: reason: maximum precision achieved", i+1)
					}

					// Do a scaled multiplication by the last prime
					if err = eval.Evaluator.Mul(tmp, scale, tmp); err != nil {
						return nil, err
					}

					// And rescale
					if err = eval.Evaluator.Rescale(tmp, tmp); err != nil {
						return nil, err
					}
				}

				// This is a given
				tmp.Scale = ctOut.Scale

				// [M^{d} + e^{d-logprec}] - [e^{d-logprec} + e'^{d-2logprec}*q1] -> [M^{d} + e'^{d-2logprec}*q1]
				if err = eval.Evaluator.Sub(ctOut, tmp, ctOut); err != nil {
					return nil, err
				}
			}
		}

		for ctOut.Level() > eval.ResidualParameters.MaxLevel() {
			eval.Evaluator.DropLevel(ctOut, 1)
		}
	}

	return
}

// EvaluateConjugateInvariant takes two ciphertext in the Conjugate Invariant ring, repacks them in a single ciphertext in the standard ring
// using the real and imaginary part, bootstrap both ciphertext, and then extract back the real and imaginary part before repacking them
// individually in two new ciphertexts in the Conjugate Invariant ring.
func (eval Evaluator) EvaluateConjugateInvariant(ctLeftN1Q0, ctRightN1Q0 *rlwe.Ciphertext) (ctLeftN1QL, ctRightN1QL *rlwe.Ciphertext, err error) {

	if ctLeftN1Q0 == nil {
		return nil, nil, fmt.Errorf("ctLeftN1Q0 cannot be nil")
	}

	// Switches ring from ring.ConjugateInvariant to ring.Standard
	ctLeftN2Q0 := eval.RealToComplexNew(ctLeftN1Q0)

	// Repacks ctRightN1Q0 into the imaginary part of ctLeftN1Q0
	// which is zero since it comes from the Conjugate Invariant ring)
	if ctRightN1Q0 != nil {
		ctRightN2Q0 := eval.RealToComplexNew(ctRightN1Q0)

		if err = eval.Evaluator.Mul(ctRightN2Q0, 1i, ctRightN2Q0); err != nil {
			return nil, nil, fmt.Errorf("cannot BootstrapMany: %w", err)
		}

		if err = eval.Evaluator.Add(ctLeftN2Q0, ctRightN2Q0, ctLeftN2Q0); err != nil {
			return nil, nil, fmt.Errorf("cannot BootstrapMany: %w", err)
		}
	}

	// Bootstraps in the ring.Standard
	var ctLeftAndRightN2QL *rlwe.Ciphertext
	if ctLeftAndRightN2QL, err = eval.Evaluate(ctLeftN2Q0); err != nil {
		return nil, nil, fmt.Errorf("cannot BootstrapMany: %w", err)
	}

	// The SlotsToCoeffs transformation scales the ciphertext by 0.5
	// This is done to compensate for the 2x factor introduced by ringStandardToConjugate(*).
	ctLeftAndRightN2QL.Scale = ctLeftAndRightN2QL.Scale.Mul(rlwe.NewScale(1 / 2.0))

	// Switches ring from ring.Standard to ring.ConjugateInvariant
	ctLeftN1QL = eval.ComplexToRealNew(ctLeftAndRightN2QL)

	// Extracts the imaginary part
	if ctRightN1Q0 != nil {
		if err = eval.Evaluator.Mul(ctLeftAndRightN2QL, -1i, ctLeftAndRightN2QL); err != nil {
			return nil, nil, fmt.Errorf("cannot BootstrapMany: %w", err)
		}
		ctRightN1QL = eval.ComplexToRealNew(ctLeftAndRightN2QL)
	}

	return
}

// checks if the current message ratio is greater or equal to the last prime times the target message ratio.
func checkMessageRatio(ct *rlwe.Ciphertext, msgRatio float64, r *ring.Ring) bool {
	level := ct.Level()
	currentMessageRatio := rlwe.NewScale(r.ModulusAtLevel[level])
	currentMessageRatio = currentMessageRatio.Div(ct.Scale)
	return currentMessageRatio.Cmp(rlwe.NewScale(r.SubRings[level].Modulus).Mul(rlwe.NewScale(msgRatio))) > -1
}

func (eval Evaluator) bootstrap(ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext, errScale *rlwe.Scale, err error) {

	// Step 1: scale to q/|m|
	if ctOut, errScale, err = eval.ScaleDown(ctIn); err != nil {
		return
	}

	// Step 2 : Extend the basis from q to Q
	if ctOut, err = eval.ModUp(ctOut); err != nil {
		return
	}

	// Step 3 : CoeffsToSlots (Homomorphic encoding)
	// ctReal = Ecd(real)
	// ctImag = Ecd(imag)
	// If n < N/2 then ctReal = Ecd(real||imag)
	var ctReal, ctImag *rlwe.Ciphertext
	if ctReal, ctImag, err = eval.CoeffsToSlots(ctOut); err != nil {
		return
	}

	// Step 4 : EvalMod (Homomorphic modular reduction)
	if ctReal, err = eval.EvalMod(ctReal); err != nil {
		return
	}

	// Step 4 : EvalMod (Homomorphic modular reduction)
	if ctImag != nil {
		if ctImag, err = eval.EvalMod(ctImag); err != nil {
			return
		}
	}

	// Step 5 : SlotsToCoeffs (Homomorphic decoding)
	if ctOut, err = eval.SlotsToCoeffs(ctReal, ctImag); err != nil {
		return
	}

	return
}

// ScaleDown brings the ciphertext level to zero and scaling factor to Q[0]/MessageRatio
// It multiplies the ciphertexts by round(currentMessageRatio / targetMessageRatio) where:
//   - currentMessageRatio = Q/ctIn.Scale
//   - targetMessageRatio = q/|m|
//
// and updates the scale of ctIn accordingly
// It then rescales the ciphertext down to q if necessary and also returns the rescaling error from this process
func (eval Evaluator) ScaleDown(ctIn *rlwe.Ciphertext) (*rlwe.Ciphertext, *rlwe.Scale, error) {

	params := &eval.BootstrappingParameters

	r := params.RingQ()

	// Removes unecessary primes
	//
	// vFHE: each Resize is a limb TRUNCATION -- the surviving limbs are untouched,
	// which is exactly what the modswitch board proves. Untraced it was still a
	// level change nothing accounted for, so the chain from the bootstrap input
	// into ScaleDown's scale-up had an unexplained discontinuity. Emitted per
	// dropped prime, at the SURVIVING limb count (the board binds only those).
	for ctIn.Level() != 0 && checkMessageRatio(ctIn, eval.Mod1Parameters.MessageRatio(), r) {
		var msOperand []ring.Poly
		if eval.TraceSink != nil {
			msOperand = make([]ring.Poly, len(ctIn.Value))
			for k := range ctIn.Value {
				msOperand[k] = *ctIn.Value[k].CopyNew()
			}
		}
		ctIn.Resize(ctIn.Degree(), ctIn.Level()-1)
		if eval.TraceSink != nil {
			emitModswitch(eval.TraceSink, r.N(), ctIn.Level()+1, msOperand, ctIn.Value)
		}
	}

	// Current Message Ratio
	currentMessageRatio := rlwe.NewScale(r.ModulusAtLevel[ctIn.Level()])
	currentMessageRatio = currentMessageRatio.Div(ctIn.Scale)

	// Desired Message Ratio
	targetMessageRatio := rlwe.NewScale(eval.Mod1Parameters.MessageRatio())

	// (Current Message Ratio) / (Desired Message Ratio)
	scaleUp := currentMessageRatio.Div(targetMessageRatio)

	if scaleUp.Cmp(rlwe.NewScale(0.5)) == -1 {
		return nil, nil, fmt.Errorf("initial Q/Scale = %f < 0.5*Q[0]/MessageRatio = %f", currentMessageRatio.Float64(), targetMessageRatio.Float64())
	}

	scaleUpBigint := scaleUp.BigInt()

	// vFHE: capture the ScaleDown public-scalar multiply (the new ScaleDown
	// primitive; the prime-drop above is a trivial truncation and the RescaleTo
	// below reuses the rescale gadget). Snapshot the operand before the in-place
	// multiply; emit operand/scalar/result after. Relation proved downstream:
	// result_i = (scalar mod q_i) * operand_i mod q_i.
	var sdOperand [][]uint64
	sdLevel := ctIn.Level()
	if eval.TraceSink != nil {
		Nn := r.N()
		sdOperand = make([][]uint64, len(ctIn.Value)*(sdLevel+1))
		for k := range ctIn.Value {
			for i := 0; i <= sdLevel; i++ {
				v := make([]uint64, Nn)
				copy(v, ctIn.Value[k].Coeffs[i])
				sdOperand[k*(sdLevel+1)+i] = v
			}
		}
	}

	if err := eval.Evaluator.Mul(ctIn, scaleUpBigint, ctIn); err != nil {
		return nil, nil, err
	}

	if eval.TraceSink != nil {
		Nn := r.N()
		Qc := r.ModuliChain()
		scalarRes := make([]uint64, Nn)
		for i := 0; i <= sdLevel; i++ {
			scalarRes[i] = new(big.Int).Mod(scaleUpBigint, new(big.Int).SetUint64(Qc[i])).Uint64()
		}
		eval.TraceSink.Poly("scaledown_scalar", 0, scalarRes)
		eval.TraceSink.Poly("scaledown_meta", 0, []uint64{uint64(sdLevel), uint64(len(ctIn.Value))})
		for k := range ctIn.Value {
			for i := 0; i <= sdLevel; i++ {
				eval.TraceSink.Poly("scaledown_operand", k*(sdLevel+1)+i, sdOperand[k*(sdLevel+1)+i])
				rv := make([]uint64, Nn)
				copy(rv, ctIn.Value[k].Coeffs[i])
				eval.TraceSink.Poly("scaledown_result", k*(sdLevel+1)+i, rv)
			}
		}
	}

	ctIn.Scale = ctIn.Scale.Mul(rlwe.NewScale(scaleUpBigint))

	// errScale = CtIn.Scale/(Q[0]/MessageRatio)
	targetScale := new(big.Float).SetPrec(256).SetInt(r.ModulusAtLevel[0])
	targetScale.Quo(targetScale, new(big.Float).SetFloat64(eval.Mod1Parameters.MessageRatio()))

	if ctIn.Level() != 0 {
		// vFHE: route through the traced form, which performs the same reduction as
		// n SINGLE-level rescales so each dropped prime gets a board. RescaleTo does
		// all n in one pass and never materialises the intermediates, leaving
		// nothing for the per-level board to bind. Falls back to RescaleTo verbatim
		// when there is no sink.
		if err := eval.RescaleToTraced(ctIn, rlwe.NewScale(targetScale), ctIn,
			eval.TraceSink); err != nil {
			return nil, nil, err
		}
	}

	// Rescaling error (if any)
	errScale := ctIn.Scale.Div(rlwe.NewScale(targetScale))

	return ctIn, &errScale, nil
}

// ModUp raise the modulus from q to Q, scales the message  and applies the Trace if the ciphertext is sparsely packed.
func (eval Evaluator) ModUp(ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext, err error) {

	// Switch to the sparse key
	if eval.EvkDenseToSparse != nil {
		// vFHE: the dense->sparse switch that opens ModUp, captured.
		if err := eval.applyEvkTraced(ctIn, eval.EvkDenseToSparse); err != nil {
			return nil, err
		}
	}

	params := eval.BootstrappingParameters

	ringQ := params.RingQ().AtLevel(ctIn.Level())
	ringP := params.RingP()

	// vFHE: the domain conversion at the head of ModUp. This is the last piece of
	// the ScaleDown -> ModUp seam, and not a formality: it PRODUCES the value the
	// centered lift then binds, so while it was untraced the lift committed to an
	// integer with no proven connection to the ciphertext that entered the
	// bootstrap. The board is gen_board_intt (standalone GS_INTT). In place, so the
	// eval-domain side is snapshotted first.
	var inttIn []ring.Poly
	if eval.TraceSink != nil {
		inttIn = make([]ring.Poly, len(ctIn.Value))
		for i := range ctIn.Value {
			inttIn[i] = *ctIn.Value[i].CopyNew()
		}
	}
	for i := range ctIn.Value {
		ringQ.INTT(ctIn.Value[i], ctIn.Value[i])
	}
	if eval.TraceSink != nil {
		emitIntt(eval.TraceSink, ringQ.N(), ctIn.Level()+1, inttIn, ctIn.Value)
	}

	// Extend the ciphertext from q to Q with zero values.
	ctIn.Resize(ctIn.Degree(), params.MaxLevel())

	levelQ := params.QCount() - 1
	levelP := params.PCount() - 1

	poolQP := eval.pool.AtLevel(levelQ, levelP)
	ringQ = ringQ.AtLevel(levelQ)

	Q := ringQ.ModuliChain()
	P := ringP.ModuliChain()
	q := Q[0]
	BRCQ := ringQ.BRedConstants()
	BRCP := ringP.BRedConstants()

	var coeff, tmp, pos, neg uint64

	N := ringQ.N()

	// ModUp q->Q for ctIn[0] centered around q. Shared with ring.ModUpCentered,
	// which performs the identical centered CRT lift and (when eval.TraceSink is
	// set) emits the vFHE proof trace (operand/sign/lift) for component 0.
	ringQ.ModUpCentered(0, levelQ, ctIn.Value[0].Coeffs[0], ctIn.Value[0], eval.TraceSink)

	if eval.EvkSparseToDense != nil {

		ks := eval.Evaluator.Evaluator

		buffDecompQP := poolQP.GetBuffDecompQP(eval.ResidualParameters.Parameters, eval.BootstrappingParameters.MaxLevelQ(), 0)
		defer eval.pool.RecycleBuffDecompQP(buffDecompQP)

		// ModUp q->QP for ctIn[1] centered around q
		for j := 0; j < N; j++ {

			coeff = ctIn.Value[1].Coeffs[0][j]
			pos, neg = 1, 0
			if coeff > (q >> 1) {
				coeff = q - coeff
				pos, neg = 0, 1
			}

			for i := 0; i < levelQ+1; i++ {
				tmp = ring.BRedAdd(coeff, Q[i], BRCQ[i])
				buffDecompQP[0].Q.Coeffs[i][j] = tmp*pos + (Q[i]-tmp)*neg

			}

			for i := 0; i < levelP+1; i++ {
				tmp = ring.BRedAdd(coeff, P[i], BRCP[i])
				buffDecompQP[0].P.Coeffs[i][j] = tmp*pos + (P[i]-tmp)*neg
			}
		}

		// vFHE: capture the V[1] centered CRT lift (q0 -> QP, the hoisted
		// key-switch's mod-up input) before the in-place NTT below. Component 1
		// uses STRICT '>' centering and lifts into every Q and P prime (incl. q0).
		if eval.TraceSink != nil {
			op1 := make([]uint64, N)
			copy(op1, ctIn.Value[1].Coeffs[0])
			sign1 := make([]uint64, N)
			for j := 0; j < N; j++ {
				if ctIn.Value[1].Coeffs[0][j] > (q >> 1) {
					sign1[j] = 1
				}
			}
			eval.TraceSink.Poly("modup_operand", 1, op1)
			eval.TraceSink.Poly("modup_sign", 1, sign1)
			for i := 0; i < levelQ+1; i++ {
				v := make([]uint64, N)
				copy(v, buffDecompQP[0].Q.Coeffs[i])
				eval.TraceSink.Poly("modup_lift", 1*(levelQ+1)+i, v)
			}
			for i := 0; i < levelP+1; i++ {
				v := make([]uint64, N)
				copy(v, buffDecompQP[0].P.Coeffs[i])
				eval.TraceSink.Poly("modup_liftP", i, v)
			}
			// vFHE NTT-tie: capture the per-stage forward NTT of each centered
			// lift (coeff -> eval), so the prover binds eval_i = NTT(lift_i) via
			// CT_NTT. Region indices ALIGN with the modup_lift/modup_liftP idxs:
			//   V[0] Q : modup_ntt  idx = i              (comp 0, lifts 1..levelQ)
			//   V[1] Q : modup_ntt  idx = (levelQ+1)+i   (comp 1, lifts 0..levelQ)
			//   V[1] P : modup_nttP idx = i              (comp 1, P lifts 0..levelP)
			// Captured BEFORE the in-place production NTTs below (lift still coeff).
			ringQ.NTTLiftTraced(ctIn.Value[0], 1, levelQ, "modup_ntt", 0, eval.TraceSink)
			ringQ.NTTLiftTraced(buffDecompQP[0].Q, 0, levelQ, "modup_ntt", levelQ+1, eval.TraceSink)
			ringP.NTTLiftTraced(buffDecompQP[0].P, 0, levelP, "modup_nttP", 0, eval.TraceSink)
		}

		for i := len(buffDecompQP) - 1; i >= 0; i-- {
			ringQ.NTT(buffDecompQP[0].Q, buffDecompQP[i].Q)
		}

		for i := len(buffDecompQP) - 1; i >= 0; i-- {
			ringP.NTT(buffDecompQP[0].P, buffDecompQP[i].P)
		}

		ringQ.NTT(ctIn.Value[0], ctIn.Value[0])

		// vFHE: this key-switch performs no gadget DECOMPOSITION. buffDecompQP[i]
		// is set to NTT(buffDecompQP[0]) above and then scaled by the scalar
		// below, so every "digit" is the SAME polynomial: the centered q0 lift
		// times that scalar, i.e. |digit| <= scalar*(q0>>1). Real key-switch
		// digits are bounded by their digit group's prime product, and the
		// prover's digit board sizes its range proof from exactly that -- which
		// for the ragged last group is a single prime and cannot hold this value.
		// Emit the true bound so the board is built against what ModUp actually
		// produces rather than an assumption it does not satisfy. Kept OUT of
		// ks_meta deliberately: a fifth meta field is read as the Galois element
		// (len(meta) >= 5), which would retag every digit shape of this switch.
		modupDigitScalar := uint64(1)
		// Scale the message from Q0/|m| to QL/|m|, where QL is the largest modulus used during the bootstrapping.
		if scale := (eval.Mod1Parameters.ScalingFactor().Float64() / eval.Mod1Parameters.MessageRatio()) / ctIn.Scale.Float64(); scale > 1 {

			scalar := uint64(math.Round(scale))
			modupDigitScalar = scalar

			for i := len(buffDecompQP) - 1; i >= 0; i-- {
				ringQ.MulScalar(buffDecompQP[0].Q, scalar, buffDecompQP[i].Q)
			}

			for i := len(buffDecompQP) - 1; i >= 0; i-- {
				ringP.MulScalar(buffDecompQP[0].P, scalar, buffDecompQP[i].P)
			}

			// vFHE: the message-ratio scaling. A pointwise multiply by a PUBLIC
			// integer, so it lands on the plainop board with the scalar broadcast
			// as its plaintext -- the same shape ScaleDown's scale-up already uses.
			// Snapshot first: the multiply is in place.
			var msBefore ring.Poly
			if eval.TraceSink != nil {
				msBefore = ringQ.NewPoly()
				msBefore.CopyLvl(levelQ, ctIn.Value[0])
			}
			ringQ.MulScalar(ctIn.Value[0], scalar, ctIn.Value[0])
			if eval.TraceSink != nil {
				vfhetrace.EmitPolyOp(eval.TraceSink, vfhetrace.OpPMUL, levelQ+1,
					[]ring.Poly{msBefore}, nil, []ring.Poly{ctIn.Value[0]},
					vfhetrace.ScalarBroadcastRef(scalar, Q, levelQ+1, N))
			}

			ctIn.Scale = ctIn.Scale.Mul(rlwe.NewScale(scale))
		}

		buffQ1 := poolQP.GetBuffPoly()
		defer poolQP.RecycleBuffPoly(buffQ1)

		ctTmp := &rlwe.Ciphertext{}
		ctTmp.Value = []ring.Poly{*buffQ1, ctIn.Value[1]}
		ctTmp.MetaData = ctIn.MetaData

		// Switch back to the dense key.
		//
		// vFHE: routed through the traced form when capturing, so this key-switch
		// lands on the same multi-P board every other one does. It is the last
		// unproved operation ModUp performs on the value it hands to CoeffsToSlots.
		// The traced form performs the identical MAC and mod-down, so production
		// gets the same result; without a sink it is not called at all.
		if eval.TraceSink != nil {
			hoistSink := btpPrefixSink{inner: eval.TraceSink, prefix: "ks_"}
			// c_in and result are the CALLER's to emit: the traced gadget product
			// proves prod -> mod-down, and what the caller does with the result is
			// what decides the board's outer relation. Here nothing is added to the
			// product and nothing is permuted, so c_in is zero and the automorphism
			// index is the identity (which assembleKSBuffer supplies when absent).
			zero := make([]uint64, ringQ.N())
			for k := 0; k < 2; k++ {
				for l := 0; l <= levelQ; l++ {
					hoistSink.Poly("c_in", k*(levelQ+1)+l, zero)
				}
			}
			if err := ks.GadgetProductHoistedMultiPTraced(levelQ, buffDecompQP,
				&eval.EvkSparseToDense.GadgetCiphertext, ctTmp, hoistSink); err != nil {
				// Capture must never change the result: fall back to the production
				// path rather than leaving ctTmp half-written.
				ks.GadgetProductHoisted(levelQ, buffDecompQP, &eval.EvkSparseToDense.GadgetCiphertext, ctTmp)
			} else {
				for k := 0; k < 2; k++ {
					for l := 0; l <= levelQ; l++ {
						hoistSink.Poly("result", k*(levelQ+1)+l,
							append([]uint64{}, ctTmp.Value[k].Coeffs[l]...))
					}
				}
				d := eval.BootstrappingParameters.Parameters.BaseRNSDecompositionVectorSize(levelQ, levelP)
				// BEFORE ks_meta: ks_meta is the flush trigger, so anything the
				// drain needs must already be in the sink. q is Q[0], the prime
				// the centered lift above is taken around.
				eval.TraceSink.Poly("ks_digit_bound", 0,
					[]uint64{modupDigitScalar * (q >> 1)})
				eval.TraceSink.Poly("ks_meta", 0, []uint64{
					uint64(ringQ.N()), uint64(levelQ + 1), uint64(levelP + 1), uint64(d)})
			}
		} else {
			ks.GadgetProductHoisted(levelQ, buffDecompQP, &eval.EvkSparseToDense.GadgetCiphertext, ctTmp)
		}
		// vFHE: folding the sparse->dense key-switch output back in. An ordinary
		// add on bare polys; the key-switch that produced ctTmp is the hoisted
		// gadget product, which still needs its own traced primitive.
		var addBefore ring.Poly
		if eval.TraceSink != nil {
			addBefore = ringQ.NewPoly()
			addBefore.CopyLvl(levelQ, ctIn.Value[0])
		}
		ringQ.Add(ctIn.Value[0], ctTmp.Value[0], ctIn.Value[0])
		if eval.TraceSink != nil {
			vfhetrace.EmitPolyOp(eval.TraceSink, vfhetrace.OpADD, levelQ+1,
				[]ring.Poly{addBefore}, []ring.Poly{ctTmp.Value[0]},
				[]ring.Poly{ctIn.Value[0]}, nil)
		}

	} else {

		for j := 0; j < N; j++ {

			coeff = ctIn.Value[1].Coeffs[0][j]
			pos, neg = 1, 0
			if coeff >= (q >> 1) {
				coeff = q - coeff
				pos, neg = 0, 1
			}

			for i := 1; i < levelQ+1; i++ {
				tmp = ring.BRedAdd(coeff, Q[i], BRCQ[i])
				ctIn.Value[1].Coeffs[i][j] = tmp*pos + (Q[i]-tmp)*neg
			}
		}

		ringQ.NTT(ctIn.Value[0], ctIn.Value[0])
		ringQ.NTT(ctIn.Value[1], ctIn.Value[1])

		// Scale the message from Q0/|m| to QL/|m|, where QL is the largest modulus used during the bootstrapping.
		if scale := (eval.Mod1Parameters.ScalingFactor().Float64() / eval.Mod1Parameters.MessageRatio()) / ctIn.Scale.Float64(); scale > 1 {

			scalar := uint64(math.Round(scale))

			// vFHE: same message-ratio scaling on the dense path, both components.
			var dBefore []ring.Poly
			if eval.TraceSink != nil {
				dBefore = []ring.Poly{ringQ.NewPoly(), ringQ.NewPoly()}
				dBefore[0].CopyLvl(levelQ, ctIn.Value[0])
				dBefore[1].CopyLvl(levelQ, ctIn.Value[1])
			}
			ringQ.MulScalar(ctIn.Value[0], scalar, ctIn.Value[0])
			ringQ.MulScalar(ctIn.Value[1], scalar, ctIn.Value[1])
			if eval.TraceSink != nil {
				vfhetrace.EmitPolyOp(eval.TraceSink, vfhetrace.OpPMUL, levelQ+1,
					dBefore, nil, []ring.Poly{ctIn.Value[0], ctIn.Value[1]},
					vfhetrace.ScalarBroadcastRef(scalar, Q, levelQ+1, N))
			}

			ctIn.Scale = ctIn.Scale.Mul(rlwe.NewScale(scale))
		}
	}

	//SubSum X -> (N/dslots) * Y^dslots
	// vFHE: the ladder is mirrored under instrumentation when capturing; falls
	// back to rlwe.Trace verbatim otherwise.
	return ctIn, eval.traceTraced(ctIn, eval.CoeffsToSlotsParameters.LogSlots)
}

// CoeffsToSlots applies the homomorphic decoding
func (eval Evaluator) CoeffsToSlots(ctIn *rlwe.Ciphertext) (ctReal, ctImag *rlwe.Ciphertext, err error) {
	// vFHE: scope the DFT trace capture to CoeffsToSlots (the dft() primitive is
	// shared with SlotsToCoeffs, which runs after EvalMod and is out of scope).
	if eval.TraceSink != nil {
		eval.DFTEvaluator.TraceSink = eval.TraceSink
		eval.DFTEvaluator.TraceStep = 0
		eval.DFTEvaluator.TracePrefix = "c2s"
		defer func() { eval.DFTEvaluator.TraceSink = nil }()
	}
	return eval.DFTEvaluator.CoeffsToSlotsNew(ctIn, eval.C2SDFTMatrix)
}

// EvalMod applies the homomorphic modular reduction by q.
func (eval Evaluator) EvalMod(ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext, err error) {
	// vFHE: capture the EvalMod op stream. The base step offset is advanced by a
	// fixed stride per call so the ctReal/ctImag EvalMod traces don't collide
	// (reset to 0 by the runtime when the bootstrap sink is set).
	if eval.TraceSink != nil {
		eval.Mod1Evaluator.TraceSink = eval.TraceSink
		base := eval.Mod1Evaluator.TraceStep
		defer func() {
			eval.Mod1Evaluator.TraceSink = nil
			eval.Mod1Evaluator.TraceStep = base + 50
		}()
	}
	if ctOut, err = eval.Mod1Evaluator.EvaluateNew(ctIn); err != nil {
		return nil, err
	}

	ctOut.Scale = eval.BootstrappingParameters.DefaultScale()
	return
}

// EvalModAndScale applies the homomorphic modular reduction by q and scales the output value (without
// consuming an additional level).
func (eval Evaluator) EvalModAndScale(ctIn *rlwe.Ciphertext, scaling complex128) (ctOut *rlwe.Ciphertext, err error) {
	if ctOut, err = eval.Mod1Evaluator.EvaluateAndScaleNew(ctIn, scaling); err != nil {
		return nil, err
	}

	ctOut.Scale = eval.BootstrappingParameters.DefaultScale()
	return
}

func (eval Evaluator) SlotsToCoeffs(ctReal, ctImag *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext, err error) {
	// vFHE: capture the SlotsToCoeffs DFT (same dft() primitive as CoeffsToSlots,
	// distinguished by the "s2c" region prefix).
	if eval.TraceSink != nil {
		eval.DFTEvaluator.TraceSink = eval.TraceSink
		eval.DFTEvaluator.TraceStep = 0
		eval.DFTEvaluator.TracePrefix = "s2c"
		defer func() { eval.DFTEvaluator.TraceSink = nil }()
	}
	return eval.DFTEvaluator.SlotsToCoeffsNew(ctReal, ctImag, eval.S2CDFTMatrix)
}

func (eval Evaluator) switchRingDegreeN1ToN2New(ctN1 *rlwe.Ciphertext) (ctN2 *rlwe.Ciphertext) {
	ctN2 = ckks.NewCiphertext(eval.BootstrappingParameters, 1, ctN1.Level())

	// Sanity check, this error should never happen unless this algorithm has been improperly
	// modified to pass invalid inputs.
	if err := eval.Evaluator.ApplyEvaluationKey(ctN1, eval.EvkN1ToN2, ctN2); err != nil {
		panic(err)
	}
	return
}

func (eval Evaluator) switchRingDegreeN2ToN1New(ctN2 *rlwe.Ciphertext) (ctN1 *rlwe.Ciphertext) {
	ctN1 = ckks.NewCiphertext(eval.ResidualParameters, 1, ctN2.Level())

	// Sanity check, this error should never happen unless this algorithm has been improperly
	// modified to pass invalid inputs.
	if err := eval.Evaluator.ApplyEvaluationKey(ctN2, eval.EvkN2ToN1, ctN1); err != nil {
		panic(err)
	}
	return
}

func (eval Evaluator) ComplexToRealNew(ctCmplx *rlwe.Ciphertext) (ctReal *rlwe.Ciphertext) {
	ctReal = ckks.NewCiphertext(eval.ResidualParameters, 1, ctCmplx.Level())

	// Sanity check, this error should never happen unless this algorithm has been improperly
	// modified to pass invalid inputs.
	if err := eval.DomainSwitcher.ComplexToReal(eval.Evaluator, ctCmplx, ctReal); err != nil {
		panic(err)
	}
	return
}

func (eval Evaluator) RealToComplexNew(ctReal *rlwe.Ciphertext) (ctCmplx *rlwe.Ciphertext) {
	ctCmplx = ckks.NewCiphertext(eval.BootstrappingParameters, 1, ctReal.Level())

	// Sanity check, this error should never happen unless this algorithm has been improperly
	// modified to pass invalid inputs.
	if err := eval.DomainSwitcher.RealToComplex(eval.Evaluator, ctReal, ctCmplx); err != nil {
		panic(err)
	}
	return
}

// packingContext contains the parameters used when packing (with Pack())
type packingContext struct {
	Params           *ckks.Parameters // Parameters of the ring we are packing to or unpacking from
	LogMaxDimensions ring.Dimensions  // maximum dimension of a packed ciphertext (logMaxDimensions <= params.LogMaxDimensions())
	LogSlots         int              // number of slots in a ct before packing (resp. after unpacking)
	NbPackedCTs      int              // number of cts to be packed (resp. to be unpacked into)
}

// PackAndSwitchN1ToN2 packs the ciphertexts into N1 and switch to N2 if N1 < N2
// then it packs the ciphertexts into N2.
func (eval Evaluator) PackAndSwitchN1ToN2(cts []rlwe.Ciphertext) ([]rlwe.Ciphertext, *packingContext, *packingContext, error) {

	var err error
	var packN1, packN2 *packingContext

	// If N1 < N2, we pack ciphertexts into N1 and then switch to N2
	if eval.ResidualParameters.N() != eval.BootstrappingParameters.N() {

		packN1 = &packingContext{&eval.ResidualParameters, eval.ResidualParameters.LogMaxDimensions(), cts[0].LogSlots(), len(cts)}

		// If the bootstrapping max slots are smaller than the max slots of N1, we only pack up to the former
		if eval.Parameters.LogMaxSlots() < eval.ResidualParameters.LogMaxSlots() {
			packN1.LogMaxDimensions = eval.Parameters.LogMaxDimensions()
		}
		if cts, err = eval.pack(cts, *packN1, eval.xPow2N1); err != nil {
			return nil, nil, nil, fmt.Errorf("cannot PackAndSwitchN1ToN2: PackN1: %w", err)
		}

		for i := range cts {
			cts[i] = *eval.switchRingDegreeN1ToN2New(&cts[i])
		}
	}

	// Packing ciphertexts into N2 (up to eval.Parameters.LogMaxDimensions())
	packN2 = &packingContext{&eval.BootstrappingParameters, eval.Parameters.LogMaxDimensions(), cts[0].LogSlots(), len(cts)}

	if cts, err = eval.pack(cts, *packN2, eval.xPow2N2); err != nil {
		return nil, nil, nil, fmt.Errorf("cannot PackAndSwitchN1ToN2: PackN1: %w", err)
	}

	return cts, packN1, packN2, nil
}

// UnpackAndSwitchN2ToN1 unpacks the ciphertexts into N2 and, if N1 < N2, it switches the ciphertexts
// to N1 and unpacks further into N1
func (eval Evaluator) UnpackAndSwitchN2ToN1(cts []rlwe.Ciphertext, ctxtN1, ctxtN2 *packingContext) ([]rlwe.Ciphertext, error) {

	var ctsOut []rlwe.Ciphertext

	logSlots := ctxtN2.LogSlots

	// Unpack ciphertexts in N2
	for i := range cts {
		ctsUnpack, err := eval.unpack(&cts[i], *ctxtN2, eval.xPow2InvN2)

		if err != nil {
			return nil, fmt.Errorf("cannot UnpackAndSwitchN2Tn1: UnpackN2: %w", err)
		}

		ctsOut = append(ctsOut, ctsUnpack...)
		ctxtN2.NbPackedCTs -= len(ctsUnpack)
	}

	// If N1 != N2 (i.e. ctxtN1 != nil): 1) switch cts to N1 2) unpack the cts in N1
	if ctxtN1 != nil {
		var ctsN1 []rlwe.Ciphertext
		logSlots = ctxtN1.LogSlots

		for i := range ctsOut {
			ctsOut[i] = *eval.switchRingDegreeN2ToN1New(&ctsOut[i])
		}

		for i := range ctsOut {
			ctsUnpack, err := eval.unpack(&ctsOut[i], *ctxtN1, eval.xPow2InvN1)
			if err != nil {
				return nil, fmt.Errorf("cannot UnpackAndSwitchN2Tn1: UnpackN1: %w", err)
			}

			ctsN1 = append(ctsN1, ctsUnpack...)
			ctxtN1.NbPackedCTs -= len(ctsUnpack)
		}

		ctsOut = ctsN1
	}

	// Set back the dimension of cts to its original value
	for i := range ctsOut {
		ctsOut[i].LogDimensions.Cols = logSlots
	}

	return ctsOut, nil
}

// unpack unpacks one sparse ciphertext of (log) dimension ctxt.logMaxDimensions
// into ctxt.NbPackedCTs ciphertexts of (log) dimension {0, ctxt.LogSlots}
func (eval Evaluator) unpack(ct *rlwe.Ciphertext, ctxt packingContext, xPow2Inv []ring.Poly) ([]rlwe.Ciphertext, error) {
	logPackCTs := ctxt.LogMaxDimensions.Cols - ctxt.LogSlots // log of number of CTs that can be packed in one ct

	cts := []rlwe.Ciphertext{*ct}
	if logPackCTs == 0 {
		return cts, nil
	}

	n := utils.Min(ctxt.NbPackedCTs, 1<<logPackCTs) // #cts to unpack from ct
	cts = append(cts, make([]rlwe.Ciphertext, n-1)...)

	for i := 1; i < len(cts); i++ {
		cts[i] = *ct.CopyNew()
	}

	r := ctxt.Params.RingQ().AtLevel(cts[0].Level())

	logGap := (ctxt.Params.LogMaxSlots() - ctxt.LogSlots) - 1 // log gap of CTs with params.N (minus one)

	/* #nosec G115 -- n-1 cannot be negative */
	for i := 0; i < utils.Min(bits.Len64(uint64(n-1)), logPackCTs); i++ {

		step := 1 << (i + 1)

		for j := 0; j < n; j += step {

			for k := step >> 1; k < step; k++ {

				if (j + k) >= n {
					break
				}

				r.MulCoeffsMontgomery(cts[j+k].Value[0], xPow2Inv[logGap-i], cts[j+k].Value[0])
				r.MulCoeffsMontgomery(cts[j+k].Value[1], xPow2Inv[logGap-i], cts[j+k].Value[1])
			}
		}
	}

	return cts, nil
}

// pack packs ctxt.NbPackedCTs sparse ciphertexts of (log) dimension {0, ctxt.LogSlots}
// into one ciphertext of (log) dimension ctxt.logMaxDimensions
func (eval Evaluator) pack(cts []rlwe.Ciphertext, ctxt packingContext, xPow2 []ring.Poly) ([]rlwe.Ciphertext, error) {

	var logSlots = ctxt.LogSlots
	var logMaxSlots = ctxt.LogMaxDimensions.Cols
	ringDegree := ctxt.Params.N()

	if logSlots > logMaxSlots {
		return nil, fmt.Errorf("cannot Pack: cts[0].LogSlots()=%d > logMaxSlots=%d", logSlots, logMaxSlots)
	}

	for i, ct := range cts {
		if s := ct.LogSlots(); s != logSlots {
			return nil, fmt.Errorf("cannot Pack: cts[%d].PlaintextLogSlots()=%d != cts[0].PlaintextLogSlots=%d", i, s, logSlots)
		}

		if N := ct.Value[0].N(); N != ringDegree {
			return nil, fmt.Errorf("cannot Pack: cts[%d].Value[0].N()=%d != params.N()=%d", i, N, ringDegree)
		}
	}

	logPackCTs := logMaxSlots - logSlots // log of number of CTs that can be packed in one ct
	logGap := (ctxt.Params.LogMaxSlots() - logSlots - 1)

	if logPackCTs == 0 {
		return cts, nil
	}

	for i := 0; i < logPackCTs; i++ {

		for j := 0; j < len(cts)>>1; j++ {

			eve := cts[j*2+0]
			odd := cts[j*2+1]

			level := utils.Min(eve.Level(), odd.Level())

			r := ctxt.Params.RingQ().AtLevel(level)

			r.MulCoeffsMontgomeryThenAdd(odd.Value[0], xPow2[logGap-i], eve.Value[0])
			r.MulCoeffsMontgomeryThenAdd(odd.Value[1], xPow2[logGap-i], eve.Value[1])

			cts[j] = eve
		}

		if len(cts)&1 == 1 {
			cts[len(cts)>>1] = cts[len(cts)-1]
			cts = cts[:len(cts)>>1+1]
		} else {
			cts = cts[:len(cts)>>1]
		}
	}

	for i := range cts {
		cts[i].LogDimensions = ctxt.LogMaxDimensions
	}

	return cts, nil
}

// emitModswitch captures one limb truncation under "ms_*" regions plus the
// "ms_meta" trigger the runtime drains. `limbs` is the SURVIVING count: the
// modswitch board binds the limbs that remain, and the dropped ones have no
// consumer and nothing provable about them.
func emitModswitch(sink ring.TraceSink, N, limbs int, operand, result []ring.Poly) {
	if sink == nil || limbs <= 0 || len(operand) == 0 {
		return
	}
	put := func(region string, polys []ring.Poly) {
		for p := 0; p < len(polys); p++ {
			for l := 0; l < limbs && l < len(polys[p].Coeffs); l++ {
				sink.Poly(region, p*limbs+l, append([]uint64{}, polys[p].Coeffs[l]...))
			}
		}
	}
	put("ms_operand", operand)
	put("ms_result", result)
	sink.Poly("ms_meta", 0, []uint64{uint64(N), uint64(limbs), uint64(len(result))})
}

// emitIntt captures the eval -> coefficient conversion under "intt_*" regions plus
// the "intt_meta" trigger the runtime drains. `limbs` is the ciphertext's limb
// count; both sides are emitted at it, since the board binds them pairwise.
func emitIntt(sink ring.TraceSink, N, limbs int, operand, result []ring.Poly) {
	if sink == nil || limbs <= 0 || len(operand) == 0 {
		return
	}
	put := func(region string, polys []ring.Poly) {
		for p := 0; p < len(polys); p++ {
			for l := 0; l < limbs && l < len(polys[p].Coeffs); l++ {
				sink.Poly(region, p*limbs+l, append([]uint64{}, polys[p].Coeffs[l]...))
			}
		}
	}
	put("intt_operand", operand)
	put("intt_result", result)
	sink.Poly("intt_meta", 0, []uint64{uint64(N), uint64(limbs), uint64(len(result))})
}

// btpPrefixSink wraps a TraceSink with a region prefix, so a captured key-switch
// lands under the "ks_*" keys the runtime's flushKS drains rather than colliding
// with the bootstrap's own regions. Mirrors the equivalent wrappers in the dft and
// mod1 packages.
type btpPrefixSink struct {
	inner  ring.TraceSink
	prefix string
}

func (p btpPrefixSink) Poly(region string, idx int, vals []uint64) {
	p.inner.Poly(p.prefix+region, idx, vals)
}
func (p btpPrefixSink) Stage(region string, idx, stage int, vals []uint64) {
	p.inner.Stage(p.prefix+region, idx, stage, vals)
}

// ksMetaFor closes a captured key-switch with the "ks_meta" trigger the runtime
// drains on. The traced automorphism emits the whole board; only the shape the
// assembler needs is the caller's.
func (eval Evaluator) ksMetaFor(level int, galEl uint64) {
	params := eval.BootstrappingParameters
	levelP := params.MaxLevelP()
	eval.TraceSink.Poly("ks_meta", 0, []uint64{
		uint64(params.N()), uint64(level + 1), uint64(levelP + 1),
		uint64(params.BaseRNSDecompositionVectorSize(level, levelP)), galEl})
}

// applyEvkTraced is ApplyEvaluationKey with the key-switch captured.
//
// ModUp opens by switching to the sparse secret, and closes by switching back.
// The closing one now goes through the traced hoisted gadget product; this is the
// opening one, which is an ordinary (non-hoisted) key-switch of c1 followed by
// result = (c0 + ks0, ks1) -- no permutation, so the identity index applies.
//
// BEST-EFFORT: any failure falls back to the production call, so a capture
// problem can never change the ciphertext.
func (eval Evaluator) applyEvkTraced(ct *rlwe.Ciphertext, evk *rlwe.EvaluationKey) error {
	params := eval.BootstrappingParameters
	levelP := params.MaxLevelP()
	if eval.TraceSink == nil || levelP < 1 || ct.Degree() != 1 {
		// Untraced on purpose when there is no sink. WITH a sink it means this
		// key-switch runs with no board and no trace at all -- see the note on the
		// other fallback below.
		if eval.TraceSink != nil {
			fmt.Fprintf(os.Stderr, "[vfhe] WARNING ModUp key-switch NOT captured "+
				"(levelP=%d degree=%d): it will change the ciphertext with no board "+
				"and no trace, so the value it produces is bound to nothing\n",
				levelP, ct.Degree())
		}
		return eval.ApplyEvaluationKey(ct, evk, ct)
	}
	level := ct.Level()
	ringQ := params.RingQ().AtLevel(level)
	N, L := ringQ.N(), level+1
	sink := btpPrefixSink{inner: eval.TraceSink, prefix: "ks_"}

	zero := make([]uint64, N)
	for l := 0; l < L; l++ {
		sink.Poly("c_in", 0*L+l, append([]uint64{}, ct.Value[0].Coeffs[l]...))
		sink.Poly("c_in", 1*L+l, zero)
	}
	ksCt := ckks.NewCiphertext(params, 1, level)
	ksCt.MetaData = ct.MetaData
	// SINGLE SPECIAL PRIME. The dense->sparse encapsulation key is generated at one
	// Q prime and one P prime (keys.go: paramsSparse takes Q[:1], P[:1]), so
	// evk.LevelP() == 0 and the MULTI-P traced product rejects it -- which used to
	// mean this key-switch silently fell back to the untraced production call. It
	// then changed the ciphertext with no board and no trace, and ModUp's INTT
	// operand became a value nothing produced.
	//
	// It needs no new gadget: one special prime IS the shape GadgetProductTraced and
	// vfhe::layout_relin already model (the same board the circuit's own relins use).
	// Emitted under its own "ks1_" prefix so the runtime assembles layout_relin
	// rather than layout_relin_multip, with the identity output permutation -- an
	// ApplyEvaluationKey does not permute.
	if evk.LevelP() == 0 {
		return eval.applyEvkSingleP(ct, evk, level)
	}
	if err := eval.GadgetProductMultiPTraced(level, ct.Value[1],
		&evk.GadgetCiphertext, ksCt, sink); err != nil {
		// A SILENT fallback here is the worst kind of gap: the ciphertext changes,
		// nothing is emitted, and the next op's operand is a value no proof produced
		// -- which is exactly how ModUp's INTT operand came to "appear nowhere else"
		// in the region index. The capture failing must never be quieter than the
		// capture succeeding.
		fmt.Fprintf(os.Stderr, "[vfhe] WARNING ModUp key-switch capture FAILED (%v); "+
			"falling back to the untraced call -- the value it produces will be bound "+
			"to nothing\n", err)
		return eval.ApplyEvaluationKey(ct, evk, ct)
	}
	ringQ.Add(ksCt.Value[0], ct.Value[0], ksCt.Value[0])
	ct.Value[0].CopyLvl(level, ksCt.Value[0])
	ct.Value[1].CopyLvl(level, ksCt.Value[1])
	for k := 0; k < 2; k++ {
		for l := 0; l < L; l++ {
			sink.Poly("result", k*L+l, append([]uint64{}, ct.Value[k].Coeffs[l]...))
		}
	}
	eval.ksMetaFor(level, 0) // no permutation: identity index
	return nil
}

// applyEvkSingleP is applyEvkTraced's single-special-prime path: the same
// key-switch, captured in the layout_relin shape.
//
// Region contract mirrors ckks.AutomorphismTraced, which is the same operation
// with a permutation on the end: target = c1, c_in = (c0, 0), then the gadget
// product (evk / mod-up / MAC / mod-down), then result = (c0 + ks0, ks1). No
// automorphism here, so the runtime binds the identity index.
//
// levelQ is clamped to the KEY's Q level: the encapsulation key lives at one prime,
// so the switch happens there whatever level the ciphertext arrived at.
func (eval Evaluator) applyEvkSingleP(ct *rlwe.Ciphertext, evk *rlwe.EvaluationKey,
	ctLevel int) error {
	params := eval.BootstrappingParameters
	levelQ := ctLevel
	if kq := evk.LevelQ(); kq < levelQ {
		levelQ = kq
	}
	if levelQ < 0 {
		return eval.ApplyEvaluationKey(ct, evk, ct)
	}
	ringQ := params.RingQ().AtLevel(levelQ)
	N, L := ringQ.N(), levelQ+1
	sink := btpPrefixSink{inner: eval.TraceSink, prefix: "ks1_"}

	zero := make([]uint64, N)
	for l := 0; l < L; l++ {
		sink.Poly("target", l, append([]uint64{}, ct.Value[1].Coeffs[l]...))
		sink.Poly("c_in", 0*L+l, append([]uint64{}, ct.Value[0].Coeffs[l]...))
		sink.Poly("c_in", 1*L+l, zero)
	}

	ksCt := ckks.NewCiphertext(params, 1, levelQ)
	ksCt.MetaData = ct.MetaData
	if err := eval.GadgetProductTraced(levelQ, ct.Value[1],
		&evk.GadgetCiphertext, ksCt, sink); err != nil {
		fmt.Fprintf(os.Stderr, "[vfhe] WARNING single-P ModUp key-switch capture "+
			"FAILED (%v); falling back to the untraced call -- the value it produces "+
			"will be bound to nothing\n", err)
		return eval.ApplyEvaluationKey(ct, evk, ct)
	}
	ringQ.Add(ksCt.Value[0], ct.Value[0], ksCt.Value[0])
	ct.Value[0].CopyLvl(levelQ, ksCt.Value[0])
	ct.Value[1].CopyLvl(levelQ, ksCt.Value[1])
	ct.Resize(ct.Degree(), levelQ)

	for k := 0; k < 2; k++ {
		for l := 0; l < L; l++ {
			sink.Poly("result", k*L+l, append([]uint64{}, ct.Value[k].Coeffs[l]...))
		}
	}
	// meta LAST: it is the runtime's flush trigger, so every region above must
	// already be present when it lands.
	eval.TraceSink.Poly("ks1_meta", 0, []uint64{uint64(N), uint64(L)})
	return nil
}

// traceTraced is rlwe.Trace with every operation captured.
//
// The Trace (SubSum) that closes ModUp is a LADDER -- a scalar pre-multiply, then
// one automorphism and one add per halving of the slot count -- and it is the last
// thing ModUp does to the value CoeffsToSlots then consumes. rlwe.Trace cannot be
// instrumented in place because the traced automorphism lives in the ckks package,
// which rlwe cannot import, so the ladder is mirrored here and each step routed
// through the traced primitives.
//
// Anything outside the bootstrap's own configuration (non-NTT input, conjugate
// invariant ring, no special prime) falls back to rlwe.Trace unchanged.
func (eval Evaluator) traceTraced(ct *rlwe.Ciphertext, logN int) error {
	params := eval.BootstrappingParameters
	levelP := params.MaxLevelP()
	if eval.TraceSink == nil || levelP < 1 || !ct.IsNTT ||
		params.RingType() != ring.Standard {
		return eval.Trace(ct, logN, ct)
	}
	level := ct.Level()
	ringQ := params.RingQ().AtLevel(level)

	gap := 1 << (params.LogN() - logN - 1)
	if logN == 0 {
		gap <<= 1
	}
	if gap <= 1 {
		return nil // Trace is the identity here
	}

	// Pre-multiplication by (N/n)^-1: a multiply by a PUBLIC integer, so it lands
	// on the plainop board with that constant reduced into each limb.
	NInv := new(big.Int).SetUint64(uint64(gap))
	NInv.ModInverse(NInv, ringQ.ModulusAtLevel[level])
	before := []ring.Poly{*ct.Value[0].CopyNew(), *ct.Value[1].CopyNew()}
	ringQ.MulScalarBigint(ct.Value[0], NInv, ct.Value[0])
	ringQ.MulScalarBigint(ct.Value[1], NInv, ct.Value[1])
	ref := make([][]uint64, level+1)
	for l := 0; l <= level; l++ {
		q := new(big.Int).SetUint64(ringQ.SubRings[l].Modulus)
		v := new(big.Int).Mod(NInv, q).Uint64()
		row := make([]uint64, ringQ.N())
		for x := range row {
			row[x] = v
		}
		ref[l] = row
	}
	vfhetrace.EmitPolyOp(eval.TraceSink, vfhetrace.OpPMUL, level+1, before, nil,
		[]ring.Poly{ct.Value[0], ct.Value[1]}, ref)

	// The ladder: automorphism, then accumulate.
	step := func(galEl uint64) error {
		buff := ckks.NewCiphertext(params, 1, level)
		buff.MetaData = ct.MetaData
		sink := btpPrefixSink{inner: eval.TraceSink, prefix: "ks_"}
		if err := eval.AutomorphismMultiPTraced(ct, galEl, buff, sink); err != nil {
			return err
		}
		eval.ksMetaFor(level, galEl)
		acc := []ring.Poly{*ct.Value[0].CopyNew(), *ct.Value[1].CopyNew()}
		ringQ.Add(ct.Value[0], buff.Value[0], ct.Value[0])
		ringQ.Add(ct.Value[1], buff.Value[1], ct.Value[1])
		vfhetrace.EmitPolyOp(eval.TraceSink, vfhetrace.OpADD, level+1, acc,
			[]ring.Poly{buff.Value[0], buff.Value[1]},
			[]ring.Poly{ct.Value[0], ct.Value[1]}, nil)
		return nil
	}
	for i := logN; i < params.LogN()-1; i++ {
		if err := step(params.GaloisElement(1 << i)); err != nil {
			return err
		}
	}
	if logN == 0 {
		if err := step(ringQ.NthRoot() - 1); err != nil {
			return err
		}
	}
	return nil
}
