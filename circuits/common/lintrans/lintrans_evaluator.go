package lintrans

import (
	"fmt"
	"os"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
	"github.com/tuneinsight/lattigo/v6/schemes"
	"github.com/tuneinsight/lattigo/v6/utils"
)

type Evaluator struct {
	schemes.Evaluator
	pool *rlwe.BufferPool

	// vFHE trace capture for the BSGS inner-sum proof (the plaintext-weighted
	// accumulate tmp_j = Σ_i diag_{j+i}·Rot(operand,i)). nil ⇒ zero overhead.
	// Set by the dft.Evaluator before each matrix Evaluate; TraceStep selects the
	// matrix-step region index, TracePrefix the c2s/s2c region prefix.
	TraceSink   ring.TraceSink
	TraceStep   int
	TracePrefix string
}

func NewEvaluator(eval schemes.Evaluator) *Evaluator {

	return &Evaluator{
		Evaluator: eval,
		pool:      rlwe.NewPool(eval.GetRLWEParameters().RingQP()),
	}
}

// prefixSink wraps a ring.TraceSink, prefixing every region name. The giant-step
// key-switch capture emits the multi-P rotation-format regions (modup_coeff,
// modup_ntt, evk, prod, md_*, c_in, result, ...) under "ks_*" so they land in the
// shared bootstrap sink disjoint from ModUp's identically-named regions.
type prefixSink struct {
	inner  ring.TraceSink
	prefix string
}

func (p prefixSink) Poly(region string, idx int, vals []uint64) {
	p.inner.Poly(p.prefix+region, idx, vals)
}

func (p prefixSink) Stage(region string, idx, stage int, vals []uint64) {
	p.inner.Stage(p.prefix+region, idx, stage, vals)
}

// captureKSRotation emits ONE multi-P key-switch rotation (the CtS baby-step /
// giant-step primitive) into the trace sink under "ks_*" regions, in the normal
// rotation format prove_relin_multip consumes: c_in=(c0,0), the grouped multi-P
// key-switch of c1 (GadgetProductMultiPTraced → modup_coeff/ntt + evk + prod + ÷P
// mod-down), then result=γ(c0+KS(c1)) + idx, ending in "ks_meta" (the runtime
// drains + dumps on that meta). Re-runs the key-switch under instrumentation (the
// production baby rotation is hoisted; the math is identical) — BEST-EFFORT: a
// failure logs and returns, never aborting the real bootstrap.
func (eval Evaluator) captureKSRotation(c0, c1 ring.Poly, galEl uint64, evk *rlwe.GaloisKey, levelQ, levelP, step, gs, babyIdx int) {
	if eval.TraceSink == nil || levelP < 1 || evk == nil {
		return
	}
	params := eval.GetRLWEParameters()
	ringQ := params.RingQ().AtLevel(levelQ)
	ks := prefixSink{inner: eval.TraceSink, prefix: "ks_"}
	Lq := levelQ + 1
	N := ringQ.N()
	zero := make([]uint64, N)
	for l := 0; l <= levelQ; l++ {
		ks.Poly("c_in", 0*Lq+l, append([]uint64{}, c0.Coeffs[l]...))
		ks.Poly("c_in", 1*Lq+l, zero)
	}
	ksCt := rlwe.NewCiphertext(params, 1, levelQ)
	ksCt.MetaData.IsNTT = true
	if e := eval.GadgetProductMultiPTraced(levelQ, c1, &evk.GadgetCiphertext, ksCt, ks); e != nil {
		fmt.Fprintf(os.Stderr, "[vFHE] CtS KS capture skipped: %v\n", e)
		return
	}
	ringQ.Add(ksCt.Value[0], c0, ksCt.Value[0])
	rotIndex := eval.AutomorphismIndex(galEl)
	out0 := ringQ.NewPoly()
	out1 := ringQ.NewPoly()
	ringQ.AutomorphismNTTWithIndex(ksCt.Value[0], rotIndex, out0)
	ringQ.AutomorphismNTTWithIndex(ksCt.Value[1], rotIndex, out1)
	for l := 0; l <= levelQ; l++ {
		ks.Poly("result", 0*Lq+l, append([]uint64{}, out0.Coeffs[l]...))
		ks.Poly("result", 1*Lq+l, append([]uint64{}, out1.Coeffs[l]...))
	}
	ks.Poly("idx", 0, append([]uint64{}, rotIndex...))
	pfxFlag := 0
	if eval.TracePrefix == "s2c" {
		pfxFlag = 1
	}
	ks.Poly("meta", 0, []uint64{uint64(N), uint64(Lq), uint64(levelP + 1),
		uint64(params.BaseRNSDecompositionVectorSize(levelQ, levelP)), galEl,
		uint64(step), uint64(int64(gs)), uint64(pfxFlag), uint64(int64(babyIdx))})
}

// captureBabyResult emits the production baby-rotation OUTPUT (ctPreRot[i].Q) for
// the op-to-op (baby-rot → inner-sum) edge: it binds the inner-sum proof's operand
// in_di to the baby-rotation proof's output region — both are ctPreRot[i].Q, so the
// edge confirms the inner sum's di-th diagonal really consumes baby rotation i (the
// composition wiring). Verifying the rotation MATH (ctPreRot[i] = γ(KS(c1)+P·c0)) is
// the baby-rotation proof's job, not the edge's — so we capture the output DIRECTLY
// (no GadgetProductHoistedLazy re-invocation, which would race on the shared eval
// pool / overwritten ctTmp and corrupt the value). bm_result is keyed by
// babyKey(step,i) and KEPT (not streamed) for the post-hoc inner-sum edge pass.
func (eval Evaluator) captureBabyResult(res *rlwe.Element[ringqp.Poly], levelQ, step, i int) {
	if eval.TraceSink == nil || res == nil {
		return
	}
	bm := prefixSink{inner: eval.TraceSink, prefix: "bm_"}
	pfxFlag := 0
	if eval.TracePrefix == "s2c" {
		pfxFlag = 1
	}
	bk := babyKey(pfxFlag, step, i)
	for k := 0; k < 2; k++ {
		for l := 0; l <= levelQ; l++ {
			bm.Poly("result", (bk*2+k)*64+l, append([]uint64{}, res.Value[k].Q.Coeffs[l]...))
		}
	}
}

// babyKey packs (prefix c2s/s2c, matrix step, baby rotation index i) into one region
// index. The prefix MUST be folded in: c2s and s2c reuse step numbers (both start at
// 0) and share baby indices, so without it s2c (run later) overwrites c2s's bm_result
// and silently corrupts the c2s baby-rot edge. Stride 2^20 / bias 2^19 keeps it
// injective for |i| < 2^19 (far beyond any DFT rotation ≤ N).
func babyKey(pfxFlag, step, i int) int { return (pfxFlag*4096+step)*(1<<20) + i + (1 << 19) }

// captureModDownP traces ONE ÷P mod-down (QP→Q) — an inner-sum / accumulate base
// reduction in the BSGS that bridges the QP-domain inner-sum / giant-accumulate
// proofs to the Q-domain key-switch / output. It computes the SAME p2Q as
// ModDownQPtoQNTT (a faithful clone, so production is unchanged) while emitting
// "mdp_*" regions: prod = input Q-part, then md_advice / md_divround / md_ntt / ks
// (the output). The meta — emitted only on the LAST component — triggers the
// runtime to drain + self-check + dump the buffer.
// kind=0 ⇒ inner-sum tmp1 ÷P (producer = the lineartransform out_c1 at (step,gs));
// kind=1 ⇒ accumulate ÷P (producer = the giant-accumulate cout at step). The meta
// carries (kind, step, gs, pfx) so the runtime can bind this ÷P's PUBLIC input
// (prod) to the producing op's PUBLIC output — the op-to-op chaining.
func (eval Evaluator) captureModDownP(p1Q, p1P, p2Q ring.Poly, comp, nComp, levelQ, levelP int, last bool, kind, gs int) {
	mdp := prefixSink{inner: eval.TraceSink, prefix: "mdp_"}
	Lq := levelQ + 1
	N := len(p1Q.Coeffs[0])
	for l := 0; l <= levelQ; l++ {
		mdp.Poly("prod", comp*Lq+l, append([]uint64{}, p1Q.Coeffs[l]...))
	}
	eval.ModDownQPtoQNTTMultiPTraced(comp, levelQ, levelP, p1Q, p1P, p2Q, Lq, levelP+1, mdp)
	if last {
		pfxFlag := 0
		if eval.TracePrefix == "s2c" {
			pfxFlag = 1
		}
		mdp.Poly("meta", 0, []uint64{uint64(N), uint64(Lq), uint64(levelP + 1), uint64(nComp),
			uint64(kind), uint64(eval.TraceStep), uint64(gs), uint64(pfxFlag)})
	}
}

// EvaluateMany takes as input a ciphertext ctIn, a list of linear transformations [M0, M1, M2, ...] and a list of pre-allocated receiver opOut
// and evaluates opOut: [M0(ctIn), M1(ctIn), M2(ctIn), ...]
func (eval Evaluator) EvaluateMany(ctIn *rlwe.Ciphertext, linearTransformations []LinearTransformation, opOut []*rlwe.Ciphertext) (err error) {

	if len(opOut) < len(linearTransformations) {
		return fmt.Errorf("output *rlwe.Ciphertext slice is too small")
	}
	for i := range linearTransformations {
		if opOut[i] == nil {
			return fmt.Errorf("output slice contains unallocated ciphertext")
		}
	}

	var levelQ int
	levelP := linearTransformations[0].LevelP
	for _, lt := range linearTransformations {
		levelQ = utils.Max(levelQ, lt.LevelQ)

		if levelP != lt.LevelP {
			return fmt.Errorf("all linearTransformations must have the same levelP")
		}
	}

	levelQ = utils.Min(levelQ, ctIn.Level())

	poolQP := eval.pool.AtLevel(levelQ, levelP)
	buffDecompQP := poolQP.GetBuffDecompQP(*eval.GetRLWEParameters(), levelQ, levelP)
	defer eval.pool.RecycleBuffDecompQP(buffDecompQP)

	eval.DecomposeNTT(levelQ, levelP, levelP+1, ctIn.Value[1], ctIn.IsNTT, buffDecompQP)

	ctPreRot := map[int]*rlwe.Element[ringqp.Poly]{}

	for i, lt := range linearTransformations {

		if lt.N1 == 0 {
			if err = eval.MultiplyByDiagMatrix(ctIn, lt, buffDecompQP, opOut[i]); err != nil {
				return
			}
		} else {

			_, _, rotN2 := lt.BSGSIndex()

			if err = eval.PreRotatedCiphertextForDiagonalMatrixMultiplication(levelQ, levelP, ctIn, buffDecompQP, rotN2, ctPreRot); err != nil {
				return
			}

			if err = eval.MultiplyByDiagMatrixBSGS(ctIn, lt, ctPreRot, opOut[i]); err != nil {
				return
			}
		}
	}

	return
}

// PreRotatedCiphertextForDiagonalMatrixMultiplication populates ctPreRot with the pre-rotated ciphertext for the rotations rots and deletes rotated ciphertexts that are not in rots.
func (eval Evaluator) PreRotatedCiphertextForDiagonalMatrixMultiplication(levelQ, levelP int, ctIn *rlwe.Ciphertext, BuffDecompQP []ringqp.Poly, rots []int, ctPreRot map[int]*rlwe.Element[ringqp.Poly]) (err error) {

	params := eval.GetRLWEParameters()

	rotsMap := map[int]bool{}

	for _, i := range rots {
		rotsMap[i] = true
	}

	// Cleans the map of non-required ciphertexts
	for i := range ctPreRot {
		if _, ok := rotsMap[i]; !ok {
			delete(ctPreRot, i)
		}
	}

	// Computes the rotation only for the ones that are not already present.
	for _, i := range rots {
		if _, ok := ctPreRot[i]; i != 0 && !ok {
			ctPreRot[i] = rlwe.NewElementExtended(params, 1, levelQ, levelP)
			if err = eval.AutomorphismHoistedLazy(levelQ, ctIn, BuffDecompQP, params.GaloisElement(i), ctPreRot[i]); err != nil {
				return fmt.Errorf("eval.AutomorphismHoistedLazy: %w", err)
			}
			// vFHE: capture this baby-step rotation's key-switch (KS(ctIn.c1),
			// c_in = ctIn.c0) in the normal multi-P rotation format. Re-runs the
			// (un-hoisted) traced key-switch; streamed + drained per rotation.
			if eval.TraceSink != nil {
				if bevk, e := eval.CheckAndGetGaloisKey(params.GaloisElement(i)); e == nil {
					// gs=-1: baby rotation (no inner-sum ÷P feeds it). babyIdx=i = the
					// rotation index, so the inner sum can bind in_di == ctPreRot[i].
					eval.captureKSRotation(ctIn.Value[0], ctIn.Value[1], params.GaloisElement(i), bevk, levelQ, levelP, eval.TraceStep, -1, i)
				}
				// FAITHFUL op-to-op edge: capture the production baby-rotation OUTPUT
				// (this exact ctPreRot[i]) so the inner sum binds in_di == ctPreRot[i].
				eval.captureBabyResult(ctPreRot[i], levelQ, eval.TraceStep, i)
			}
		}
	}

	return
}

// EvaluateSequential takes as input a ciphertext ctIn and a list of linear transformations [M0, M1, M2, ...] and evaluates opOut:...M2(M1(M0(ctIn))
func (eval Evaluator) EvaluateSequential(ctIn *rlwe.Ciphertext, linearTransformations []LinearTransformation, opOut *rlwe.Ciphertext) (err error) {

	if err = eval.EvaluateMany(ctIn, linearTransformations[:1], []*rlwe.Ciphertext{opOut}); err != nil {
		return
	}

	if err = eval.Rescale(opOut, opOut); err != nil {
		return
	}

	for i := 1; i < len(linearTransformations); i++ {

		if err = eval.EvaluateMany(opOut, linearTransformations[i:i+1], []*rlwe.Ciphertext{opOut}); err != nil {
			return
		}

		if err = eval.Rescale(opOut, opOut); err != nil {
			return
		}
	}

	return
}

// MultiplyByDiagMatrix multiplies the Ciphertext ctIn by the plaintext matrix and returns the result on the Ciphertext
// opOut. Memory buffers for the decomposed ciphertext BuffDecompQP, BuffDecompQP must be provided, those are list of poly of ringQ and ringP
// respectively, each of size params.Beta().
// The naive approach is used (single hoisting and no baby-step giant-step), which is faster than MultiplyByDiagMatrixBSGS
// for matrix of only a few non-zero diagonals but uses more keys.
func (eval Evaluator) MultiplyByDiagMatrix(ctIn *rlwe.Ciphertext, matrix LinearTransformation, BuffDecompQP []ringqp.Poly, opOut *rlwe.Ciphertext) (err error) {

	*opOut.MetaData = *ctIn.MetaData
	opOut.Scale = opOut.Scale.Mul(matrix.Scale)

	params := eval.GetRLWEParameters()

	levelQ := utils.Min(opOut.Level(), utils.Min(ctIn.Level(), matrix.LevelQ))
	levelP := matrix.LevelP

	ringQP := params.RingQP().AtLevel(levelQ, levelP)
	poolQP := eval.pool.AtLevel(levelQ, levelP)
	ringQ := ringQP.RingQ
	ringP := ringQP.RingP

	buffQP0 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP0)
	buffQP1 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP1)
	buffQP2 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP2)
	buffQP3 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP3)
	buffQP4 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP4)
	buffQP5 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP5)
	buffCt := eval.pool.GetBuffCt(1, ringQ.Level())
	defer eval.pool.RecycleBuffCt(buffCt)

	opOut.Resize(opOut.Degree(), levelQ)

	QiOverF := params.QiOverflowMargin(levelQ)
	PiOverF := params.PiOverflowMargin(levelP)

	c0OutQP := ringqp.Poly{Q: opOut.Value[0], P: (*buffQP5).Q}
	c1OutQP := ringqp.Poly{Q: opOut.Value[1], P: (*buffQP5).P}

	ct0TimesP := (*buffQP0).Q // ct0 * P mod Q
	tmp0QP := *buffQP1
	tmp1QP := *buffQP2

	cQP := &rlwe.Element[ringqp.Poly]{}
	cQP.Value = []ringqp.Poly{*buffQP3, *buffQP4}
	cQP.MetaData = &rlwe.MetaData{}
	cQP.MetaData.IsNTT = true

	buffCt.Value[0].CopyLvl(levelQ, ctIn.Value[0])
	buffCt.Value[1].CopyLvl(levelQ, ctIn.Value[1])

	ctInTmp0, ctInTmp1 := buffCt.Value[0], buffCt.Value[1]

	ringQ.MulScalarBigint(ctInTmp0, ringP.ModulusAtLevel[levelP], ct0TimesP) // P*c0

	slots := 1 << matrix.LogDimensions.Cols

	keys := utils.GetSortedKeys(matrix.Vec)

	var state bool
	if keys[0] == 0 {
		state = true
		keys = keys[1:]
	}

	for i, k := range keys {

		k &= (slots - 1)

		galEl := params.GaloisElement(k)

		var evk *rlwe.GaloisKey
		var err error
		if evk, err = eval.CheckAndGetGaloisKey(galEl); err != nil {
			return fmt.Errorf("cannot MultiplyByDiagMatrix: Automorphism: CheckAndGetGaloisKey: %w", err)
		}

		if evk.LevelP() != levelP {
			return fmt.Errorf("LinearTransformation.LevelP = %d != GaloiKey[%d].LevelP() = %d: ensure that the levelP of the linear transformation is the same as the levelP of the GaloisKeys", levelP, galEl, evk.LevelP())
		}

		index := eval.AutomorphismIndex(galEl)

		if err = eval.GadgetProductHoistedLazy(levelQ, BuffDecompQP, &evk.GadgetCiphertext, cQP); err != nil {
			return fmt.Errorf("eval.GadgetProductHoistedLazy: %w", err)
		}

		ringQ.Add(cQP.Value[0].Q, ct0TimesP, cQP.Value[0].Q)
		ringQP.AutomorphismNTTWithIndex(cQP.Value[0], index, tmp0QP)
		ringQP.AutomorphismNTTWithIndex(cQP.Value[1], index, tmp1QP)

		pt := matrix.Vec[k]

		if i == 0 {
			// keyswitch(c1_Q) = (d0_QP, d1_QP)
			ringQP.MulCoeffsMontgomery(pt, tmp0QP, c0OutQP)
			ringQP.MulCoeffsMontgomery(pt, tmp1QP, c1OutQP)
		} else {
			// keyswitch(c1_Q) = (d0_QP, d1_QP)
			ringQP.MulCoeffsMontgomeryThenAdd(pt, tmp0QP, c0OutQP)
			ringQP.MulCoeffsMontgomeryThenAdd(pt, tmp1QP, c1OutQP)
		}

		if i%QiOverF == QiOverF-1 {
			ringQ.Reduce(c0OutQP.Q, c0OutQP.Q)
			ringQ.Reduce(c1OutQP.Q, c1OutQP.Q)
		}

		if i%PiOverF == PiOverF-1 {
			ringP.Reduce(c0OutQP.P, c0OutQP.P)
			ringP.Reduce(c1OutQP.P, c1OutQP.P)
		}
	}

	if len(keys)%QiOverF == 0 {
		ringQ.Reduce(c0OutQP.Q, c0OutQP.Q)
		ringQ.Reduce(c1OutQP.Q, c1OutQP.Q)
	}

	if len(keys)%PiOverF == 0 {
		ringP.Reduce(c0OutQP.P, c0OutQP.P)
		ringP.Reduce(c1OutQP.P, c1OutQP.P)
	}

	eval.ModDownQPtoQNTT(levelQ, levelP, c0OutQP.Q, c0OutQP.P, c0OutQP.Q) // sum(phi(c0 * P + d0_QP))/P
	eval.ModDownQPtoQNTT(levelQ, levelP, c1OutQP.Q, c1OutQP.P, c1OutQP.Q) // sum(phi(d1_QP))/P

	if state { // Rotation by zero
		ringQ.MulCoeffsMontgomeryThenAdd(matrix.Vec[0].Q, ctInTmp0, c0OutQP.Q) // opOut += c0_Q * plaintext
		ringQ.MulCoeffsMontgomeryThenAdd(matrix.Vec[0].Q, ctInTmp1, c1OutQP.Q) // opOut += c1_Q * plaintext
	}

	return
}

// MultiplyByDiagMatrixBSGS multiplies the Ciphertext "ctIn" by the plaintext matrix "matrix" and returns the result on the Ciphertext "opOut".
// ctInPreRotated can be obtained with [PreRotatedCiphertextForDiagonalMatrixMultiplication].
// The BSGS approach is used (double hoisting with baby-step giant-step), which is faster than MultiplyByDiagMatrix
// for matrix with more than a few non-zero diagonals and uses significantly less keys.
func (eval Evaluator) MultiplyByDiagMatrixBSGS(ctIn *rlwe.Ciphertext, matrix LinearTransformation, ctInPreRot map[int]*rlwe.Element[ringqp.Poly], opOut *rlwe.Ciphertext) (err error) {

	params := eval.GetRLWEParameters()

	*opOut.MetaData = *ctIn.MetaData
	opOut.Scale = opOut.Scale.Mul(matrix.Scale)

	levelQ := utils.Min(opOut.Level(), utils.Min(ctIn.Level(), matrix.LevelQ))
	levelP := matrix.LevelP

	ringQP := params.RingQP().AtLevel(levelQ, levelP)
	poolQP := eval.pool.AtLevel(levelQ, levelP)
	ringQ := ringQP.RingQ
	ringP := ringQP.RingP

	buffQP1 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP1)
	buffQP2 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP2)
	buffQP3 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP3)
	buffQP4 := poolQP.GetBuffPolyQP()
	defer poolQP.RecycleBuffPolyQP(buffQP4)
	buffP1 := poolQP.PoolP.GetBuffPoly()
	defer poolQP.PoolP.RecycleBuffPoly(buffP1)
	buffP2 := poolQP.PoolP.GetBuffPoly()
	defer poolQP.PoolP.RecycleBuffPoly(buffP2)
	buffCt := eval.pool.GetBuffCt(1, ringQ.Level())
	defer eval.pool.RecycleBuffCt(buffCt)

	opOut.Resize(opOut.Degree(), levelQ)

	QiOverF := params.QiOverflowMargin(levelQ) >> 1
	PiOverF := params.PiOverflowMargin(levelP) >> 1

	// Computes the N2 rotations indexes of the non-zero rows of the diagonalized DFT matrix for the baby-step giant-step algorithm
	index, _, _ := matrix.BSGSIndex()

	buffCt.Value[0].CopyLvl(levelQ, ctIn.Value[0])
	buffCt.Value[1].CopyLvl(levelQ, ctIn.Value[1])

	ctInTmp0, ctInTmp1 := buffCt.Value[0], buffCt.Value[1]

	// Accumulator inner loop
	tmp0QP := (*buffQP1)
	tmp1QP := (*buffQP2)

	// Accumulator outer loop
	cQP := &rlwe.Element[ringqp.Poly]{}
	cQP.Value = []ringqp.Poly{*buffQP3, *buffQP4}
	cQP.MetaData = &rlwe.MetaData{}
	cQP.IsNTT = true

	// Result in QP
	c0OutQP := ringqp.Poly{Q: opOut.Value[0], P: *buffP1}
	c1OutQP := ringqp.Poly{Q: opOut.Value[1], P: *buffP2}

	ringQ.MulScalarBigint(ctInTmp0, ringP.ModulusAtLevel[levelP], ctInTmp0) // P*c0
	ringQ.MulScalarBigint(ctInTmp1, ringP.ModulusAtLevel[levelP], ctInTmp1) // P*c1

	keys := utils.GetSortedKeys(index)

	// OUTER LOOP
	var cnt0 int
	for _, j := range keys {

		// INNER LOOP
		var cnt1 int
		for _, i := range index[j] {

			pt := matrix.Vec[j+i]
			ct := ctInPreRot[i]

			if i == 0 {
				if cnt1 == 0 {
					ringQ.MulCoeffsMontgomeryLazy(pt.Q, ctInTmp0, tmp0QP.Q)
					ringQ.MulCoeffsMontgomeryLazy(pt.Q, ctInTmp1, tmp1QP.Q)
					tmp0QP.P.Zero()
					tmp1QP.P.Zero()
				} else {
					ringQ.MulCoeffsMontgomeryLazyThenAddLazy(pt.Q, ctInTmp0, tmp0QP.Q)
					ringQ.MulCoeffsMontgomeryLazyThenAddLazy(pt.Q, ctInTmp1, tmp1QP.Q)
				}
			} else {
				if cnt1 == 0 {
					ringQP.MulCoeffsMontgomeryLazy(pt, ct.Value[0], tmp0QP)
					ringQP.MulCoeffsMontgomeryLazy(pt, ct.Value[1], tmp1QP)
				} else {
					ringQP.MulCoeffsMontgomeryLazyThenAddLazy(pt, ct.Value[0], tmp0QP)
					ringQP.MulCoeffsMontgomeryLazyThenAddLazy(pt, ct.Value[1], tmp1QP)
				}
			}

			if cnt1%QiOverF == QiOverF-1 {
				ringQ.Reduce(tmp0QP.Q, tmp0QP.Q)
				ringQ.Reduce(tmp1QP.Q, tmp1QP.Q)
			}

			if cnt1%PiOverF == PiOverF-1 {
				ringP.Reduce(tmp0QP.P, tmp0QP.P)
				ringP.Reduce(tmp1QP.P, tmp1QP.P)
			}

			cnt1++
		}

		if cnt1%QiOverF != 0 {
			ringQ.Reduce(tmp0QP.Q, tmp0QP.Q)
			ringQ.Reduce(tmp1QP.Q, tmp1QP.Q)
		}

		if cnt1%PiOverF != 0 {
			ringP.Reduce(tmp0QP.P, tmp0QP.P)
			ringP.Reduce(tmp1QP.P, tmp1QP.P)
		}

		// vFHE: capture this giant-step's plaintext-weighted inner accumulate
		// (tmp = Σ_i diag_{j+i}·rotop_i) BEFORE the giant-step mod-down/rotate
		// below mutates tmp1QP. Proved by the LinearTransform_gadget.
		if eval.TraceSink != nil {
			emitLinTransInner(eval.TraceSink, eval.TracePrefix, eval.TraceStep, cnt0,
				ringQ, ringP, matrix.Vec, index[j], j, ctInTmp0, ctInTmp1, ctInPreRot,
				tmp0QP, tmp1QP)
		}

		// If j != 0, then rotates ((tmp0QP.Q, tmp0QP.P), (tmp1QP.Q, tmp1QP.P)) by N1*j and adds the result on ((cQP.Value[0].Q, cQP.Value[0].P), (cQP.Value[1].Q, cQP.Value[1].P))
		if j != 0 {

			// Hoisting of the ModDown of sum(sum(phi(d1) * plaintext))
			// vFHE: trace this inner-sum ÷P (QP→Q) — the reduction that feeds the
			// giant-step key-switch — when capturing; else the plain mod-down.
			if eval.TraceSink != nil && levelP >= 1 {
				eval.captureModDownP(tmp1QP.Q, tmp1QP.P, tmp1QP.Q, 0, 1, levelQ, levelP, true, 0, cnt0)
			} else {
				eval.ModDownQPtoQNTT(levelQ, levelP, tmp1QP.Q, tmp1QP.P, tmp1QP.Q) // c1 * plaintext + sum(phi(d1) * plaintext) + phi(c1) * plaintext mod Q
			}

			galEl := params.GaloisElement(j)

			var evk *rlwe.GaloisKey
			var err error
			if evk, err = eval.CheckAndGetGaloisKey(galEl); err != nil {
				return fmt.Errorf("eval.CheckAndGetGaloisKey(%d): CheckAndGetGaloisKey: %w", galEl, err)
			}

			if evk.LevelP() != levelP {
				return fmt.Errorf("LinearTransformation.LevelP = %d != GaloiKey[%d].LevelP() = %d: ensure that the levelP of the linear transformation is the same as the levelP of the GaloisKeys", levelP, galEl, evk.LevelP())
			}

			rotIndex := eval.AutomorphismIndex(galEl)

			// vFHE: capture THIS giant-step's key-switch (cQP_j = KS(tmp1_j), c_in =
			// tmp0_j) in the normal multi-P rotation format. Streamed + drained per
			// rotation by the runtime (flushKS on the trailing "ks_meta"), so all
			// giant-steps of all matrix steps are traced without unbounded memory.
			if eval.TraceSink != nil {
				eval.captureKSRotation(tmp0QP.Q, tmp1QP.Q, galEl, evk, levelQ, levelP, eval.TraceStep, cnt0, -1)
			}

			// EvaluationKey(P*phi(tmpRes_1)) = (d0, d1) in base QP
			if err = eval.GadgetProductLazy(levelQ, tmp1QP.Q, &evk.GadgetCiphertext, cQP); err != nil {
				return fmt.Errorf("eval.GadgetProductLazy: %w", err)
			}

			ringQP.Add(cQP.Value[0], tmp0QP, cQP.Value[0])

			// vFHE: capture this giant-step's rotate input cQP_j (= KS(tmp1)+tmp0)
			// and its automorphism index, BEFORE the rotate+accumulate below.
			if eval.TraceSink != nil {
				emitLinTransGiant(eval.TraceSink, eval.TracePrefix, eval.TraceStep, cnt0,
					ringQ, cQP.Value[0].Q, cQP.Value[1].Q, rotIndex, false)
			}

			// Outer loop rotations
			if cnt0 == 0 {
				ringQP.AutomorphismNTTWithIndex(cQP.Value[0], rotIndex, c0OutQP)
				ringQP.AutomorphismNTTWithIndex(cQP.Value[1], rotIndex, c1OutQP)
			} else {
				ringQP.AutomorphismNTTWithIndexThenAddLazy(cQP.Value[0], rotIndex, c0OutQP)
				ringQP.AutomorphismNTTWithIndexThenAddLazy(cQP.Value[1], rotIndex, c1OutQP)
			}

			// Else directly adds on ((cQP.Value[0].Q, cQP.Value[0].P), (cQP.Value[1].Q, cQP.Value[1].P))
		} else {
			// vFHE: j==0 giant-step is the identity automorphism of (tmp0, tmp1).
			if eval.TraceSink != nil {
				emitLinTransGiant(eval.TraceSink, eval.TracePrefix, eval.TraceStep, cnt0,
					ringQ, tmp0QP.Q, tmp1QP.Q, nil, true)
			}
			if cnt0 == 0 {
				c0OutQP.CopyLvl(levelQ, levelP, tmp0QP)
				c1OutQP.CopyLvl(levelQ, levelP, tmp1QP)
			} else {
				ringQP.AddLazy(c0OutQP, tmp0QP, c0OutQP)
				ringQP.AddLazy(c1OutQP, tmp1QP, c1OutQP)
			}
		}

		if cnt0%QiOverF == QiOverF-1 {
			ringQ.Reduce(opOut.Value[0], opOut.Value[0])
			ringQ.Reduce(opOut.Value[1], opOut.Value[1])
		}

		if cnt0%PiOverF == PiOverF-1 {
			ringP.Reduce(c0OutQP.P, c0OutQP.P)
			ringP.Reduce(c1OutQP.P, c1OutQP.P)
		}

		cnt0++
	}

	if cnt0%QiOverF != 0 {
		ringQ.Reduce(opOut.Value[0], opOut.Value[0])
		ringQ.Reduce(opOut.Value[1], opOut.Value[1])
	}

	if cnt0%PiOverF != 0 {
		ringP.Reduce(c0OutQP.P, c0OutQP.P)
		ringP.Reduce(c1OutQP.P, c1OutQP.P)
	}

	// vFHE: capture the accumulated output c_out = Σ_j γ_{N1·j}(cQP_j) BEFORE the
	// final ÷P mod-down, so the giant-step permutation+accumulate can be checked.
	if eval.TraceSink != nil {
		emitLinTransGiantOut(eval.TraceSink, eval.TracePrefix, eval.TraceStep,
			ringQ, opOut.Value[0], opOut.Value[1], cnt0)
	}

	// vFHE: trace the FINAL accumulate ÷P (both components, QP→Q) — the reduction
	// producing this matrix step's output — when capturing; else plain mod-down.
	if eval.TraceSink != nil && levelP >= 1 {
		eval.captureModDownP(opOut.Value[0], c0OutQP.P, opOut.Value[0], 0, 2, levelQ, levelP, false, 1, 0)
		eval.captureModDownP(opOut.Value[1], c1OutQP.P, opOut.Value[1], 1, 2, levelQ, levelP, true, 1, 0)
	} else {
		eval.ModDownQPtoQNTT(levelQ, levelP, opOut.Value[0], c0OutQP.P, opOut.Value[0]) // sum(phi(c0 * P + d0_QP))/P
		eval.ModDownQPtoQNTT(levelQ, levelP, opOut.Value[1], c1OutQP.P, opOut.Value[1]) // sum(phi(d1_QP))/P
	}

	return
}
