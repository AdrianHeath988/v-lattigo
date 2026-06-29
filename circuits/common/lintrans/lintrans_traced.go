package lintrans

// vFHE trace capture for the homomorphic linear transform (CoeffsToSlots /
// SlotsToCoeffs DFT matrix), BSGS form. Captures the genuinely-new arithmetic:
// the plaintext-weighted inner accumulate of one giant-step
//
//   tmp_j = Σ_{i ∈ index[j]}  diag_{j+i} · Rot(operand, i)      (eval domain)
//
// proved by the C++ LinearTransform_gadget (out = Σ_k w_k·in_k). Lattigo's
// production mult is `MulCoeffsMontgomeryLazy(diag, rotop)` = diag·rotop·R^-1
// with the DIAGONAL in Montgomery form and the rotated operand standard, so the
// Montgomery R cancels: tmp = Σ IMForm(diag)·rotop. We therefore capture the
// diagonals de-Montgomery'd (IMForm) and the operands / inner-sums reduced to
// canonical residues — the form the gadget proves over plain residues.
//
// BOTH the Q-domain and the P-domain (extended basis) inner sums are captured:
// the P part feeds the giant-step key-switch, so proving it closes the inner
// transform over the full QP basis. In the P domain the i==0 diagonal contributes
// nothing (its operand is the Q-only P*c0), so its P operand is emitted as zeros —
// out_P = Σ w_P·in_P then still holds with in_P[0]=0. The giant-step rotations +
// outer accumulate (consuming these QP inner sums) reuse the rotate/add gadgets.

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
)

const (
	ltMaxGS   = 256 // giant-steps per matrix
	ltMaxDiag = 64  // baby-step diagonals per giant-step
	ltMaxLimb = 64  // RNS limbs
)

// Packing: each sink.Poly emits ONE limb (length-N coeffs). The base index packs
// (step, giant-step, diagonal, component); the per-limb idx is base*ltMaxLimb+l.
// Q and P domains use disjoint region-name suffixes (P regions carry a "P").
func ltOutBase(step, gs, comp int) int { return (step*ltMaxGS+gs)*2 + comp }
func ltWBase(step, gs, di int) int     { return (step*ltMaxGS+gs)*ltMaxDiag + di }
func ltInBase(step, gs, di, comp int) int {
	return ((step*ltMaxGS+gs)*ltMaxDiag+di)*2 + comp
}
func ltMetaIdx(step, gs int) int { return step*ltMaxGS + gs }

// emitLinTransInner captures one giant-step's canonical inner sum + its inputs,
// over both the Q and P domains. gs is the giant-step counter (cnt0); diags =
// index[j]; j the giant-step key.
func emitLinTransInner(sink ring.TraceSink, pfx string, step, gs int,
	ringQ, ringP *ring.Ring, vec map[int]ringqp.Poly, diags []int, j int,
	ct0, ct1 ring.Poly, preRot map[int]*rlwe.Element[ringqp.Poly],
	tmp0, tmp1 ringqp.Poly) {

	// emitDomain captures the inner sum out_c = Σ_di w_di·in_di for one RNS basis
	// (Q or P). suffix is "" for Q, "P" for P. diagP picks the diagonal's domain
	// poly; outP the inner-sum domain poly; opP(i,c) the rotated operand (or nil
	// for an absent/zero operand, e.g. i==0 in the P domain).
	emitDomain := func(rng *ring.Ring, suffix string,
		diagOf func(k int) ring.Poly, outOf func(c int) ring.Poly,
		opOf func(i, c int) (ring.Poly, bool)) {

		level := rng.Level()
		buf := rng.NewPoly()
		zero := make([]uint64, ringQ.N())
		rin := "lt" + pfx + "_in" + suffix
		rw := "lt" + pfx + "_w" + suffix
		rout := "lt" + pfx + "_out" + suffix

		emitCanon := func(region string, base int, src ring.Poly) {
			rng.Reduce(src, buf) // lazy → canonical [0,q)
			for l := 0; l <= level; l++ {
				sink.Poly(region, base*ltMaxLimb+l, append([]uint64{}, buf.Coeffs[l]...))
			}
		}

		emitCanon(rout, ltOutBase(step, gs, 0), outOf(0))
		emitCanon(rout, ltOutBase(step, gs, 1), outOf(1))

		w := rng.NewPoly()
		for di, i := range diags {
			rng.IMForm(diagOf(i), w) // de-Montgomery the diagonal
			rng.Reduce(w, w)
			for l := 0; l <= level; l++ {
				sink.Poly(rw, ltWBase(step, gs, di)*ltMaxLimb+l,
					append([]uint64{}, w.Coeffs[l]...))
			}
			for c := 0; c < 2; c++ {
				op, present := opOf(i, c)
				if !present {
					for l := 0; l <= level; l++ {
						sink.Poly(rin, ltInBase(step, gs, di, c)*ltMaxLimb+l,
							append([]uint64{}, zero[:rng.N()]...))
					}
					continue
				}
				emitCanon(rin, ltInBase(step, gs, di, c), op)
			}
		}
	}

	// Q domain: operand for i==0 is P*operand (ctInTmp0/1, Q-only); else preRot.Q.
	emitDomain(ringQ, "",
		func(k int) ring.Poly { return vec[j+k].Q },
		func(c int) ring.Poly {
			if c == 0 {
				return tmp0.Q
			}
			return tmp1.Q
		},
		func(i, c int) (ring.Poly, bool) {
			if i == 0 {
				if c == 0 {
					return ct0, true
				}
				return ct1, true
			}
			return preRot[i].Value[c].Q, true
		})

	// P domain: i==0 contributes no P operand (emitted as zeros); else preRot.P.
	emitDomain(ringP, "P",
		func(k int) ring.Poly { return vec[j+k].P },
		func(c int) ring.Poly {
			if c == 0 {
				return tmp0.P
			}
			return tmp1.P
		},
		func(i, c int) (ring.Poly, bool) {
			if i == 0 {
				return ring.Poly{}, false
			}
			return preRot[i].Value[c].P, true
		})

	sink.Poly("lt"+pfx+"_meta", ltMetaIdx(step, gs),
		[]uint64{uint64(ringQ.Level()), 2, uint64(len(diags)), uint64(j),
			uint64(ringP.Level())})

	// vFHE op-to-op: the rotation index (baby-step offset) of each diagonal, so the
	// runtime can bind in_di == the baby rotation ctPreRot[diags[di]].
	di := make([]uint64, len(diags))
	for k, i := range diags {
		di[k] = uint64(int64(i))
	}
	sink.Poly("lt"+pfx+"_diagidx", ltMetaIdx(step, gs), di)
}

// --- Giant-step capture (the outer accumulate that ties the proven inner sums
// into the matrix-step output) ---
//
// For each giant-step j the BSGS forms the rotate INPUT
//   cQP_j = ( KS(tmp1_j)[0] + tmp0_j ,  KS(tmp1_j)[1] )      (j != 0)
//   cQP_j = ( tmp0_j , tmp1_j )                              (j == 0, identity)
// and accumulates its automorphism by N1·j into the output
//   c_out[comp][m] = Σ_j  cQP_j[comp][ idx_j[m] ]            (eval/NTT domain)
// (AutomorphismNTTWithIndex: out[m] = in[idx[m]]; idx_j = identity for j==0).
// The key-switch KS(tmp1_j) is the rotate/relin gadget (proved separately); this
// capture proves the PERMUTATION + ACCUMULATE glue at the residue level. Q domain
// (the permutation is position-based, identical across all RNS limbs).

func ltgCqpBase(step, gs, comp int) int { return (step*ltMaxGS+gs)*2 + comp }
func ltgCoutBase(step, comp int) int    { return step*2 + comp }

// emitLinTransGiant captures one giant-step's rotate input cQP_j + its
// automorphism index (nil ⇒ identity, j==0). c0Q/c1Q are the two components' Q
// polys (lazy; reduced here).
func emitLinTransGiant(sink ring.TraceSink, pfx string, step, gs int,
	ringQ *ring.Ring, c0Q, c1Q ring.Poly, idx []uint64, identity bool) {

	level := ringQ.Level()
	buf := ringQ.NewPoly()
	emit := func(comp int, src ring.Poly) {
		ringQ.Reduce(src, buf)
		for l := 0; l <= level; l++ {
			sink.Poly("ltg"+pfx+"_cqp", ltgCqpBase(step, gs, comp)*ltMaxLimb+l,
				append([]uint64{}, buf.Coeffs[l]...))
		}
	}
	emit(0, c0Q)
	emit(1, c1Q)

	N := ringQ.N()
	id := make([]uint64, N)
	if identity {
		for m := 0; m < N; m++ {
			id[m] = uint64(m)
		}
	} else {
		copy(id, idx)
	}
	sink.Poly("ltg"+pfx+"_idx", step*ltMaxGS+gs, id)
}

// emitLinTransGiantOut captures the accumulated output c_out (both components, Q,
// lazy → reduced) after the giant-step loop, BEFORE the final ÷P mod-down. nGS is
// the giant-step count for this matrix step.
func emitLinTransGiantOut(sink ring.TraceSink, pfx string, step int,
	ringQ *ring.Ring, c0Q, c1Q ring.Poly, nGS int) {

	level := ringQ.Level()
	buf := ringQ.NewPoly()
	emit := func(comp int, src ring.Poly) {
		ringQ.Reduce(src, buf)
		for l := 0; l <= level; l++ {
			sink.Poly("ltg"+pfx+"_cout", ltgCoutBase(step, comp)*ltMaxLimb+l,
				append([]uint64{}, buf.Coeffs[l]...))
		}
	}
	emit(0, c0Q)
	emit(1, c1Q)
	sink.Poly("ltg"+pfx+"_gmeta", step, []uint64{uint64(nGS), uint64(level)})
}
