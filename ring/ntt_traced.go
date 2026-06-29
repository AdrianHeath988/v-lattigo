package ring

// Non-unrolled, stage-snapshotting clones of the forward/inverse NTT butterfly
// networks (ntt.go nttLazy / inttLazy). They run the EXACT production butterfly
// (Montgomery roots + MRedLazy), so the math is bit-identical to the hot path;
// the only differences are (a) no loop unrolling, so each of the log2(N)
// butterfly stages is a clean snapshot point, and (b) each stage is reduced to
// its canonical residue and handed to the sink. Reducing the lazy production
// state mod q recovers exactly the canonical per-stage value (lazy reduction
// only adds multiples of q, which the final modulo removes), so the captured
// stages match a canonical recomputation while being the engine's real values.

import "math/bits"

// ForwardNTTTraced computes p2 = NTT(p1) on a single SubRing, emitting the
// canonical full-array state after each of the log2(N) butterfly stages via
// sink.Stage(region, idx, stage, .). The final stage equals the fully reduced
// NTT(p1), and p2 is left holding it. Mirrors nttLazy (ntt.go:223).
func ForwardNTTTraced(p1, p2 []uint64, N int, Q, MRedConstant uint64, roots []uint64,
	region string, idx int, sink TraceSink) {

	fourQ := 4 * Q
	twoQ := 2 * Q

	// Stage 0: m = 1, t = N/2, F = roots[1].
	t := N >> 1
	F := roots[1]
	for jx, jy := 0, t; jx < t; jx, jy = jx+1, jy+1 {
		p2[jx], p2[jy] = butterfly(p1[jx], p1[jy], F, twoQ, fourQ, Q, MRedConstant)
	}
	stage := 0
	emitStage(p2, N, Q, region, idx, stage, sink)

	// Stages 1..log2(N)-1: m = 2, 4, ..., N/2.
	for m := 2; m < N; m <<= 1 {
		t >>= 1
		for i := 0; i < m; i++ {
			j1 := (i * t) << 1
			j2 := j1 + t
			F = roots[m+i]
			for jx, jy := j1, j1+t; jx < j2; jx, jy = jx+1, jy+1 {
				p2[jx], p2[jy] = butterfly(p2[jx], p2[jy], F, twoQ, fourQ, Q, MRedConstant)
			}
		}
		stage++
		emitStage(p2, N, Q, region, idx, stage, sink)
	}

	// Match production NTTStandard's trailing full reduction so p2 holds the
	// canonical NTT output (the last emitted stage is already canonical).
	for i := 0; i < N; i++ {
		p2[i] = reduceCanonical(p2[i], Q)
	}
}

// BackwardNTTTraced computes p2 = INTT(p1) on a single SubRing, emitting the
// canonical state after each of the log2(N) inverse butterfly stages. Middle
// stages are the raw (reduced) butterfly outputs; the LAST stage carries the
// N^-1 normalization (production applies xNInv as a separate trailing pass) so
// that it equals INTT(p1). This matches the GS_INTT gadget contract: the
// gadget's final-stage constraint multiplies the snapshot by N to recover the
// raw butterfly sum and binds the snapshot to the normalized output. Mirrors
// inttLazy (ntt.go:568) + the INTTStandard NInv pass.
func BackwardNTTTraced(p1, p2 []uint64, N int, NInv, Q, MRedConstant uint64, roots []uint64,
	region string, idx int, sink TraceSink) {

	twoQ := Q << 1
	fourQ := Q << 2

	// Stage 0: first inverse butterfly block (t = 1, h = N/2).
	t := 1
	h := N >> 1
	for i, j1, j2 := 0, 0, t; i < h; i, j1, j2 = i+1, j1+2*t, j2+2*t {
		F := roots[h+i]
		for jx, jy := j1, j1+t; jx < j2; jx, jy = jx+1, jy+1 {
			p2[jx], p2[jy] = invbutterfly(p1[jx], p1[jy], F, twoQ, fourQ, Q, MRedConstant)
		}
	}
	logN := bits.TrailingZeros(uint(N))
	stage := 0
	if stage < logN-1 {
		emitStage(p2, N, Q, region, idx, stage, sink)
	}
	t <<= 1

	// Stages 1..log2(N)-1: m = N/2, N/4, ..., 2.
	for m := N >> 1; m > 1; m >>= 1 {
		h = m >> 1
		for i, j1, j2 := 0, 0, t; i < h; i, j1, j2 = i+1, j1+2*t, j2+2*t {
			F := roots[h+i]
			for jx, jy := j1, j1+t; jx < j2; jx, jy = jx+1, jy+1 {
				p2[jx], p2[jy] = invbutterfly(p2[jx], p2[jy], F, twoQ, fourQ, Q, MRedConstant)
			}
		}
		stage++
		t <<= 1
		if stage < logN-1 {
			emitStage(p2, N, Q, region, idx, stage, sink)
		}
	}

	// Trailing xN^-1 normalization (production INTTStandard). The normalized
	// output is both written to p2 and emitted as the LAST stage snapshot.
	out := make([]uint64, N)
	for i := 0; i < N; i++ {
		out[i] = MRed(reduceCanonical(p2[i], Q), NInv, Q, MRedConstant)
	}
	copy(p2, out)
	if sink != nil {
		sink.Stage(region, idx, logN-1, out)
	}
}

// emitStage reduces the current butterfly state to canonical residues and hands
// a copy to the sink.
func emitStage(p []uint64, N int, Q uint64, region string, idx, stage int, sink TraceSink) {
	if sink == nil {
		return
	}
	snap := make([]uint64, N)
	for k := 0; k < N; k++ {
		snap[k] = reduceCanonical(p[k], Q)
	}
	sink.Stage(region, idx, stage, snap)
}
