package ring

import (
	"testing"
)

// TestTracedNTTMatchesProduction checks that the non-unrolled, stage-snapshotting
// traced clones reproduce the production (unrolled) NTT/INTT at the canonical
// (fully reduced) level, and that the emitted last stages equal the canonical
// forward output / N^-1-normalized inverse output.
func TestTracedNTTMatchesProduction(t *testing.T) {
	// N=16 exercises the production UNROLLED path while the clone is non-unrolled,
	// so this also checks cross-path equivalence. q=97 ≡ 1 mod 2N (NTT-friendly).
	const N = 16
	r, err := NewRing(N, []uint64{97})
	if err != nil {
		t.Fatalf("NewRing: %v", err)
	}
	s := r.SubRings[0]
	q := s.Modulus

	in := make([]uint64, N)
	for i := range in {
		in[i] = uint64((i*7 + 3) % int(q))
	}

	// Production reference.
	prodNTT := make([]uint64, N)
	s.NTT(append([]uint64{}, in...), prodNTT)

	// Traced forward.
	type cap struct {
		stages map[int][]uint64
	}
	fwd := &cap{stages: map[int][]uint64{}}
	tracedFwd := make([]uint64, N)
	ForwardNTTTraced(append([]uint64{}, in...), tracedFwd, N, q, s.MRedConstant, s.RootsForward,
		"ntt", 0, sinkFunc{stage: func(_ string, _, st int, v []uint64) {
			fwd.stages[st] = append([]uint64{}, v...)
		}})

	for i := 0; i < N; i++ {
		if tracedFwd[i] != prodNTT[i] {
			t.Fatalf("forward p2[%d]=%d != production %d", i, tracedFwd[i], prodNTT[i])
		}
	}
	last := fwd.stages[len(fwd.stages)-1]
	for i := 0; i < N; i++ {
		if last[i] != prodNTT[i] {
			t.Fatalf("forward last-stage[%d]=%d != production NTT %d", i, last[i], prodNTT[i])
		}
	}

	// Production inverse reference (round-trips to `in`).
	prodINTT := make([]uint64, N)
	s.INTT(append([]uint64{}, prodNTT...), prodINTT)
	for i := 0; i < N; i++ {
		if prodINTT[i] != in[i] {
			t.Fatalf("production INTT roundtrip failed at %d", i)
		}
	}

	// Traced inverse: last stage must be the normalized output == in.
	inv := &cap{stages: map[int][]uint64{}}
	tracedInv := make([]uint64, N)
	BackwardNTTTraced(append([]uint64{}, prodNTT...), tracedInv, N, s.NInv, q, s.MRedConstant, s.RootsBackward,
		"intt", 0, sinkFunc{stage: func(_ string, _, st int, v []uint64) {
			inv.stages[st] = append([]uint64{}, v...)
		}})
	for i := 0; i < N; i++ {
		if tracedInv[i] != in[i] {
			t.Fatalf("traced INTT p2[%d]=%d != original %d", i, tracedInv[i], in[i])
		}
	}
	invLast := inv.stages[len(inv.stages)-1]
	for i := 0; i < N; i++ {
		if invLast[i] != in[i] {
			t.Fatalf("inverse last-stage[%d]=%d != original %d (N^-1 normalization)", i, invLast[i], in[i])
		}
	}
}

// sinkFunc is a tiny TraceSink adapter for tests.
type sinkFunc struct {
	poly  func(region string, idx int, vals []uint64)
	stage func(region string, idx, stage int, vals []uint64)
}

func (s sinkFunc) Poly(region string, idx int, vals []uint64) {
	if s.poly != nil {
		s.poly(region, idx, vals)
	}
}
func (s sinkFunc) Stage(region string, idx, stage int, vals []uint64) {
	if s.stage != nil {
		s.stage(region, idx, stage, vals)
	}
}
