package ring

import "testing"

// TestModUpCenteredLift validates the centered CRT lift: each lifted limb is the
// centered q0-integer reduced mod q_i, with |x| < q0/2 (the non-vacuous bound the
// prover binds). Checks the emitted trace too (operand/sign/lift consistency).
func TestModUpCenteredLift(t *testing.T) {
	// N=16, NTT-friendly primes ≡ 1 mod 2N. q0 small so centering is exercised.
	const N = 16
	r, err := NewRing(N, []uint64{97, 193, 257, 353})
	if err != nil {
		t.Fatalf("NewRing: %v", err)
	}
	levelQ := r.level
	q0 := r.SubRings[0].Modulus
	half := q0 >> 1

	// q0-limb coefficients spanning the full range (some >= q0/2 -> negative x).
	a := make([]uint64, N)
	for j := 0; j < N; j++ {
		a[j] = uint64((j * 13) % int(q0))
	}

	cap := map[string][]uint64{}
	lift := map[int][]uint64{}
	sink := sinkFunc{poly: func(region string, idx int, v []uint64) {
		cp := append([]uint64{}, v...)
		if region == "modup_lift" {
			lift[idx] = cp
		} else {
			cap[region+"_"+itoa(idx)] = cp
		}
	}}

	out := r.NewPoly()
	r.ModUpCentered(0, levelQ, a, out, sink)

	// Reference: signed centered integer, lifted to each prime by exact modular reduction.
	for j := 0; j < N; j++ {
		// centered integer x in (-q0/2, q0/2]
		var xMod = func(qi uint64) uint64 {
			if a[j] < half {
				return a[j] % qi
			}
			// x = a - q0 < 0 ; x mod qi = qi - ((q0 - a) mod qi)
			t := (q0 - a[j]) % qi
			if t == 0 {
				return 0
			}
			return qi - t
		}
		if out.Coeffs[0][j] != a[j] {
			t.Fatalf("q0 limb changed at %d: %d != %d", j, out.Coeffs[0][j], a[j])
		}
		for i := 1; i <= levelQ; i++ {
			qi := r.SubRings[i].Modulus
			want := xMod(qi)
			if out.Coeffs[i][j] != want {
				t.Fatalf("lift limb %d coeff %d: got %d want %d (a=%d q0=%d qi=%d)",
					i, j, out.Coeffs[i][j], want, a[j], q0, qi)
			}
		}
	}

	// Trace consistency: operand == a; sign flags negative-x; lift regions match out.
	if op := cap["modup_operand_0"]; op != nil {
		for j := 0; j < N; j++ {
			if op[j] != a[j] {
				t.Fatalf("operand trace mismatch at %d", j)
			}
		}
	} else {
		t.Fatal("no modup_operand emitted")
	}
	sign := cap["modup_sign_0"]
	for j := 0; j < N; j++ {
		want := uint64(0)
		if a[j] >= half {
			want = 1
		}
		if sign[j] != want {
			t.Fatalf("sign[%d]=%d want %d", j, sign[j], want)
		}
	}
	for i := 1; i <= levelQ; i++ {
		lv := lift[0*(levelQ+1)+i]
		for j := 0; j < N; j++ {
			if lv[j] != out.Coeffs[i][j] {
				t.Fatalf("lift trace limb %d coeff %d mismatch", i, j)
			}
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}
