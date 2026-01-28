//go:build goexperiment.simd

/*
 * Copyright (c) 2011 Madelynn Blue <blue.mlynn@gmail.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

package fft

import (
	"simd/archsimd"
	"unsafe"
)

// useSIMD checks if SIMD operations are available and beneficial
var useSIMD = archsimd.X86.AVX2()

func radix2Worker(
	lx, start, end int,
	stage, s_2, blocks int,
	r, t, factors []complex128,
) {
	// Choose worker function based on SIMD availability
	if useSIMD && lx >= 8 {
		// SIMD-optimized worker
		processBlocksSIMD(start, end, stage, s_2, blocks, r, t, factors)
	} else {
		// Fallback to scalar worker
		radix2WorkerScalar(lx, start, end, stage, s_2, blocks, r, t, factors)
	}
}

// processBlocksSIMD performs butterfly operations using SIMD instructions
func processBlocksSIMD(start, end, stage, s_2, blocks int, r, t []complex128, factors []complex128) {
	// Process blocks in this work unit
	for nb := start; nb < end; nb += stage {
		if stage == 2 {
			// Handle stage 2 separately (no twiddle factor)
			n1 := nb + 1
			rn := r[nb]
			rn1 := r[n1]
			t[nb] = rn + rn1
			t[n1] = rn - rn1
			continue
		}

		// Process using SIMD when possible
		// Process 2 complex numbers at a time (4 float64 values)
		j := 0
		simdLimit := (s_2 / 2) * 2 // Round down to nearest even number

		if simdLimit >= 2 {
			// SIMD path: process 2 complex128 values at once
			for ; j < simdLimit; j += 2 {
				idx1 := j + nb
				idx2 := idx1 + s_2
				idx1_next := idx1 + 1
				idx2_next := idx2 + 1

				// Load r[idx] and r[idx+1] values
				// Each complex128 is 16 bytes (2 float64s)
				rPtr := (*[2]float64)(unsafe.Pointer(&r[idx1]))
				rPtrNext := (*[2]float64)(unsafe.Pointer(&r[idx1_next]))

				// Load the real and imaginary parts
				r1 := archsimd.LoadFloat64x2(rPtr)
				r2 := archsimd.LoadFloat64x2(rPtrNext)

				// Load r[idx2] and r[idx2+1] values
				r2Ptr := (*[2]float64)(unsafe.Pointer(&r[idx2]))
				r2PtrNext := (*[2]float64)(unsafe.Pointer(&r[idx2_next]))

				rIdx2_1 := archsimd.LoadFloat64x2(r2Ptr)
				rIdx2_2 := archsimd.LoadFloat64x2(r2PtrNext)

				// Load twiddle factors
				w1 := factors[blocks*j]
				w2 := factors[blocks*(j+1)]

				// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
				// For w_n = r[idx2] * factors[blocks*j]

				// Process first pair
				w_n_1 := complexMulSIMD(rIdx2_1, w1)
				w_n_2 := complexMulSIMD(rIdx2_2, w2)

				// Butterfly operations: t[idx] = r[idx] + w_n, t[idx2] = r[idx] - w_n
				t1_add := r1.Add(w_n_1)
				t1_sub := r1.Sub(w_n_1)
				t2_add := r2.Add(w_n_2)
				t2_sub := r2.Sub(w_n_2)

				// Store results
				tPtr1 := (*[2]float64)(unsafe.Pointer(&t[idx1]))
				tPtr2 := (*[2]float64)(unsafe.Pointer(&t[idx2]))
				tPtr1Next := (*[2]float64)(unsafe.Pointer(&t[idx1_next]))
				tPtr2Next := (*[2]float64)(unsafe.Pointer(&t[idx2_next]))

				t1_add.Store(tPtr1)
				t1_sub.Store(tPtr2)
				t2_add.Store(tPtr1Next)
				t2_sub.Store(tPtr2Next)
			}
		}

		// Scalar fallback for remaining elements
		for ; j < s_2; j++ {
			idx := j + nb
			idx2 := idx + s_2
			ridx := r[idx]
			w_n := r[idx2] * factors[blocks*j]
			t[idx] = ridx + w_n
			t[idx2] = ridx - w_n
		}
	}
}

// complexMulSIMD performs complex multiplication using SIMD
// Input: vec contains [real, imag] and w is the complex twiddle factor
// Computes: vec * w = (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
func complexMulSIMD(vec archsimd.Float64x2, w complex128) archsimd.Float64x2 {
	// Extract real and imaginary parts of w
	wReal := real(w)
	wImag := imag(w)

	// Create vectors for w's components
	wRealVec := archsimd.BroadcastFloat64x2(wReal) // [wReal, wReal]
	wImagVec := archsimd.BroadcastFloat64x2(wImag) // [wImag, wImag]

	// vec contains [a, b] where a is real, b is imaginary
	// Step 1: vec * [wReal, wReal] = [a*wReal, b*wReal]
	acTerm := vec.Mul(wRealVec)

	// Step 2: Permute vec to get [b, a]
	vecSwapped := vec.SelectFromPair(1, 0, vec)

	// Step 3: vecSwapped * [wImag, wImag] = [b*wImag, a*wImag]
	bdTerm := vecSwapped.Mul(wImagVec)

	// Step 4: Use ADDSUBPD to compute [ac - bd, bc + ad]
	// ADDSUBPD subtracts element 0 and adds element 1
	result := acTerm.AddSub(bdTerm)

	return result
}
