/*
 * Copyright (c) 2012 Madelynn Blue <blue.mlynn@gmail.com>
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

// Package spectral provides spectral analysis functions for digital signal processing.
package spectral

// Segment x segmented into segments of length size with specified noverlap.
// Number of segments returned is (len(x) - size) / (size - noverlap) + 1.
func Segment(x []float64, size, noverlap int) [][]float64 {
	stride := size - noverlap
	lx := len(x)

	var segments int
	if lx == size {
		segments = 1
	} else if lx > size {
		segments = (len(x)-size)/stride + 1
	} else {
		segments = 0
	}

	r := make([][]float64, segments)
	for i, offset := 0, 0; i < segments; i++ {
		r[i] = make([]float64, size)

		for j := 0; j < size; j++ {
			r[i][j] = x[offset+j]
		}

		offset += stride
	}

	return r
}
