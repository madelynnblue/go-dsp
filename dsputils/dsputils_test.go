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

package dsputils

import (
	"testing"
)

type segmentTest struct {
	segs     int
	noverlap float64
	slices   [][]int
}

var segmentTests = []segmentTest{
	{
		3,
		.5,
		[][]int{
			{0, 8},
			{4, 12},
			{8, 16},
		},
	},
}

func TestSegment(t *testing.T) {
	x := make([]complex128, 16)
	for n := range x {
		x[n] = complex(float64(n), 0)
	}

	for _, st := range segmentTests {
		v := Segment(x, st.segs, st.noverlap)
		s := make([][]complex128, st.segs)
		for i, sl := range st.slices {
			s[i] = x[sl[0]:sl[1]]
		}

		if !PrettyClose2(v, s) {
			t.Error("Segment error: expected:", s, ", output:", v)
		}
	}
}
