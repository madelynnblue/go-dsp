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

package spectral

import (
	"testing"

	"github.com/madelynnblue/go-dsp/dsputils"
)

type segmentTest struct {
	size,
	noverlap int
	out [][]float64
}

var segmentTests = []segmentTest{
	{
		4, 0,
		[][]float64{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		},
	},
	{
		4, 1,
		[][]float64{
			{1, 2, 3, 4},
			{4, 5, 6, 7},
			{7, 8, 9, 10},
		},
	},
	{
		4, 2,
		[][]float64{
			{1, 2, 3, 4},
			{3, 4, 5, 6},
			{5, 6, 7, 8},
			{7, 8, 9, 10},
		},
	},
}

func TestSegment(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	for _, v := range segmentTests {
		o := Segment(x, v.size, v.noverlap)
		if !dsputils.PrettyClose2F(o, v.out) {
			t.Error("Segment error\n  output:", o, "\nexpected:", v.out)
		}
	}
}
