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

// Package wav provides support for the WAV file format.
//
// Supported formats are PCM 8- and 16-bit, and IEEE float. Extended chunks
// (JUNK, bext, and others added by tools like ProTools) are ignored.
package wav

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"time"
)

const (
	wavFormatPCM       = 1
	wavFormatIEEEFloat = 3
)

// Header contains Wav fmt chunk data.
type Header struct {
	AudioFormat   uint16
	NumChannels   uint16
	SampleRate    uint32
	ByteRate      uint32
	BlockAlign    uint16
	BitsPerSample uint16
}

// Wav reads wav files.
type Wav struct {
	Header
	// Samples is the total number of available samples.
	Samples int
	// Duration is the estimated duration based on reported samples.
	Duration time.Duration

	r io.Reader
}

// New reads the WAV header from r.
func New(r io.Reader) (*Wav, error) {
	var w Wav
	header := make([]byte, 16)
	if _, err := io.ReadFull(r, header[:12]); err != nil {
		return nil, err
	}
	if string(header[0:4]) != "RIFF" {
		return nil, fmt.Errorf("wav: missing RIFF")
	}
	if string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("wav: missing WAVE")
	}
	hasFmt := false
	for {
		if _, err := io.ReadFull(r, header[:8]); err != nil {
			return nil, err
		}
		sz := binary.LittleEndian.Uint32(header[4:])
		switch typ := string(header[:4]); typ {
		case "fmt ":
			if sz < 16 {
				return nil, fmt.Errorf("wav: bad fmt size")
			}
			f := make([]byte, sz)
			if _, err := io.ReadFull(r, f); err != nil {
				return nil, err
			}
			if err := binary.Read(bytes.NewBuffer(f), binary.LittleEndian, &w.Header); err != nil {
				return nil, err
			}
			switch w.AudioFormat {
			case wavFormatPCM:
			case wavFormatIEEEFloat:
			default:
				return nil, fmt.Errorf("wav: unknown audio format: %02x", w.AudioFormat)
			}
			hasFmt = true
		case "data":
			if !hasFmt {
				return nil, fmt.Errorf("wav: unexpected fmt chunk")
			}
			w.Samples = int(sz) / int(w.BitsPerSample) * 8
			w.Duration = time.Duration(w.Samples) * time.Second / time.Duration(w.SampleRate) / time.Duration(w.NumChannels)
			w.r = io.LimitReader(r, int64(sz))
			return &w, nil
		default:
			io.CopyN(ioutil.Discard, r, int64(sz))
		}
	}
}

// ReadSamples returns a [n]T, where T is uint8, int16, or float32, based on the
// wav data. n is the number of samples to return.
func (w *Wav) ReadSamples(n int) (interface{}, error) {
	var data interface{}
	switch w.AudioFormat {
	case wavFormatPCM:
		switch w.BitsPerSample {
		case 8:
			data = make([]uint8, n)
		case 16:
			data = make([]int16, n)
		default:
			return nil, fmt.Errorf("wav: unknown bits per sample: %v", w.BitsPerSample)
		}
	case wavFormatIEEEFloat:
		data = make([]float32, n)
	default:
		return nil, fmt.Errorf("wav: unknown audio format")
	}
	if err := binary.Read(w.r, binary.LittleEndian, data); err != nil {
		return nil, err
	}
	return data, nil
}

// ReadFloats is like ReadSamples, but it converts any underlying data to a
// float32.
func (w *Wav) ReadFloats(n int) ([]float32, error) {
	d, err := w.ReadSamples(n)
	if err != nil {
		return nil, err
	}
	var f []float32
	switch d := d.(type) {
	case []uint8:
		f = make([]float32, len(d))
		for i, v := range d {
			f[i] = float32(v) / math.MaxUint8
		}
	case []int16:
		f = make([]float32, len(d))
		for i, v := range d {
			f[i] = (float32(v) - math.MinInt16) / (math.MaxInt16 - math.MinInt16)
		}
	case []float32:
		f = d
	default:
		return nil, fmt.Errorf("wav: unknown type: %T", d)
	}
	return f, nil
}
