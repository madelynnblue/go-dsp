package wav

import (
	"bytes"
	"os"
	"reflect"
	"testing"
)

func checkHeader(b []byte) error {
	_, err := New(bytes.NewBuffer(b))
	return err
}

func TestShortHeaderValidation(t *testing.T) {
	shortHeader := []byte{0x52, 0x49, 0x46, 0x46, 0x72, 0x8C, 0x34, 0x00, 0x57, 0x41, 0x56, 0x45}
	if err := checkHeader(shortHeader); err == nil {
		t.Fatal("Expected short header to fail validation, but validation passed")
	}
	if err := checkHeader(nil); err == nil {
		t.Fatal("Expected nil header to fail validation, but validation passed")
	}
}

func TestInvalidHeaderValidation(t *testing.T) {
	validateErrorForMissingHeaderValue := func(err error, value string) {
		if err == nil {
			t.Fatalf("Invalid header data missing '%s' should fail validation", value)
		}
	}

	// 44 empty bytes
	invalidHeader := make([]byte, 44)
	err := checkHeader(invalidHeader)
	validateErrorForMissingHeaderValue(err, "RIFF")

	// RIFF and 40 empty bytes
	_ = copy(invalidHeader[:4], []byte{0x52, 0x49, 0x46, 0x46})
	err = checkHeader(invalidHeader)
	validateErrorForMissingHeaderValue(err, "WAVE")

	// RIFF, WAVE, and 36 empty bytes
	_ = copy(invalidHeader[8:12], []byte{0x57, 0x41, 0x56, 0x45})
	err = checkHeader(invalidHeader)
	validateErrorForMissingHeaderValue(err, "fmt")

	// RIFF, WAVE, fmt, and 32 empty bytes
	_ = copy(invalidHeader[12:16], []byte{0x66, 0x6D, 0x74, 0x20})
	err = checkHeader(invalidHeader)
	validateErrorForMissingHeaderValue(err, "data")
}

type wavTest struct {
	w   Wav
	typ reflect.Type
}

func TestWav(t *testing.T) {
	wavTests := map[string]wavTest{
		"small.wav": {
			w: Wav{
				Header: Header{
					AudioFormat:   1,
					NumChannels:   1,
					SampleRate:    44100,
					ByteRate:      88200,
					BlockAlign:    2,
					BitsPerSample: 16,
				},
				Samples:  41888,
				Duration: 949841269,
			},
			typ: reflect.TypeOf(make([]uint8, 0)),
		},
		"float.wav": {
			w: Wav{
				Header: Header{
					AudioFormat:   3,
					NumChannels:   1,
					SampleRate:    44100,
					ByteRate:      176400,
					BlockAlign:    4,
					BitsPerSample: 32,
				},
				Samples:  1889280 / 4,
				Duration: 10710204081,
			},
			typ: reflect.TypeOf(make([]float32, 0)),
		},
	}
	for name, wt := range wavTests {
		f, err := os.Open(name)
		if err != nil {
			t.Fatal(err)
		}
		w, err := New(f)
		if err != nil {
			t.Fatalf("%v: %v", name, err)
		}
		if !eq(*w, wt.w) {
			t.Errorf("wavs not equal: %v\ngot: %v\nexpected: %v", name, w, &wt.w)
		}
		err = f.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
}

func eq(x, y Wav) bool {
	x.r, y.r = nil, nil
	return x == y
}
