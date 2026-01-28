# go-dsp

[![Go Reference](https://pkg.go.dev/badge/github.com/madelynnblue/go-dsp.svg)](https://pkg.go.dev/github.com/madelynnblue/go-dsp)

Digital signal processing packages.

* **[dsputils](http://godoc.org/github.com/madelynnblue/go-dsp/dsputils)** - utilities and data structures for DSP
* **[fft](http://godoc.org/github.com/madelynnblue/go-dsp/fft)** - fast Fourier transform
* **[spectral](http://godoc.org/github.com/madelynnblue/go-dsp/spectral)** - power spectral density functions (e.g., Pwelch)
* **[wav](http://godoc.org/github.com/madelynnblue/go-dsp/wav)** - wav file reader functions
* **[window](http://godoc.org/github.com/madelynnblue/go-dsp/window)** - window functions (e.g., Hamming, Hann, Bartlett)

## Installation and Usage

```$ go get github.com/madelynnblue/go-dsp/fft```

```
package main

import (
        "fmt"
        
        "github.com/madelynnblue/go-dsp/fft"
)

func main() {
        fmt.Println(fft.FFTReal([]float64 {1, 2, 3}))
}
```
