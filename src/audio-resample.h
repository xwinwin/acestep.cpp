#pragma once
// audio-resample.h: sample rate conversion via windowed sinc interpolation.
// Kaiser window, configurable quality. No external dependencies.
// Part of acestep.cpp. MIT license.

#include <cmath>
#include <cstdlib>
#include <cstring>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

// Modified Bessel function I0 (first kind, zeroth order).
// Used by the Kaiser window. Series expansion, converges fast.
static double audio_resample_bessel_i0(double x) {
    double sum  = 1.0;
    double term = 1.0;
    double y    = x * x * 0.25;
    for (int k = 1; k < 30; k++) {
        term *= y / ((double) k * (double) k);
        sum += term;
        if (term < sum * 1e-15) {
            break;
        }
    }
    return sum;
}

// Resample a planar float audio buffer from sr_in to sr_out.
//
// in:     planar float [ch0: n_in samples][ch1: n_in samples][...]
// n_in:   samples per channel in the input
// sr_in:  input sample rate (e.g. 44100)
// sr_out: output sample rate (e.g. 48000)
// nch:    number of channels (1 or 2)
// n_out:  receives the output sample count per channel
//
// Returns a malloc'd planar buffer [ch0: n_out][ch1: n_out][...].
// Caller must free() the result.
// Returns NULL on error (bad sr, alloc failure).
static float * audio_resample(const float * in, int n_in, int sr_in, int sr_out, int nch, int * n_out) {
    if (!in || n_in <= 0 || sr_in <= 0 || sr_out <= 0 || nch <= 0) {
        *n_out = 0;
        return NULL;
    }

    // passthrough: no conversion needed
    if (sr_in == sr_out) {
        *n_out      = n_in;
        size_t  sz  = (size_t) n_in * (size_t) nch * sizeof(float);
        float * out = (float *) malloc(sz);
        if (out) {
            memcpy(out, in, sz);
        }
        return out;
    }

    double ratio = (double) sr_out / (double) sr_in;
    *n_out       = (int) ((double) n_in * ratio);
    if (*n_out <= 0) {
        *n_out = 0;
        return NULL;
    }

    float * out = (float *) malloc((size_t) (*n_out) * (size_t) nch * sizeof(float));
    if (!out) {
        *n_out = 0;
        return NULL;
    }

    // filter half length in input samples.
    // 32 taps (64 total) for high quality music resampling.
    int half_len = 32;

    // Kaiser window parameter (beta=9.0 gives ~80 dB stopband)
    double beta    = 9.0;
    double inv_i0b = 1.0 / audio_resample_bessel_i0(beta);

    // cutoff: lowpass at the lower of the two rates to prevent aliasing
    double fc = 0.5 * ((ratio < 1.0) ? ratio : 1.0);

    for (int ch = 0; ch < nch; ch++) {
        const float * src = in + ch * n_in;
        float *       dst = out + ch * (*n_out);

        for (int i = 0; i < *n_out; i++) {
            // position in input sample space
            double center = (double) i / ratio;
            int    start  = (int) floor(center) - half_len + 1;
            int    end    = (int) floor(center) + half_len;

            double sum = 0.0;
            double wgt = 0.0;

            for (int j = start; j <= end; j++) {
                double d = center - (double) j;

                // windowed sinc
                double sinc_val;
                if (fabs(d) < 1e-9) {
                    sinc_val = 2.0 * fc;
                } else {
                    sinc_val = sin(2.0 * M_PI * fc * d) / (M_PI * d);
                }

                // Kaiser window
                double t = d / (double) half_len;
                double win;
                if (t < -1.0 || t > 1.0) {
                    win = 0.0;
                } else {
                    win = audio_resample_bessel_i0(beta * sqrt(1.0 - t * t)) * inv_i0b;
                }

                double h = sinc_val * win;

                // clamp to input bounds (repeat edge samples)
                int idx = j;
                if (idx < 0) {
                    idx = 0;
                }
                if (idx >= n_in) {
                    idx = n_in - 1;
                }

                sum += (double) src[idx] * h;
                wgt += h;
            }

            // normalize to compensate for edge effects
            dst[i] = (wgt > 1e-12) ? (float) (sum / wgt) : 0.0f;
        }
    }

    return out;
}
