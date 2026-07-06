# fftconv — CDNA3 (gfx942): correctness-valid

Port of QuixiCore-CUDA/kernels/fftconv (ThunderKittens). Reference is
y = ifft(fft(u)*fft(k)).real[:L]; by the convolution theorem this is the circular
convolution y[n]=Re(sum_m u[m]*k[(n-m) mod N]), computed directly here
(mathematically identical to the FFT path). Complex inputs, real output.
Validated vs an fp64 oracle: PASS (essentially exact). `make test`.
Perf follow-up: a Cooley-Tukey / rocFFT O(N log N) path.
