#include "temp/JadeInvokersCpu.tpp"

namespace bm {
// ==================================================================
    // =========={.......... STD / VAR REDUCTIONS ..........}============
    // ==================================================================
    template<typename T, bool IS_STD>
    void cpu_std_var_invoke(JadeReactor& jr) {
        double global_sum = 0.0;
        double global_sum_sq = 0.0;

        if (jr.is_contiguous) {
            auto in = static_cast<T*>(jr.phys[1]);
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, in, global_sum, global_sum_sq)
#endif
            {
                double local_sum = 0.0;
                double local_sum_sq = 0.0;
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
                for (uint64_t i = 0; i < jr.numel; ++i) {
                    auto val = static_cast<double>(in[i]);
                    local_sum += val;
                    local_sum_sq += (val * val);
                }
#if defined(_OPENMP)
#pragma omp critical
#endif
                {
                    global_sum += local_sum;
                    global_sum_sq += local_sum_sq;
                }
            }
        }
        else {
            auto in = static_cast<T*>(jr.phys[1]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, in, global_sum, global_sum_sq, RE_MAX_DIMS)
#endif
////////////////////////////////////////////////####}
            {

                double local_sum = 0.0;
                double local_sum_sq = 0.0;
                int thread = 0, num_threads = 1;
////////////////////////////////////////////////####{
#if defined(_OPENMP)
                thread = omp_get_thread_num();
                num_threads = omp_get_num_threads();
#endif
////////////////////////////////////////////////####}
                uint64_t chop = jr.numel / num_threads;
                uint64_t r = jr.numel % num_threads;
                uint64_t begin = thread * chop + std::min((uint64_t) thread, r);
                uint64_t piece = chop + (thread < r ? 1 : 0);
                uint64_t end = begin + piece;

                if (piece > 0) {
                    uint64_t foot_step[RE_MAX_DIMS] = {0};
                    get_cursor(begin, foot_step, jr.shape, jr.ndims);

                    uint64_t off = 0;
                    for (uint64_t d = 0; d < jr.ndims; ++d) {
                        off += foot_step[d] * jr.strides[1][d];
                    }

                    for (uint64_t i = begin; i < end; ++i) {
                        auto val = static_cast<double>(in[off]);
                        local_sum += val;
                        local_sum_sq += (val * val);

                        for (long long dim = jr.ndims - 1; dim >= 0; --dim) {
                            foot_step[dim]++;
                            off += jr.strides[1][dim];
                            if (foot_step[dim] < jr.shape[dim]) break;
                            foot_step[dim] = 0;
                            off -= jr.shape[dim] * jr.strides[1][dim];
                        }
                    }
                }
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp critical
#endif
////////////////////////////////////////////////####}
                {
                    global_sum += local_sum;
                    global_sum_sq += local_sum_sq;
                }
            }
        }

        auto n = static_cast<double>(jr.numel);
        double variance = (global_sum_sq - (global_sum * global_sum / n)) / (n - 1.0);
        auto out = static_cast<T*>(jr.phys[0]);

        if constexpr (IS_STD) {// compile time
            out[0] = static_cast<T>(std::sqrt(std::max(0.0, variance)));
        } else {
            out[0] = static_cast<T>(std::max(0.0, variance));
        }
    }

    template<typename T> void cpu_std_invoke(JadeReactor& jr) {
        cpu_std_var_invoke<T, true>(jr);
    }
    template<typename T> void cpu_var_invoke(JadeReactor& jr) {
        cpu_std_var_invoke<T, false>(jr);
    }
}