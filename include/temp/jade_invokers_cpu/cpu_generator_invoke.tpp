#include "temp/JadeInvokersCpu.tpp"

namespace bm {
// --- Generator ---
    template<typename T, typename Func>
    void cpu_generator_invoke(JadeReactor &jr, Func lambda) {
        if (jr.is_contiguous) {
            auto out = static_cast<T *>(jr.phys[0]);

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(jr, out, lambda)
#endif
////////////////////////////////////////////////####}

            for (uint64_t i = 0; i < jr.numel; ++i) {
                out[i] = lambda(i);
            }
            return;
        }

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, lambda, RE_MAX_DIMS)
#endif
////////////////////////////////////////////////####}
        {
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

                uint64_t off_out = 0;
                for (uint64_t d = 0; d < jr.ndims; ++d) {
                    off_out += foot_step[d] * jr.strides[0][d];
                }

                auto phys_out = static_cast<T *>(jr.phys[0]);

                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off_out] = lambda(i);

                    for (long long dim = jr.ndims - 1; dim >= 0; --dim) {
                        foot_step[dim]++;
                        off_out += jr.strides[0][dim];
                        if (foot_step[dim] < jr.shape[dim]) break;
                        foot_step[dim] = 0;
                        off_out -= jr.shape[dim] * jr.strides[0][dim];
                    }
                }
            }
        }
    }

}