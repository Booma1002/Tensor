#include "temp/JadeInvokersCpu.tpp"

namespace bm {
    // --- Elementwise Binary ---
    template<typename T, typename Func>
    void cpu_elementwise_binary_invoke(JadeReactor &jr, Func lambda) {
        std::string msg = std::format("[CPU Invoker] Performing Contiguous Jade Binary Reaction. Reactor Ndims={}.",
                                      std::to_string(jr.ndims));
        LOG_INFO(msg);
        if (jr.is_contiguous) {
            auto OUT = static_cast<T *>(jr.phys[0]);
            auto A = static_cast<T *>(jr.phys[1]);
            auto B = static_cast<T *>(jr.phys[2]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(jr, OUT, A, B, lambda)
#endif
////////////////////////////////////////////////####}
            for (uint64_t i = 0; i < jr.numel; ++i) {
                OUT[i] = lambda(A[i], B[i]);
            }
            return;
        }

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, lambda, RE_MAX_DIMS)
#endif
////////////////////////////////////////////////####}
        {
            int thread = 0;
            int num_threads = 1;
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

                uint64_t off[3] = {0, 0, 0};
                for (uint64_t d = 0; d < jr.ndims; ++d) {
                    off[0] += foot_step[d] * jr.strides[0][d];
                    off[1] += foot_step[d] * jr.strides[1][d];
                    off[2] += foot_step[d] * jr.strides[2][d];
                }

                auto phys_out = static_cast<T *>(jr.phys[0]);
                auto phys_a = static_cast<T *>(jr.phys[1]);
                auto phys_b = static_cast<T *>(jr.phys[2]);

                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off[0]] = lambda(phys_a[off[1]], phys_b[off[2]]);

                    for (long long dim = jr.ndims - 1; dim >= 0; --dim) {
                        foot_step[dim]++;
                        off[0] += jr.strides[0][dim];
                        off[1] += jr.strides[1][dim];
                        off[2] += jr.strides[2][dim];
                        if (foot_step[dim] < jr.shape[dim]) break;
                        foot_step[dim] = 0;
                        off[0] -= jr.shape[dim] * jr.strides[0][dim];
                        off[1] -= jr.shape[dim] * jr.strides[1][dim];
                        off[2] -= jr.shape[dim] * jr.strides[2][dim];
                    }
                }
            }
        }
    }
}