#include "temp/JadeInvokersCpu.tpp"

namespace bm {
// --- Elementwise Unary ---
    template<typename T, typename Func>
    void cpu_elementwise_unary_invoke(JadeReactor &jr, Func lambda) {
        std::string msg = std::format("[CPU Invoker] Performing Contiguous Jade Unary Reaction. Reactor Ndims={}{}",
                                      std::to_string(jr.ndims), ".");
        LOG_INFO(msg);
        if (jr.is_contiguous) {
            auto out = static_cast<T *>(jr.phys[0]);
            auto in = static_cast<T *>(jr.phys[1]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(jr, out, in, lambda)
#endif
////////////////////////////////////////////////####}
            for (uint64_t i = 0; i < jr.numel; ++i) {
                out[i] = lambda(in[i]);
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

                uint64_t off[2] = {0, 0};
                for (uint64_t d = 0; d < jr.ndims; ++d) {
                    off[0] += foot_step[d] * jr.strides[0][d];
                    off[1] += foot_step[d] * jr.strides[1][d];
                }

                auto phys_out = static_cast<T *>(jr.phys[0]);
                auto phys_in = static_cast<T *>(jr.phys[1]);


                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off[0]] = lambda(phys_in[off[1]]);
                    for (long long dim = jr.ndims - 1; dim >= 0; --dim) {
                        foot_step[dim]++;
                        off[0] += jr.strides[0][dim];
                        off[1] += jr.strides[1][dim];
                        if (foot_step[dim] < jr.shape[dim]) break;
                        foot_step[dim] = 0;
                        off[0] -= jr.shape[dim] * jr.strides[0][dim];
                        off[1] -= jr.shape[dim] * jr.strides[1][dim];
                    }
                }

            }
        }
    }
}