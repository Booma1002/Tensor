#include "temp/JadeInvokersCpu.tpp"

namespace bm {
// --- Binary Reduction ---
    template<typename T, typename Func>
    void cpu_reduction_binary_invoke(JadeReactor &jr, T init_val, Func lambda) {
        T global_acc = init_val;

        if (jr.is_contiguous) {
            auto a = static_cast<T *>(jr.phys[1]);
            auto b = static_cast<T *>(jr.phys[2]);

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, a, b, global_acc, init_val, lambda)
#endif
////////////////////////////////////////////////####}
            {
                T local_acc = init_val;
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
////////////////////////////////////////////////####}
                for (uint64_t i = 0; i < jr.numel; ++i) {
                    local_acc = lambda(local_acc, a[i], b[i]);
                }
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp critical
#endif
////////////////////////////////////////////////####}
                { global_acc += local_acc; }
            }
        }
        else {
            auto a = static_cast<T *>(jr.phys[1]);
            auto b = static_cast<T *>(jr.phys[2]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, a, b, global_acc, init_val, lambda, RE_MAX_DIMS)
#endif
////////////////////////////////////////////////####}
            {
                T local_acc = init_val;
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
                        local_acc = lambda(local_acc, a[off], b[off]);

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
                { global_acc += local_acc; }
            }


        }
        auto out = static_cast<T *>(jr.phys[0]);
        out[0] = global_acc;
    }
}