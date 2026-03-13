#include "temp/JadeInvokersCpu.tpp"

namespace bm {
// ==================================================================
// =========={.......... ARGMAX / ARGMIN REDUCTIONS ..........}======
// ==================================================================
    template<typename T>
    struct ArgAcc {
        T val;
        uint64_t idx;
    };

    template<typename T, bool MAX_MODE>
    void cpu_arg_invoke(JadeReactor &jr) {
        T init_limit = MAX_MODE ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
        ArgAcc<T> global_acc = {init_limit, 0};

        if (jr.is_contiguous) {
            auto in = static_cast<T *>(jr.phys[1]);
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, in, global_acc, init_limit)
#endif
            {
                ArgAcc<T> local_acc = {init_limit, 0};
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
                for (uint64_t i = 0; i < jr.numel; ++i) {
                    if constexpr (MAX_MODE) {
                        if (in[i] > local_acc.val) {
                            local_acc.val = in[i];
                            local_acc.idx = i;
                        }
                    } else {
                        if (in[i] < local_acc.val) {
                            local_acc.val = in[i];
                            local_acc.idx = i;
                        }
                    }
                }
#if defined(_OPENMP)
#pragma omp critical
#endif
                {
                    if constexpr (MAX_MODE) {
                        if (local_acc.val > global_acc.val) global_acc = local_acc;
                    } else {
                        if (local_acc.val < global_acc.val) global_acc = local_acc;
                    }
                }
            }
        } else {
            auto in = static_cast<T *>(jr.phys[1]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, in, global_acc, RE_MAX_DIMS, init_limit)
#endif
////////////////////////////////////////////////####}
            {
                ArgAcc<T> local_acc;
                if constexpr (MAX_MODE)
                    local_acc = {std::numeric_limits<T>::lowest(), 0};
                else local_acc = {std::numeric_limits<T>::max(), 0};
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
                        if constexpr (MAX_MODE) {//compile time
                            if (in[off] > local_acc.val) { //running time
                                local_acc.val = in[off];
                                local_acc.idx = i;
                            }
                        } else {
                            if (in[off] < local_acc.val) {
                                local_acc.val = in[off];
                                local_acc.idx = i;
                            }
                        }

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
                    if constexpr (MAX_MODE) {
                        if (local_acc.val > global_acc.val) global_acc = local_acc;
                    } else {
                        if (local_acc.val < global_acc.val) global_acc = local_acc;
                    }
                }
            }
        }

        auto out = static_cast<uint64_t *>(jr.phys[0]);
        out[0] = global_acc.idx;
    }
}