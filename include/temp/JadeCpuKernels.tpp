#pragma once
#include <limits>
namespace bm {
// ==================================================================
// =========={..........CPU K-nary Invoking..........}===============
// ==================================================================
    ;

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

    // --- Elementwise Scalar ---
    template<typename T, typename Func>
    void cpu_elementwise_scalar_invoke(JadeReactor &jr, Func lambda) {
        if (jr.is_contiguous) {
            auto out = static_cast<T *>(jr.phys[0]);

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(jr, out, lambda)
#endif
////////////////////////////////////////////////####}
            for (uint64_t i = 0; i < jr.numel; ++i) {
                out[i] = lambda(out[i]);
            }
            return;
        }
        double val = *static_cast<double*>(jr.args[0]);

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, lambda, RE_MAX_DIMS, val)
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

                uint64_t off_out = 0;
                for (uint64_t d = 0; d < jr.ndims; ++d) {
                    off_out += foot_step[d] * jr.strides[0][d];
                }

                auto phys_out = static_cast<T *>(jr.phys[0]);

                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off_out] = lambda(val);

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

    // --- MatMul Binary ---
    template<typename T>
    void cpu_MatMul_binary_invoke(JadeReactor &jr) {
        std::string msg = std::format("[CPU Invoker] Performing Contiguous Jade MatMul Reaction. Reactor Ndims {}{}",
                                      std::to_string(jr.ndims), ".");
        LOG_INFO(msg);
        uint64_t M = jr.shape[jr.ndims - 2];
        uint64_t N = jr.shape[jr.ndims - 1];
        uint64_t K = jr.inner_k;

        uint64_t strOut_m = jr.strides[0][jr.ndims - 2];
        uint64_t strOut_n = jr.strides[0][jr.ndims - 1];
        uint64_t strA_m = jr.strides[1][jr.ndims - 2];
        uint64_t strA_k = jr.strides[1][jr.ndims - 1];
        uint64_t strB_k = jr.strides[2][jr.ndims - 2];
        uint64_t strB_n = jr.strides[2][jr.ndims - 1];

        long long B_ndim = jr.ndims - 2;
        uint64_t BATCH = 1;
        for (int i = 0; i < B_ndim; ++i) BATCH *= jr.shape[i];

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) \
        shared(jr, M, N, K, B_ndim, BATCH, RE_MAX_DIMS, \
               strOut_m, strOut_n, strA_m, strA_k, strB_k, strB_n)
#endif
////////////////////////////////////////////////####}
        for (uint64_t b = 0; b < BATCH; ++b) {
            uint64_t foot_step[RE_MAX_DIMS] = {0};
            get_cursor(b, foot_step, jr.shape, B_ndim);

            uint64_t off_out = 0, off_a = 0, off_b = 0;
            for (int i = 0; i < B_ndim; ++i) {
                off_out += foot_step[i] * jr.strides[0][i];
                off_a += foot_step[i] * jr.strides[1][i];
                off_b += foot_step[i] * jr.strides[2][i];
            }

            auto OUT = static_cast<T *>(jr.phys[0]) + off_out;
            auto A = static_cast<T *>(jr.phys[1]) + off_a;
            auto B = static_cast<T *>(jr.phys[2]) + off_b;

            for (uint64_t i = 0; i < M; ++i)
                for (uint64_t j = 0; j < N; ++j) OUT[i * strOut_m + j * strOut_n] = 0.0f;

            for (uint64_t i = 0; i < M; ++i) {
                for (uint64_t k = 0; k < K; ++k) {
                    double valA = A[i * strA_m + k * strA_k];
                    for (uint64_t j = 0; j < N; ++j) {
                        OUT[i * strOut_m + j * strOut_n] += valA * B[k * strB_k + j * strB_n];
                    }
                }
            }
        }
    }
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


    template<typename T, typename Func>
    void cpu_reduction_unary_invoke(JadeReactor &jr, T init_val, Func lambda) {
        T global_acc = init_val;

        if (jr.is_contiguous) {
            auto in = static_cast<T *>(jr.phys[1]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, in, global_acc, init_val, lambda)
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
                    local_acc = lambda(local_acc, in[i]);
                }
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp critical
#endif
////////////////////////////////////////////////####}
                { global_acc = lambda(global_acc, local_acc); }

            }
        }
        else {
            auto in = static_cast<T *>(jr.phys[1]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, in, global_acc, init_val, lambda, RE_MAX_DIMS)
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
                        local_acc = lambda(local_acc, in[off]);

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
                {global_acc = lambda(global_acc, local_acc);}
            }

        }
        auto out = static_cast<T *>(jr.phys[0]);
        out[0] = global_acc;
    }

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
    // --- MAX ---
    template<typename T>
    void cpu_max_invoke(JadeReactor& jr) {
        bool f = true;
        jr.args[0] = const_cast<void*>(static_cast<const void*>(&f));
        cpu_reduction_unary_invoke<T>(jr, std::numeric_limits<T>::lowest(),
                                      [](T acc, T val) { return std::max(acc, val); });
    }
    // --- MIN ---
    template<typename T>
    void cpu_min_invoke(JadeReactor& jr) {
        bool f = false;
        jr.args[0] = const_cast<void*>(static_cast<const void*>(&f));
        cpu_reduction_unary_invoke<T>(jr, std::numeric_limits<T>::max(),
                                      [](T acc, T val) { return std::min(acc, val); });
    }

    // --- MEAN ---
    template<typename T>
    void cpu_mean_invoke(JadeReactor& jr) {
        // process: sum everything starting from 0
        cpu_reduction_unary_invoke<T>(jr, static_cast<T>(0),
                                      [](T acc, T val) { return acc + val; });

        // postprocessing: divide the single output value by N
        auto out = static_cast<T*>(jr.phys[0]);
        out[0] /= static_cast<T>(jr.numel);
    }

    // --- DOT ---
    template<typename T>
    void cpu_dot_invoke(JadeReactor& jr) {
        cpu_reduction_binary_invoke<T>(jr, static_cast<T>(0),
                                       [](T acc, T a, T b) { return acc + (a * b); });
    }

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

    template<typename T> void cpu_std_invoke(JadeReactor& jr) { cpu_std_var_invoke<T, true>(jr); }
    template<typename T> void cpu_var_invoke(JadeReactor& jr) { cpu_std_var_invoke<T, false>(jr); }

    // ==================================================================
    // =========={.......... ARGMAX / ARGMIN REDUCTIONS ..........}======
    // ==================================================================
    template<typename T>
    struct ArgAcc {
        T val;
        uint64_t idx;
    };

    template<typename T, bool MAX_MODE>
    void cpu_arg_invoke(JadeReactor& jr) {
        T init_limit = MAX_MODE ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
        ArgAcc<T> global_acc = {init_limit, 0};

        if (jr.is_contiguous) {
            auto in = static_cast<T*>(jr.phys[1]);
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
                        if (in[i] > local_acc.val) { local_acc.val = in[i]; local_acc.idx = i; }
                    } else {
                        if (in[i] < local_acc.val) { local_acc.val = in[i]; local_acc.idx = i; }
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
        }
        else {
            auto in = static_cast<T*>(jr.phys[1]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(jr, in, global_acc, RE_MAX_DIMS, init_limit)
#endif
////////////////////////////////////////////////####}
            {
                ArgAcc<T> local_acc;
                if constexpr (MAX_MODE)
                    local_acc = {std::numeric_limits<T>::lowest(), 0};
                else local_acc = {std::numeric_limits<T>::max(),0};
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
                        if constexpr (MAX_MODE){//compile time
                                if (in[off] > local_acc.val) { //running time
                                    local_acc.val = in[off];
                                    local_acc.idx = i;
                                }
                        }else {
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

        auto out = static_cast<uint64_t*>(jr.phys[0]);
        out[0] = global_acc.idx;
    }

    template<typename T> void cpu_argmax_invoke(JadeReactor& jr) { cpu_arg_invoke<T, true>(jr); }
    template<typename T> void cpu_argmin_invoke(JadeReactor& jr) { cpu_arg_invoke<T, false>(jr); }



}