#include "temp/JadeInvokersCpu.tpp"

namespace bm {
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

}