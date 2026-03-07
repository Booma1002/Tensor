#pragma once
#include <cmath>
#include <omp.h>
#include "Registry.hpp"
#include "JadeReactor.hpp"
namespace bm {

// ======================================================
// =========={..........Helpers..........}===============
// ======================================================
    ;

    constexpr void get_cursor(uint64_t linear_idx, uint64_t *cursor, const uint64_t *shape, uint64_t ndims);

// ==================================================================
// =========={..........CPU K-nary Invoking..........}===============
// ==================================================================
    ;

    template<typename Func>
    void cpu_elementwise_unary_invoke(JadeReactor &jr, Func kernel);

    template<typename Func>
    void cpu_elementwise_scalar_invoke(JadeReactor &react, Func lambda);

    template<typename Func>
    void cpu_elementwise_binary_invoke(JadeReactor &react, Func lambda);

    void cpu_MatMul_binary_invoke(JadeReactor &react);



// ======================================================
// =========={..........Kernels..........}===============
// ======================================================
    ;

    void cpu_add_kernel(JadeReactor &jr);

    void cpu_sub_kernel(JadeReactor &jr);

    void cpu_matmul_kernel(JadeReactor &jr);

    void cpu_copy_kernel(JadeReactor &jr);

    void cpu_fill_kernel(JadeReactor &jr);

    void cpu_mul_kernel(JadeReactor &jr);

    void cpu_mul_kernel(JadeReactor &jr);

    void cpu_sin_kernel(JadeReactor &jr);

    void cpu_cos_kernel(JadeReactor &jr);

    void cpu_tan_kernel(JadeReactor &jr);

    void cpu_exp_kernel(JadeReactor &jr);

    void cpu_log_kernel(JadeReactor &jr);

    void cpu_clip_kernel(JadeReactor &jr);

    void cpu_arange_kernel(JadeReactor& reactor);

// ===========================================================
// =========={..........Registration..........}===============
// ===========================================================
    ;

    REGISTER_KERNEL(ADD, CPU, cpu_add_kernel);
    REGISTER_KERNEL(SUB, CPU, cpu_sub_kernel);
    REGISTER_KERNEL(MATMUL, CPU, cpu_matmul_kernel);
    REGISTER_KERNEL(COPY, CPU, cpu_copy_kernel);
    REGISTER_KERNEL(FILL, CPU, cpu_fill_kernel);
    REGISTER_KERNEL(MUL, CPU, cpu_mul_kernel);
    REGISTER_KERNEL(SIN, CPU, cpu_sin_kernel);
    REGISTER_KERNEL(COS, CPU, cpu_cos_kernel);
    REGISTER_KERNEL(TAN, CPU, cpu_tan_kernel);
    REGISTER_KERNEL(EXP, CPU, cpu_exp_kernel);
    REGISTER_KERNEL(LOG, CPU, cpu_log_kernel);
    REGISTER_KERNEL(CLIP, CPU, cpu_clip_kernel);
    REGISTER_KERNEL(ARANGE, CPU, cpu_arange_kernel);

}

#include "temp/JadeCpuKernels.tpp"