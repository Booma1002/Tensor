#pragma once
#include <limits>

// ===========================================================
// ===========================================================
// =========={..........CPU Invoking..........}===============
// ===========================================================
// ===========================================================
#include "temp/jade_invokers_cpu/cpu_arg_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_elementwise_binary_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_elementwise_unary_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_elementwise_scalar_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_MatMul_binary_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_reduction_unary_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_reduction_binary_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_std_var_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_generator_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_max_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_min_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_mean_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_argmax_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_argmin_invoke.tpp"
#include "temp/jade_invokers_cpu/cpu_dot_invoke.tpp"
// ===========================================================
// ===========================================================