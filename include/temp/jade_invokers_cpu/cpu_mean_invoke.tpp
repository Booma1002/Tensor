#include "temp/JadeInvokersCpu.tpp"

namespace bm {
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
}