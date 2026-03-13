#include "temp/JadeInvokersCpu.tpp"

namespace bm {
    // --- DOT ---
    template<typename T>
    void cpu_dot_invoke(JadeReactor& jr, double val) {
        cpu_elementwise_scalar_invoke<T>(jr, [&val](T a) { return static_cast<T>(val * a);});
    }
}