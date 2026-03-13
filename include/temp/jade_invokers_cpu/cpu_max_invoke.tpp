#include "temp/JadeInvokersCpu.tpp"

namespace bm {
// --- MAX ---
    template<typename T>
    void cpu_max_invoke(JadeReactor& jr) {
        bool f = true;
        jr.args[0] = const_cast<void*>(static_cast<const void*>(&f));
        cpu_reduction_unary_invoke<T>(jr, std::numeric_limits<T>::lowest(),
                                      [](T acc, T val) { return std::max(acc, val); });
    }
}