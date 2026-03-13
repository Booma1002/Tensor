#include "temp/JadeInvokersCpu.tpp"

namespace bm {
// --- MIN ---
    template<typename T>
    void cpu_min_invoke(JadeReactor& jr) {
        bool f = false;
        jr.args[0] = const_cast<void*>(static_cast<const void*>(&f));
        cpu_reduction_unary_invoke<T>(jr, std::numeric_limits<T>::max(),
                                      [](T acc, T val) { return std::min(acc, val); });
    }
}