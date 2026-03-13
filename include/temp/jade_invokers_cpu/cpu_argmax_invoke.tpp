#include "temp/JadeInvokersCpu.tpp"

namespace bm {
    template<typename T> void cpu_argmax_invoke(JadeReactor& jr) {
        cpu_arg_invoke<T, true>(jr);
    }
}