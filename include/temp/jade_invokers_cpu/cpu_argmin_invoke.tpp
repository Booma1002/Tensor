#include "temp/JadeInvokersCpu.tpp"

namespace bm {
    template<typename T> void cpu_argmin_invoke(JadeReactor& jr) {
        cpu_arg_invoke<T, false>(jr);
    }

}