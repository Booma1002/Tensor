#pragma once
#include "header/Jade.hpp"
#include "header/Dispatcher.hpp"
namespace bm {
Jade Jade::arange(DType dtype, Slice range) {
    long long start = range.start;
    long long stop = range.stop;
    long long step = range.step;

    if (step == 0) {
        std::string msg = "[JadeFactory] Arange step cannot be zero.";
        LOG_ERR(msg);
        throw std::invalid_argument(msg);
    }

    // 1. Calculate length mathematically without wrapping negative values
    uint64_t len = 0;
    if (step > 0 && stop > start) {
        len = (stop - start + step - 1) / step; // Ceiling division
    } else if (step < 0 && stop < start) {
        len = (start - stop - step - 1) / -step; // Ceiling division for negative steps
    }

    // 2. Allocate the empty slab
    uint64_t shape[] = {len};
    Jade output(dtype, 0.0, shape, 1);

    if (len == 0) return output; // Return empty tensor if range is invalid

    // 3. Ignite the Reactor
    // We pass 'output' as both the out and the dummy input to reuse the unary dispatcher.
    // We cast start and step to doubles so the void* array captures them perfectly.
    double arg_start = static_cast<double>(start);
    double arg_step  = static_cast<double>(step);

    Dispatcher::execute_unary(OpCode::ARANGE, output, output, arg_start, arg_step);

    return output;
}
}