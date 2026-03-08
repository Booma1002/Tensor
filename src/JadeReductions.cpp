#include "header/Jade.hpp"
#include "header/Dispatcher.hpp"
using namespace bm;

Jade Jade::std(const Jade& input) {
    Jade output(input.dtype, 0.0, 1ULL);
    Dispatcher::execute_reduction(OpCode::STD, output, input);
    return output;
}

Jade Jade::var(const Jade& input) {
    Jade output(input.dtype, 0.0, 1ULL);
    Dispatcher::execute_reduction(OpCode::VAR, output, input);
    return output;
}

Jade Jade::max(const Jade& input) {
    Jade output(input.dtype, 0.0, 1ULL);
    Dispatcher::execute_reduction(OpCode::MAX, output, input);
    return output;
}

Jade Jade::min(const Jade& input) {
    Jade output(input.dtype, 0.0, 1ULL);
    Dispatcher::execute_reduction(OpCode::MIN, output, input);
    return output;
}

Jade Jade::mean(const Jade& input) {
    Jade output(input.dtype, 0.0, 1ULL);
    Dispatcher::execute_reduction(OpCode::MEAN, output, input);
    return output;
}

Jade Jade::dot(const Jade& other) const {
    Jade output(this->dtype, 0.0, 1ULL);
    Dispatcher::execute_reduction_binary(OpCode::DOT, output, *this, other);
    return output;
}

Jade Jade::argmax(const Jade& input) {
    // Indexes are strictly 64-bit unsigned integers
    Jade output(DType::UINT64, 0.0, 1ULL);
    Dispatcher::execute_reduction(OpCode::ARGMAX, output, input);
    return output;
}

Jade Jade::argmin(const Jade& input) {
    Jade output(DType::UINT64, 0.0, 1ULL);
    Dispatcher::execute_reduction(OpCode::ARGMIN, output, input);
    return output;
}