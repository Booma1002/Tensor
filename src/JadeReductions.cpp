#include "header/Jade.hpp"
#include "header/Dispatcher.hpp"
using namespace bm;

Jade Jade::std(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction_unary(OpCode::STD, output, input);
    else Dispatcher::execute_reduction_unary(OpCode::STD, output, input[axes]);
    return output;
}

Jade Jade::var(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction_unary(OpCode::VAR, output, input);
    else Dispatcher::execute_reduction_unary(OpCode::VAR, output, input[axes]);
    return output;
}

Jade Jade::max(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction_unary(OpCode::MAX, output, input);
    else Dispatcher::execute_reduction_unary(OpCode::MAX, output, input[axes]);
    return output;
}

Jade Jade::min(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction_unary(OpCode::MIN, output, input);
    else Dispatcher::execute_reduction_unary(OpCode::MIN, output, input[axes]);
    return output;
}

Jade Jade::mean(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction_unary(OpCode::MEAN, output, input);
    else Dispatcher::execute_reduction_unary(OpCode::MEAN, output, input[axes]);
    return output;
}

Jade Jade::dot(const Jade& other) const {
    if (this->ndims == 1 && other.ndims == 1) {
        if (this->shape[0] != other.shape[0]) {
            std::string msg = "Dot product requires two 1D Jades of identical size.";
            LOG_ERR(msg);
            throw ShapeMismatchException(msg);
        }
        Jade view(this->dtype, 0.0f, static_cast<uint64_t*>(nullptr), static_cast<uint64_t>(0));
        Dispatcher::execute_binary(OpCode::DOT, view, *this, other);
        return view;
    }

    if (this->ndims == 2 && other.ndims == 2) {
        LOG_WARN("Force-routed Jade Through matmul instead of dot().");
        return this->matmul(other);
    }
    std::string msg = "ND-Dot contraction is distinct from Batched MatMul. Use matmul().";
    LOG_ERR(msg);
    throw ShapeMismatchException(msg);
}

Jade Jade::argmax(const Jade& input, std::initializer_list<uint64_t> axes) {
    // indexes are uint64_t
    Jade output(DType::UINT64, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction_unary(OpCode::ARGMAX, output, input);
    else Dispatcher::execute_reduction_unary(OpCode::ARGMAX, output, input[axes]);
    return output;
}

Jade Jade::argmin(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(DType::UINT64, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction_unary(OpCode::ARGMIN, output, input);
    else Dispatcher::execute_reduction_unary(OpCode::ARGMIN, output, input[axes]);
    return output;
}

