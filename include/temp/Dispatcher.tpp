#pragma once
#include "header/JadeReactor.hpp"

namespace bm{
    template<typename... Args>
    void Dispatcher::execute_binary(OpCode op, Jade& out, const Jade& a, const Jade& b, Args&... args) {
        // TODO: check if A, B are on the same device
        Device target_device = Device::CPU;
        JadeReactor react;
        if (op == OpCode::MATMUL) react = JadeReactor::react_matmul(op, out, a, b, args...);
        else react = JadeReactor::react_binary(op, out, a, b, args...);
        Kernel kernel_func = Registry::get().lookup(op, target_device);
        kernel_func(react);
        std::string msg = std::format("[Dispatcher] Executed Binary Reaction: OpCode={:#x}" ,static_cast<int>(op));
        LOG_DEBUG(msg);
    }


    template<typename... Args>
    void Dispatcher::execute_unary(OpCode op, Jade& out, const Jade& a, Args&... args) {
        Device target_device = Device::CPU;
        JadeReactor react = JadeReactor::react_unary(op, out, a, args...);
        Kernel kernel_func = Registry::get().lookup(op, target_device);
        kernel_func(react);
        std::string msg = std::format("[Dispatcher] Executed Unary Reaction: OpCode={:#x}" ,static_cast<int>(op));
        LOG_DEBUG(msg);
    }

    template<typename... Args>
    void Dispatcher::execute_scalar(OpCode op, Jade& out, const double a, Args&... args) {
        Device target_device = Device::CPU;
        JadeReactor react = JadeReactor::react_scalar(op, out, a, args...);
        Kernel kernel_func = Registry::get().lookup(op, target_device);
        kernel_func(react);
        std::string msg = std::format("[Dispatcher] Executed Scalar Reaction: OpCode={:#x}" ,static_cast<int>(op));
        LOG_DEBUG(msg);
    }

}