#include "header/Engine.hpp"
using namespace bm;

std::atomic_uint64_t watching;
void see(Jade& t, std::string msg =""){
    std::cout << std::format("\n\n{:x}) {}:-\n",
             watching.load(std::memory_order::relaxed), msg);
    watching.fetch_add(1, std::memory_order_relaxed);
    t.display(2, 6);
}

int main(){
    LOG_INFO("[Engine] Initiating Bare Metal (BM) Ignition Sequence...");
    watching.store(0, std::memory_order_relaxed);

    bm::Jade W(DType::FLOAT64, 6.1f, 128, 64);
    bm::Jade b(DType::FLOAT64, 4.3f, 64);
    W+=b;
    see(W, "W");
    see(b,"b");
    bm::Jade Ar = Jade::arange(DType::FLOAT64, Slice(11, 111, 5));
    see(Ar, "arange");
}