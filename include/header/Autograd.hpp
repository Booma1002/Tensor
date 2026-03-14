#pragma once
#include "Jade.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

namespace bm {

    class Node : public std::enable_shared_from_this<Node> {
    public:
        Jade data;
        Jade grad;
        bool requires_grad;

        std::shared_ptr<Node> parents[3]; // unary - binary - ternary
        uint8_t num_parents;

        std::function<void()> backward_op; // chain-rule closure

        Node(const Jade& data_in, bool req_grad = false);

        // DAG wiring ->
        std::shared_ptr<Node> add(const std::shared_ptr<Node>& other);
        std::shared_ptr<Node> mul(const std::shared_ptr<Node>& other);

        void backward();

    private:
        // one-time topological sorting.
        // STL allowed here
        void build_topo(std::vector<std::shared_ptr<Node>>& topo,
                        std::unordered_set<std::shared_ptr<Node>>& visited);
    };

} // namespace bm