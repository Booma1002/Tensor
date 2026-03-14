#include "header/Autograd.hpp"
#include <stdexcept>

namespace bm {

    Node::Node(const Jade& data_in, bool req_grad)
            : data(data_in), requires_grad(req_grad), num_parents(0) {
        if (requires_grad) grad = Jade::zeros_like(data);
    }

// ---------------------------------------------------------
// ------------------    ADDITION WIRING    ----------------
// ---------------------------------------------------------
    std::shared_ptr<Node> Node::add(const std::shared_ptr<Node>& other) {
        Jade out_data = this->data + other->data;
        bool out_req_grad = this->requires_grad || other->requires_grad;

        auto out_node = std::make_shared<Node>(out_data, out_req_grad);

        if (out_req_grad) {
            // Hot Loop: O(1) array assignment instead of std::vector::push_back
            out_node->parents[0] = shared_from_this();
            out_node->parents[1] = other;
            out_node->num_parents = 2;

            // Capture pointers by value to keep the graph alive
            auto this_ptr = shared_from_this();
            auto other_ptr = other;

            out_node->backward_op = [this_ptr, other_ptr, out_node]() {
                if (this_ptr->requires_grad) {
                    this_ptr->grad += out_node->grad; // Accumulate!
                }
                if (other_ptr->requires_grad) {
                    other_ptr->grad += out_node->grad; // Accumulate!
                }
            };
        }
        return out_node;
    }

// ---------------------------------------------------------
// --------------    MULTIPLICATION WIRING    --------------
// ---------------------------------------------------------
    std::shared_ptr<Node> Node::mul(const std::shared_ptr<Node>& other) {
        Jade out_data = this->data * other->data;
        bool out_req_grad = this->requires_grad || other->requires_grad;

        auto out_node = std::make_shared<Node>(out_data, out_req_grad);

        if (out_req_grad) {
            out_node->parents[0] = shared_from_this();
            out_node->parents[1] = other;
            out_node->num_parents = 2;

            auto this_ptr = shared_from_this();
            auto other_ptr = other;

            out_node->backward_op = [this_ptr, other_ptr, out_node]() {
                if (this_ptr->requires_grad) {
                    // Product Rule: d(A*B)/dA = B. Chain Rule: B * upstream_grad
                    this_ptr->grad += (other_ptr->data * out_node->grad);
                }
                if (other_ptr->requires_grad) {
                    // Product Rule: d(A*B)/dB = A. Chain Rule: A * upstream_grad
                    other_ptr->grad += (this_ptr->data * out_node->grad);
                }
            };
        }
        return out_node;
    }

// ---------------------------------------------------------
// ------    TOPOLOGICAL SORT & BACKWARD EXECUTION    ------
// ---------------------------------------------------------
    void Node::build_topo(std::vector<std::shared_ptr<Node>>& topo,
                          std::unordered_set<std::shared_ptr<Node>>& visited) {
        if (visited.find(shared_from_this()) == visited.end()) {
            visited.insert(shared_from_this());

            // Loop strictly up to num_parents. No iterator overhead.
            for (uint8_t i = 0; i < num_parents; ++i) {
                parents[i]->build_topo(topo, visited);
            }
            topo.push_back(shared_from_this());
        }
    }

    void Node::backward() {
        if (!requires_grad) return;

        // Seed the initial gradient to 1.0 (dLoss/dLoss = 1.0)
        this->grad = Jade::ones_like(this->data);

        std::vector<std::shared_ptr<Node>> topo;
        std::unordered_set<std::shared_ptr<Node>> visited;

        // Build the ordered graph
        build_topo(topo, visited);

        // Execute the chain rule closures in reverse topological order
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_op) {
                (*it)->backward_op();
            }
        }
    }

} // namespace bm