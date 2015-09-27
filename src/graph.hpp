#include <map>
#include <vector>

using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_SPARSE_GRAPH_HEADER
#define OPTONAUT_SPARSE_GRAPH_HEADER

namespace optonaut {
    
    template <typename ValueType>
    struct Edge {
        size_t from;
        size_t to;

        ValueType value;
        
        Edge(size_t from, size_t to, ValueType value) : from(from), to(to), value(value) {
            
        }
    };
    
    template <typename ValueType>
    class SparseGraph {
    public:
        map<size_t, vector<Edge<ValueType>>> adj;

        SparseGraph() : adj() {

        }
        
        const map<size_t, vector<Edge<ValueType>>> &GetEdges() const {
            return adj;
        }

        void Insert(size_t from, size_t to, ValueType &value) {
            adj[from].push_back(Edge<ValueType>(from, to, value));
        }
    };
}
#endif
