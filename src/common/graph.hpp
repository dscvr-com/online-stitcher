#include <map>
#include <vector>

using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_SPARSE_GRAPH_HEADER
#define OPTONAUT_SPARSE_GRAPH_HEADER

namespace optonaut {
    
    template <typename ValueType>
    struct _Edge {
        size_t from;
        size_t to;

        ValueType value;
        
        _Edge(size_t from, size_t to, ValueType value) : 
            from(from), to(to), value(value) {
            
        }
    };
    
    template <typename ValueType>
    class SparseGraph {
    public:
        typedef _Edge<ValueType> Edge;
        typedef vector<Edge> Edges;
        typedef map<size_t, Edges> AdjList;

        AdjList adj;

        SparseGraph() : adj() {

        }
        
        const AdjList &GetEdges() const {
            return adj;
        }

        void Insert(size_t from, size_t to, ValueType &value) {
            adj[from].push_back(Edge(from, to, value));
        }
    };
}
#endif
