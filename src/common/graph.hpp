#include <map>
#include <vector>

using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_SPARSE_GRAPH_HEADER
#define OPTONAUT_SPARSE_GRAPH_HEADER

namespace optonaut {
    
    /*
     * Edge in a graph. 
     *
     * @tparam ValueType The type of the data associated with the edge. 
     */
    template <typename ValueType>
    struct _Edge {
        /*
         * Source node of the edge. 
         */
        size_t from;
        /*
         * Destination node of the edge. 
         */
        size_t to;

        /*
         * Value associated with the edge. 
         */
        ValueType value;
        
        _Edge(size_t from, size_t to, const ValueType &value) : 
            from(from), to(to), value(value) {
        }
    };
    
    /*
     * General-purpose graph, based on an adjacency list. 
     * Good for sparse data. 
     *
     * @tparam ValueType The type of the data associated with each edge. 
     */
    template <typename ValueType>
    class SparseGraph {
    public:
        typedef _Edge<ValueType> Edge;
        typedef vector<Edge> Edges;
        typedef map<size_t, Edges> AdjList;

        AdjList adj;

        /*
         * Creates a new, empty graph. 
         */
        SparseGraph() : adj() {

        }
        
        /*
         * Gets the adjacency list of this graph. 
         */
        AdjList &GetEdges() {
            return adj;
        }

        /*
         * Inserts a new edge into the graph. 
         */
        void Insert(size_t from, size_t to, const ValueType &value) {
            adj[from].emplace_back(from, to, value);
        }
           
        /*
         * Gets the edge between two nodes. 
         *
         * Returns true, if the operation was successful.
         */ 
        bool GetEdge(const size_t from, const size_t to, Edge &edge) {
            for(auto &e : GetEdges()[from]) {
                if(e.to == to) {
                    edge = e;
                    return true;
                }
            } 
            return false;
        }

        vector<Edge*> GetEdges(const size_t from, const size_t to) {
            vector<Edge*> edges;
            for(auto &e : GetEdges()[from]) {
                if(e.to == to) {
                    edges.push_back(&e);
                }
            } 

            return edges;

        }
    };
}
#endif
