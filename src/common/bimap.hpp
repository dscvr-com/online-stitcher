#include <opencv2/opencv.hpp>
#include <map>

#ifndef OPTONAUT_BIMAP_HEADER
#define OPTONAUT_BIMAP_HEADER

namespace optonaut {

    /*
     * Bidirectional map. Constant time lookup 
     * in both directions. 
     */
    template <typename T, typename V>
    class BiMap {
        private:
        std::map<T, V> forward;
        std::map<V, T> backward;

        public:
        /*
         * Inserts a new key-value pair. 
         */
        void Insert(const T &key, const V &val) {
            forward[key] = val;
            backward[val] = key;
        } 

        /*
         * Gets the value for a key in constant time.
         * 
         * @returns True, if the value was found. False otherwise. 
         */ 
        bool GetValue(const T &key, V &value) const {
            auto it = forward.find(key);
            if(it != forward.end()) {
                value = it->second;
                return true;
            }

            return false;
        }
        
        /*
         * Gets the key for a value in constant time.
         * 
         * @returns True, if the key was found. False otherwise. 
         */ 
        bool GetKey(const V &value, T &key) const {
            auto it = backward.find(value);
            if(it != backward.end()) {
                key = it->second;
                return true;
            }

            return false;
        }

        /*
         * Gets an interator pointing at the first element. 
         */
        typename std::map<V, T>::iterator begin() const {
            return forward.begin();
        }

        /*
         * Gets an interator pointing at the last element. 
         */
        typename std::map<V, T>::iterator end() const {
            return forward.end();
        }

        /* 
         * Gets the count of elements in this bimap. 
         */
        size_t Size() const {
            return forward.size();
        }
        
    };
}
#endif
