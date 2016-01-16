#include <opencv2/opencv.hpp>
#include <map>

#ifndef OPTONAUT_BIMAP_HEADER
#define OPTONAUT_BIMAP_HEADER

namespace optonaut {
    template <typename T, typename V>
    class BiMap {
        private:
        std::map<T, V> forward;
        std::map<V, T> backward;

        public:
        //Add graph vs image queries and insertion.
        void Insert(const T &key, const V &val) {
            forward[key] = val;
            backward[val] = key;
        } 

        bool GetValue(const T &key, V &value) const {
            auto it = forward.find(key);
            if(it != forward.end()) {
                value = it->second;
                return true;
            }

            return false;
        }
        
        bool GetKey(const V &value, T &key) const {
            auto it = backward.find(value);
            if(it != backward.end()) {
                key = it->second;
                return true;
            }

            return false;
        }

        typename std::map<V, T>::iterator begin() const {
            return forward.begin();
        }

        typename std::map<V, T>::iterator end() const {
            return forward.end();
        }

        size_t Size() const {
            return forward.size();
        }
        
    };
}
#endif
