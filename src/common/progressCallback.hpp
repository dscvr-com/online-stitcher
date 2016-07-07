#include <functional>
#include <vector>
// Note: We need to stick with cassert here
// since this code might be included from app side
// and that does not work well with the Assert header.
#include <cassert>

#ifndef OPTONAUT_PROGRESS_CALLBACK_HEADER
#define OPTONAUT_PROGRESS_CALLBACK_HEADER
namespace optonaut {
    /*
     * Definition of callback signature for progress. 
     */
    typedef std::function<bool(float)> ProgressCallbackFunction;
    
    /*
     * Encapsulates a progress callback. 
     */
    class ProgressCallback {
    private:
        ProgressCallbackFunction callback;
        float progress;

    public:

        /*
         * Creates a new instance of this class, using the given callback function.
         *
         * @param function The callback to call on progress change. 
         */
        ProgressCallback(ProgressCallbackFunction function) : callback(function), progress(0) { }
        
        /*
         * Notifies the registered callback about the progress update.
         */
        bool operator()(float progress) {
            assert(progress >= 0);
            this->progress = progress;
            return callback(progress);
        }
       
        /*
         * Returns the last progress. 
         */ 
        float GetLastProgress() {
            return progress;
        }

        /*
         * An empty progress callback, nothing is listening to it. 
         */
        static ProgressCallback Empty;
    };
   
    /*
     * A progress callback that accumuates multiple child progress callbacks. 
     */ 
    class ProgressCallbackAccumulator {
    private:
        size_t count;
        ProgressCallback &parent;
        std::vector<float> weights;
        std::vector<ProgressCallback> callbacks;
        
        bool updateAndCall() {
            float progress = 0;
            
            for (size_t i = 0; i < count; i++) {
                progress += weights[i] * callbacks[i].GetLastProgress();
            }
            
            return parent(progress);
        }
    public:
        /*
         * Creates a new instance of this class.
         *
         * @param parent The parent progress callback to call. 
         * @param weights The weights for each child progress callback. For each weight passed, a child callback is created.  
         */
        ProgressCallbackAccumulator(ProgressCallback &parent, std::vector<float> weights) : count (weights.size()), parent(parent), weights(weights) {
            callbacks.reserve(count);
            for (size_t i = 0; i < count; i++) {
                callbacks.emplace_back([&](float) -> bool {
                    return updateAndCall();
                });
            }
        }

        /*
         * Gets the child progress callback at the given index. 
         */
        ProgressCallback &At(size_t id) {
            assert(id < callbacks.size());
            return callbacks[id];
        }
    };

}

#endif
