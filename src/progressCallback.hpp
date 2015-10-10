
#ifndef OPTONAUT_PROGRESS_CALLBACK_HEADER
#define OPTONAUT_PROGRESS_CALLBACK_HEADER
namespace optonaut {
    typedef std::function<bool(float)> ProgressCallbackFunction;
    
    class ProgressCallback {
    private:
        ProgressCallbackFunction callback;
        float progress;
    public:
        ProgressCallback(ProgressCallbackFunction function) : callback(function), progress(0) { }
        
        bool operator()(float progress) {
            this->progress = progress;
            return callback(progress);
        }
        
        float GetLastProgress() {
            return progress;
        }
    };
    
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
        ProgressCallbackAccumulator(ProgressCallback &parent, std::vector<float> weights) : count (weights.size()), parent(parent), weights(weights) {
            callbacks.reserve(count);
            for (size_t i = 0; i < count; i++) {
                callbacks.emplace_back([&](float) -> bool {
                    return updateAndCall();
                });
            }
        }
        ProgressCallback &At(size_t id) {
            assert(id < callbacks.size());
            return callbacks[id];
        }
    };
}

#endif
