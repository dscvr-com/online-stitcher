#include "logger.hpp"

#ifndef OPTONAUT_IMAGE_SINK_HEADER
#define OPTONAUT_IMAGE_SINK_HEADER

namespace optonaut {

    template <typename DataType> class Sink {
    public:
        virtual void Push(DataType in) = 0;
        virtual void Finish() = 0;
    protected: 
	};

    template <typename DataType> class FunctionSink : public Sink<DataType> {
    private:
        std::function<void(DataType)> func;
    public:
        FunctionSink(
                std::function<void(DataType)> func
                ) : func(func) { }

        virtual void Push(DataType in) {
            func(in);
        }
        virtual void Finish() { }
    };
    
    template <typename InType, typename OutType> class MapSink : public Sink<InType> {
    private:
        std::function<OutType(InType)> func;
        Sink<OutType> &outSink;
    public:
        MapSink(std::function<OutType(InType)> func,
                Sink<OutType> &outSink
                ) : func(func), outSink(outSink) { }

        virtual void Push(InType in) {
            outSink.Push(func(in));
        }

        virtual void Finish() { 
            outSink.Finish(); 
        }
    };

    template <typename DataType> class TeeSink : public Sink<DataType> {
    private:
        Sink<DataType> &a, &b; 
    public:
        TeeSink(Sink<DataType> &a,
                Sink<DataType> &b
                ) : a(a), b(b) { }
        virtual void Push(DataType in) {
            Log << "Received Image.";
            a.Push(in);
            b.Push(in);
        }

        virtual void Finish() {
            a.Finish();
            b.Finish();
        }
    };
}

#endif
