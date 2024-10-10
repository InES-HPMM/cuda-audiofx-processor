#pragma once

#include <signal_graph.hpp>

#include "gpu_fx.cuh"

class IGpuSignalVertex {
   public:
    virtual ~IGpuSignalVertex() {}

    virtual const BufferRackSpecs& getOutputSpecs() = 0;
    virtual const BufferRackSpecs& getInputSpecs() = 0;
    virtual const std::vector<Buffer*>& getSrcPtr() = 0;
    virtual const std::vector<Buffer*>& getDestPtr() = 0;
    virtual std::vector<IGpuSignalVertex*>& getParents() = 0;
    virtual std::vector<IGpuSignalVertex*>& getChildren() = 0;
    virtual cudaGraphNode_t getProcessNode() = 0;
    virtual cudaGraphNode_t* getProcessNodePtr() = 0;
    virtual size_t getIncomingChannelCount() = 0;
    virtual size_t getOutgoingChannelCount() = 0;
};

class IGpuSignalGraph : public ISignalGraph {
   public:
    static IGpuSignalGraph* createGpuSignalGraph();
    virtual ~IGpuSignalGraph() {}
    virtual IGpuSignalVertex* add(IGpuFx* fx, IGpuSignalVertex* parent = nullptr) = 0;
    virtual IGpuSignalVertex* addRoot(IGpuFx* fx) = 0;
    virtual std::vector<IGpuSignalVertex*> split(std::vector<IGpuFx*> fx, IGpuSignalVertex* parent = nullptr) = 0;
    virtual IGpuSignalVertex* merge(IGpuFx* fx, std::vector<IGpuSignalVertex*> parents) = 0;
};