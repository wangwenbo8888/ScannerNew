#include "Pipeline.h"
#include <algorithm>

namespace Scanner::workflow {

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::addStage(Stage* stage) {
    stages_.push_back(stage);
}

Result Pipeline::start() {
    if (running_) {
        return Result::fail("Pipeline already running");
    }
    if (stages_.empty()) {
        return Result::fail("No stages configured");
    }

    running_ = true;
    for (auto* stage : stages_) {
        stage->stop();
        threads_.emplace_back(&Pipeline::stageLoop, this, stage);
    }
    return Result::ok();
}

void Pipeline::stop() {
    running_ = false;
    for (auto* stage : stages_) {
        stage->stop();
    }
    waitComplete();
}

void Pipeline::waitComplete() {
    for (auto& t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    threads_.clear();
}

void Pipeline::stageLoop(Stage* stage) {
    while (!stage->isStopped() && running_) {
        Result r = stage->process();
        if (!r.success) {
            stage->stop();
        }
    }
}

} // namespace Scanner::workflow
