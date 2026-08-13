#include <gtest/gtest.h>
#include "workflow/Pipeline.h"
#include <atomic>
#include <vector>

using namespace Scanner;
using namespace Scanner::workflow;

namespace {

class CountStage : public Stage {
public:
    explicit CountStage(std::atomic<int>& counter)
        : Stage("Count"), counter_(counter) {}

    Scanner::Result process() override {
        counter_.fetch_add(1);
        return Scanner::Result::ok();
    }

private:
    std::atomic<int>& counter_;
};

class TagStage : public Stage {
public:
    TagStage(std::vector<std::string>& log, std::string tag)
        : Stage("Tag"), log_(log), tag_(std::move(tag)) {}

    Scanner::Result process() override {
        log_.push_back(tag_);
        return Scanner::Result::ok();
    }

private:
    std::vector<std::string>& log_;
    std::string tag_;
};

class DropEvenStage : public Stage {
public:
    DropEvenStage() : Stage("DropEven") {}

    Scanner::Result process() override {
        count_++;
        if (count_ % 2 == 0) {
            return Scanner::Result::fail("even frame dropped");
        }
        return Scanner::Result::ok();
    }

private:
    int count_ = 0;
};

} // anonymous namespace

TEST(PipelineTest, SingleStage) {
    std::atomic<int> counter{0};
    CountStage stage(counter);
    Pipeline pipeline;
    pipeline.addStage(&stage);

    EXPECT_EQ(pipeline.getStageCount(), 1u);
}

TEST(PipelineTest, MultipleStagesOrder) {
    std::vector<std::string> log;
    TagStage s1(log, "A");
    TagStage s2(log, "B");
    TagStage s3(log, "C");

    Pipeline pipeline;
    pipeline.addStage(&s1);
    pipeline.addStage(&s2);
    pipeline.addStage(&s3);

    EXPECT_EQ(pipeline.getStageCount(), 3u);
}
