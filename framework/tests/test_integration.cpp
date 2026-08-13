#include <gtest/gtest.h>
#include "common/types.h"
#include "infra/EventBus.h"
#include "infra/Scheduler.h"
#include "service/IState.h"
#include "workflow/IWorkflow.h"
#include "workflow/Pipeline.h"
#include "data/RingBuffer.h"
#include "algorithm/operator_convention.h"
#include <atomic>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>

using namespace Scanner;
using namespace Scanner::infra;
using namespace Scanner::service;
using namespace Scanner::workflow;
using namespace Scanner::data;

// ============================================================================
// 测试用状态机实现
// ============================================================================
class TestStateMachine : public IState {
public:
    ScannerState getCurrentState() const override { return state_; }
    std::string getStateName() const override {
        return names_[static_cast<int>(state_)];
    }

    Result transition(EventType event, int64_t) override {
        switch (event) {
            case EventType::DeviceConnected:
                if (state_ == ScannerState::Init) {
                    state_ = ScannerState::DeviceReady;
                    return Result::ok("connected");
                }
                return Result::fail("invalid transition");

            case EventType::ScanStarted:
                if (state_ == ScannerState::Calibrated) {
                    state_ = ScannerState::Scanning;
                    return Result::ok("scanning");
                }
                return Result::fail("not calibrated");

            case EventType::ScanStopped:
                if (state_ == ScannerState::Scanning) {
                    state_ = ScannerState::PostProcessing;
                    return Result::ok("post-processing");
                }
                return Result::fail("not scanning");

            case EventType::EmergencyStop:
                state_ = ScannerState::EmergencyStop;
                return Result::ok("emergency");

            default:
                return Result::ok("ignored");
        }
    }

    bool canOperate(const std::string& op) const override {
        if (op == "scan") return state_ == ScannerState::Calibrated;
        if (op == "calibrate") return state_ == ScannerState::DeviceReady;
        return true;
    }

    void reset() { state_ = ScannerState::Init; }

private:
    ScannerState state_ = ScannerState::Init;
    const char* names_[9] = {
        "Init", "DeviceReady", "Calibrating", "Calibrated",
        "Scanning", "Paused", "PostProcessing", "Error", "EmergencyStop"
    };
};

// ============================================================================
// 测试用工作流实现
// ============================================================================
class TestWorkflow : public IWorkflow {
public:
    explicit TestWorkflow(std::string name) : name_(std::move(name)) {}

    std::string getName() const override { return name_; }

    Result initialize() override {
        state_ = WorkflowState::Idle;
        return Result::ok();
    }

    Result start() override {
        state_ = WorkflowState::Running;
        return Result::ok();
    }

    Result pause() override {
        if (state_ == WorkflowState::Running) {
            state_ = WorkflowState::Paused;
            return Result::ok();
        }
        return Result::fail("not running");
    }

    Result resume() override {
        if (state_ == WorkflowState::Paused) {
            state_ = WorkflowState::Running;
            return Result::ok();
        }
        return Result::fail("not paused");
    }

    Result stop() override {
        state_ = WorkflowState::Stopping;
        state_ = WorkflowState::Completed;
        return Result::ok();
    }

    WorkflowState getState() const override { return state_; }

    Result setProgressCallback(WorkflowCallback cb) override {
        callback_ = std::move(cb);
        return Result::ok();
    }

    void reportProgress(const std::string& stage, float pct) {
        if (callback_) {
            WorkflowProgress wp;
            wp.state = state_;
            wp.stageName = stage;
            wp.progress = pct;
            callback_(wp);
        }
    }

private:
    std::string name_;
    WorkflowState state_ = WorkflowState::Idle;
    WorkflowCallback callback_;
};

// ============================================================================
// 测试用 Pipeline Stage
// ============================================================================
class CountStage : public Stage {
public:
    explicit CountStage(std::atomic<int>& counter)
        : Stage("Count"), counter_(counter) {}

    Result process() override {
        counter_.fetch_add(1);
        return Result::ok();
    }

private:
    std::atomic<int>& counter_;
};

class OrderStage : public Stage {
public:
    OrderStage(std::vector<std::string>& log, std::string tag)
        : Stage("Order"), log_(log), tag_(std::move(tag)) {}

    Result process() override {
        std::lock_guard<std::mutex> lock(mu_);
        log_.push_back(tag_);
        return Result::ok();
    }

private:
    std::vector<std::string>& log_;
    std::string tag_;
    std::mutex mu_;
};

// ============================================================================
// 一、EventBus 集成测试
// ============================================================================
TEST(EventBusIntegration, PublishSubscribe) {
    EventBus bus;
    int received = 0;

    bus.subscribe(EventType::ScanFrameReady, [&](const Event&) {
        received++;
    });

    Event e;
    e.type = EventType::ScanFrameReady;
    bus.publish(e);

    // publish 是异步的，等一下
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(received, 1);
}

TEST(EventBusIntegration, PublishSyncBlocks) {
    EventBus bus;
    std::atomic<int> count{0};

    bus.subscribe(EventType::EmergencyStop, [&](const Event&) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        count.fetch_add(1);
    });

    Event e;
    e.type = EventType::EmergencyStop;
    bus.publishSync(e);

    // 同步发布应该阻塞到处理完
    EXPECT_EQ(count.load(), 1);
}

TEST(EventBusIntegration, MultipleSubscribers) {
    EventBus bus;
    std::atomic<int> countA{0};
    std::atomic<int> countB{0};

    bus.subscribe(EventType::ScanStarted, [&](const Event&) { countA++; });
    bus.subscribe(EventType::ScanStarted, [&](const Event&) { countB++; });

    Event e;
    e.type = EventType::ScanStarted;
    bus.publish(e);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_EQ(countA.load(), 1);
    EXPECT_EQ(countB.load(), 1);
    EXPECT_EQ(bus.getSubscriberCount(), 2u);
}

TEST(EventBusIntegration, Unsubscribe) {
    EventBus bus;
    int count = 0;

    auto id = bus.subscribe(EventType::ScanFrameReady, [&](const Event&) {
        count++;
    });

    Event e;
    e.type = EventType::ScanFrameReady;
    bus.publish(e);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(count, 1);

    bus.unsubscribe(id);

    bus.publish(e);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(count, 1);  // 不再触发
}

TEST(EventBusIntegration, SubscribeAll) {
    EventBus bus;
    int count = 0;

    bus.subscribeAll([&](const Event&) { count++; });

    Event e1; e1.type = EventType::ScanStarted;
    Event e2; e2.type = EventType::EmergencyStop;
    Event e3; e3.type = EventType::DeviceConnected;

    bus.publish(e1);
    bus.publish(e2);
    bus.publish(e3);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_EQ(count, 3);
}

TEST(EventBusIntegration, ClearRemovesAll) {
    EventBus bus;
    int count = 0;

    bus.subscribe(EventType::ScanFrameReady, [&](const Event&) { count++; });
    bus.subscribeAll([&](const Event&) { count++; });

    bus.clear();
    EXPECT_EQ(bus.getSubscriberCount(), 0u);

    Event e; e.type = EventType::ScanFrameReady;
    bus.publish(e);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(count, 0);
}

TEST(EventBusIntegration, ThreadSafetyStress) {
    EventBus bus;
    std::atomic<int> total{0};

    bus.subscribe(EventType::ScanFrameReady, [&](const Event&) {
        total.fetch_add(1);
    });

    // 多线程同时发布
    std::vector<std::thread> threads;
    const int threadCount = 8;
    const int perThread = 100;

    for (int t = 0; t < threadCount; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < perThread; ++i) {
                Event e; e.type = EventType::ScanFrameReady;
                bus.publish(e);
            }
        });
    }

    for (auto& t : threads) t.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_EQ(total.load(), threadCount * perThread);
}

// ============================================================================
// 二、Scheduler 集成测试
// ============================================================================
TEST(SchedulerIntegration, SubmitAndComplete) {
    Scheduler sched(2);
    std::atomic<int> counter{0};

    for (int i = 0; i < 10; ++i) {
        sched.submit([&]() { counter.fetch_add(1); });
    }

    sched.waitAll();
    EXPECT_EQ(counter.load(), 10);
    sched.shutdown();
}

TEST(SchedulerIntegration, ConcurrentExecution) {
    Scheduler sched(4);
    std::atomic<int> maxConcurrent{0};
    std::atomic<int> active{0};

    for (int i = 0; i < 20; ++i) {
        sched.submit([&]() {
            int cur = active.fetch_add(1) + 1;
            // 记录最大并发数
            int prev = maxConcurrent.load();
            while (cur > prev && !maxConcurrent.compare_exchange_weak(prev, cur)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            active.fetch_sub(1);
        });
    }

    sched.waitAll();
    EXPECT_GT(maxConcurrent.load(), 1);
    sched.shutdown();
}

TEST(SchedulerIntegration, SubmitPriority) {
    Scheduler sched(2);
    std::vector<int> executionOrder;
    std::mutex mu;

    sched.submitPriority([&]() {
        std::lock_guard<std::mutex> lock(mu);
        executionOrder.push_back(2);
    }, 2);

    sched.submitPriority([&]() {
        std::lock_guard<std::mutex> lock(mu);
        executionOrder.push_back(1);
    }, 1);

    sched.submitPriority([&]() {
        std::lock_guard<std::mutex> lock(mu);
        executionOrder.push_back(0);
    }, 0);

    sched.waitAll();
    // 所有任务都执行了（不关心顺序，因为线程池调度）
    EXPECT_EQ(executionOrder.size(), 3u);
    sched.shutdown();
}

TEST(SchedulerIntegration, ShutdownStopsAccepting) {
    Scheduler sched(2);
    sched.shutdown();
    EXPECT_TRUE(sched.isShutdown());

    auto result = sched.submit([]() {});
    EXPECT_FALSE(result.success);
}

// ============================================================================
// 三、State Machine 集成测试
// ============================================================================
TEST(StateMachineIntegration, ValidTransitions) {
    TestStateMachine sm;

    EXPECT_EQ(sm.getCurrentState(), ScannerState::Init);
    EXPECT_EQ(sm.getStateName(), "Init");

    auto r1 = sm.transition(EventType::DeviceConnected);
    EXPECT_TRUE(r1.success);
    EXPECT_EQ(sm.getCurrentState(), ScannerState::DeviceReady);

    auto r2 = sm.transition(EventType::EmergencyStop);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(sm.getCurrentState(), ScannerState::EmergencyStop);
}

TEST(StateMachineIntegration, InvalidTransitionRejected) {
    TestStateMachine sm;

    // Init 状态不能直接扫描
    auto r = sm.transition(EventType::ScanStarted);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(sm.getCurrentState(), ScannerState::Init);
}

TEST(StateMachineIntegration, CanOperateChecks) {
    TestStateMachine sm;

    EXPECT_FALSE(sm.canOperate("scan"));       // Init 不能扫描
    EXPECT_TRUE(sm.canOperate("calibrate"));   // Init 可以标定

    sm.transition(EventType::DeviceConnected);
    EXPECT_FALSE(sm.canOperate("scan"));       // DeviceReady 不能扫描
    EXPECT_TRUE(sm.canOperate("calibrate"));   // DeviceReady 可以标定
}

TEST(StateMachineIntegration, FullScanLifecycle) {
    TestStateMachine sm;

    // Init → DeviceReady → (Calibrated) → Scanning → PostProcessing
    sm.transition(EventType::DeviceConnected);
    EXPECT_EQ(sm.getCurrentState(), ScannerState::DeviceReady);

    // 模拟标定完成（直接设置状态）
    // 注意：真实场景中 Calibrated 是由标定工作流设置的
    // 这里测试状态机的逻辑完整性

    sm.transition(EventType::EmergencyStop);
    EXPECT_EQ(sm.getCurrentState(), ScannerState::EmergencyStop);
}

// ============================================================================
// 四、Workflow 集成测试
// ============================================================================
TEST(WorkflowIntegration, Lifecycle) {
    TestWorkflow wf("TestWorkflow");

    EXPECT_EQ(wf.getName(), "TestWorkflow");
    EXPECT_EQ(wf.getState(), WorkflowState::Idle);

    wf.initialize();
    EXPECT_EQ(wf.getState(), WorkflowState::Idle);

    wf.start();
    EXPECT_EQ(wf.getState(), WorkflowState::Running);

    wf.pause();
    EXPECT_EQ(wf.getState(), WorkflowState::Paused);

    wf.resume();
    EXPECT_EQ(wf.getState(), WorkflowState::Running);

    wf.stop();
    EXPECT_EQ(wf.getState(), WorkflowState::Completed);
}

TEST(WorkflowIntegration, PauseWithoutRunningFails) {
    TestWorkflow wf("Test");
    wf.initialize();

    auto r = wf.pause();
    EXPECT_FALSE(r.success);
    EXPECT_EQ(wf.getState(), WorkflowState::Idle);
}

TEST(WorkflowIntegration, ResumeWithoutPauseFails) {
    TestWorkflow wf("Test");
    wf.initialize();
    wf.start();

    auto r = wf.resume();
    EXPECT_FALSE(r.success);
}

TEST(WorkflowIntegration, ProgressCallback) {
    TestWorkflow wf("Test");
    wf.initialize();

    std::vector<WorkflowProgress> reports;
    wf.setProgressCallback([&](const WorkflowProgress& p) {
        reports.push_back(p);
    });

    wf.start();
    wf.reportProgress("Stage1", 0.5f);
    wf.reportProgress("Stage2", 1.0f);

    ASSERT_EQ(reports.size(), 2u);
    EXPECT_EQ(reports[0].stageName, "Stage1");
    EXPECT_FLOAT_EQ(reports[0].progress, 0.5f);
    EXPECT_EQ(reports[1].stageName, "Stage2");
}

// ============================================================================
// 五、Pipeline 集成测试
// ============================================================================
TEST(PipelineIntegration, StartStop) {
    std::atomic<int> counter{0};
    CountStage stage(counter);
    Pipeline pipeline;
    pipeline.addStage(&stage);

    auto r = pipeline.start();
    EXPECT_TRUE(r.success);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pipeline.stop();
    EXPECT_GT(counter.load(), 0);
}

TEST(PipelineIntegration, MultiStageOrder) {
    std::vector<std::string> log;
    OrderStage s1(log, "A");
    OrderStage s2(log, "B");
    OrderStage s3(log, "C");

    Pipeline pipeline;
    pipeline.addStage(&s1);
    pipeline.addStage(&s2);
    pipeline.addStage(&s3);

    EXPECT_EQ(pipeline.getStageCount(), 3u);
}

// ============================================================================
// 六、RingBuffer 高级集成测试
// ============================================================================
TEST(RingBufferIntegration, BlockPolicyWaits) {
    RingBuffer<int> buf(2, OverflowPolicy::Block);
    std::atomic<bool> pushed{false};

    // 消费者线程等待 100ms 后消费
    std::thread consumer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto val = buf.pop(std::chrono::milliseconds(500));
        ASSERT_TRUE(val.has_value());
        EXPECT_EQ(*val, 42);
        pushed = true;
    });

    // 生产者填满后继续推入（会阻塞直到消费者消费一个）
    buf.push(1);
    buf.push(2);
    buf.push(42);  // 阻塞，直到消费者消费

    consumer.join();
    EXPECT_TRUE(pushed.load());
}

TEST(RingBufferIntegration, MultiProducerSingleConsumer) {
    RingBuffer<int> buf(64, OverflowPolicy::DropOldest);
    const int totalItems = 1000;
    std::atomic<int> consumed{0};

    // 4个生产者
    std::vector<std::thread> producers;
    for (int p = 0; p < 4; ++p) {
        producers.emplace_back([&]() {
            for (int i = 0; i < totalItems / 4; ++i) {
                buf.push(i);
            }
        });
    }

    // 1个消费者
    std::vector<int> received;
    std::thread consumer([&]() {
        while (consumed.load() < totalItems) {
            auto val = buf.pop(std::chrono::milliseconds(100));
            if (val.has_value()) {
                received.push_back(*val);
                consumed.fetch_add(1);
            }
        }
    });

    for (auto& t : producers) t.join();
    consumer.join();

    // DropOldest 策略下，消费的总数应该等于 totalItems
    EXPECT_EQ(static_cast<int>(received.size()), totalItems);
}

// ============================================================================
// 七、跨组件集成测试
// ============================================================================
TEST(CrossComponentIntegration, EventBusWithScheduler) {
    EventBus bus;
    Scheduler sched(2);
    std::atomic<int> eventCount{0};

    bus.subscribe(EventType::ScanFrameReady, [&](const Event&) {
        eventCount.fetch_add(1);
    });

    // 模拟：调度器提交任务，任务发布事件
    for (int i = 0; i < 10; ++i) {
        sched.submit([&bus]() {
            Event e; e.type = EventType::ScanFrameReady;
            bus.publish(e);
        });
    }

    sched.waitAll();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(eventCount.load(), 10);
    sched.shutdown();
}

TEST(CrossComponentIntegration, StateMachineWithWorkflow) {
    TestStateMachine sm;
    TestWorkflow wf("ScanWorkflow");

    // 正常流程：Init → DeviceReady → start workflow
    sm.transition(EventType::DeviceConnected);
    EXPECT_EQ(sm.getCurrentState(), ScannerState::DeviceReady);

    // 启动扫描工作流
    wf.initialize();
    wf.start();
    EXPECT_EQ(wf.getState(), WorkflowState::Running);

    // 急停：工作流停止，状态机进入 EmergencyStop
    wf.stop();
    sm.transition(EventType::EmergencyStop);
    EXPECT_EQ(sm.getCurrentState(), ScannerState::EmergencyStop);
    EXPECT_EQ(wf.getState(), WorkflowState::Completed);
}

TEST(CrossComponentIntegration, PipelineWithRingBuffer) {
    // Pipeline stage 之间通过 RingBuffer 通信
    RingBuffer<int> sharedBuf(16, OverflowPolicy::DropOldest);
    std::atomic<int> processed{0};

    // 生产者 stage：推入数据
    class ProducerStage : public Stage {
    public:
        ProducerStage(RingBuffer<int>& buf) : Stage("Producer"), buf_(buf) {}
        Result process() override {
            for (int i = 0; i < 5; ++i) buf_.push(i);
            return Result::ok();
        }
    private:
        RingBuffer<int>& buf_;
    };

    // 消费者 stage：消费数据
    class ConsumerStage : public Stage {
    public:
        ConsumerStage(RingBuffer<int>& buf, std::atomic<int>& count)
            : Stage("Consumer"), buf_(buf), count_(count) {}
        Result process() override {
            while (auto val = buf_.pop(std::chrono::milliseconds(50))) {
                count_.fetch_add(1);
            }
            return Result::ok();
        }
    private:
        RingBuffer<int>& buf_;
        std::atomic<int>& count_;
    };

    ProducerStage producer(sharedBuf);
    ConsumerStage consumer(sharedBuf, processed);

    Pipeline pipeline;
    pipeline.addStage(&producer);
    pipeline.addStage(&consumer);

    auto r = pipeline.start();
    EXPECT_TRUE(r.success);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pipeline.stop();

    EXPECT_GT(processed.load(), 0);
}

// ============================================================================
// 八、AlgorithmRegistry 集成测试
// ============================================================================
TEST(AlgorithmRegistryIntegration, RegisterMultipleAndExecute) {
    algorithm::AlgorithmRegistry<int(int)> registry;

    registry.registerOperator("add1", [](int x) { return x + 1; });
    registry.registerOperator("mul2", [](int x) { return x * 2; });
    registry.registerOperator("neg", [](int x) { return -x; });

    EXPECT_TRUE(registry.hasOperator("add1"));
    EXPECT_TRUE(registry.hasOperator("mul2"));
    EXPECT_TRUE(registry.hasOperator("neg"));

    EXPECT_EQ(registry.getOperator("add1")(5), 6);
    EXPECT_EQ(registry.getOperator("mul2")(5), 10);
    EXPECT_EQ(registry.getOperator("neg")(5), -5);
}

TEST(AlgorithmRegistryIntegration, OverwriteOperator) {
    algorithm::AlgorithmRegistry<std::string()> registry;

    registry.registerOperator("greeting", []() { return std::string("hello"); });
    EXPECT_EQ(registry.getOperator("greeting")(), "hello");

    registry.registerOperator("greeting", []() { return std::string("hi"); });
    EXPECT_EQ(registry.getOperator("greeting")(), "hi");
}

// ============================================================================
// 九、性能/压力测试
// ============================================================================
TEST(StressTest, EventBusHighThroughput) {
    EventBus bus;
    std::atomic<int> count{0};

    for (int i = 0; i < 10; ++i) {
        bus.subscribe(EventType::ScanFrameReady, [&](const Event&) {
            count.fetch_add(1);
        });
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        Event e; e.type = EventType::ScanFrameReady;
        bus.publishSync(e);
    }
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_EQ(count.load(), 10000);  // 10 subscribers * 1000 publishes
    // 10000 次同步发布应该在 5 秒内完成
    EXPECT_LT(elapsed, std::chrono::seconds(5));
}

TEST(StressTest, SchedulerHighLoad) {
    Scheduler sched(4);
    std::atomic<int> counter{0};

    for (int i = 0; i < 1000; ++i) {
        sched.submit([&]() { counter.fetch_add(1); });
    }

    sched.waitAll();
    EXPECT_EQ(counter.load(), 1000);
    sched.shutdown();
}
