# base/ — 共享内核（通俗版）

> base/ 是整个工程**最底层的"公共零件箱"**。
> 原则一句话：**谁都依赖它，它不依赖任何人**（除了 C++ 标准库）。

---

## 1. 这是什么

工程里很多地方都要用到同一批"基础件"——比如统一的返回结果、位姿、事件、设备状态等。
如果各模块各写一份，就会打架、对不上。所以把它们抽出来集中放在 `base/`，所有人共用同一份。

它**极薄**：只放公共类型 + 事件总线，**不放任何业务逻辑**。

---

## 2. 里面有什么

| 文件 | 干什么用 | 关键内容 |
|---|---|---|
| `types.h` | 公共类型（零件清单） | `Result`、`Pose`、`Event`/`EventType`、`DeviceState`、`ScanMode`、`QualityFlag` 等 + `UPtr`/`SPtr` 别名 |
| `EventBus.h` / `.cpp` | 事件总线（通知系统） | 订阅 / 发布事件，线程安全，谁关心谁听 |

### 2.1 `types.h` —— 公共零件

- **`Result`**：统一的"执行结果"。所有算子/操作都返回它，用工厂方法造：
  - `Result::ok()` 成功
  - `Result::fail(code, msg)` 失败（另有 `fail(msg)` 重载，错误码 -1）
  - `Result::degraded()` 降级（能用但打折）
  - `Result::warning()` 有警告但成功
  - 带 `success` / `errorCode` / `message` / `qualityFlag` 四个字段
- **`QualityFlag`**：质量等级 `Normal / Degraded / Warning / Fault`
- **`FaultSeverity`**：故障严重度 `Info / Warning / Error / Critical`
- **`ContractLevel`**：接口契约级别 `Stable / Internal / Experimental`
- **`Pose`**：位姿（旋转 `R[9]` + 平移 `t[3]` + 帧号 + 时间戳）
- **`FrameId` / `TimestampMs`**：帧号（`uint64_t`）/ 毫秒时间戳（`uint64_t`）别名
- **`EventType` / `Event`**：事件类型与事件结构（设备连接、扫描开始/停止、故障、会话…）
- **`DeviceState`**：设备状态 `Offline / Connected / Streaming / Error`
- **`ScanMode`**：扫描模式 `MarkerOnly / MarkerPlusLaser`
- **`UPtr<T>` / `SPtr<T>`**：`unique_ptr` / `shared_ptr` 的工程内统一别名

### 2.2 `EventBus` —— 通知系统

事件总线像一个**广播站**：发布者喊一嗓子，所有订阅了这个类型的人都会收到。**只传通知（控制/状态），不搬数据载荷**。

主要操作：

| 方法 | 作用 |
|---|---|
| `subscribe(type, handler)` | 订阅某类事件，拿到一个订阅号 |
| `subscribeAll(handler)` | 订阅所有事件 |
| `publish(event)` | 发布事件（同步触发所有订阅者） |
| `publishSync(event)` | 关键通道用（当前等同 publish） |
| `unsubscribe(id)` | 凭订阅号退订 |
| `clear()` | 清空所有订阅 |
| `getSubscriberCount()` | 查当前订阅总数 |

> ⚠ **`UserDefined` 是通配哨兵**：`subscribeAll` 内部就是 `subscribe(EventType::UserDefined, h)`，而 `publish` 会把 `UserDefined` 订阅者当"全订阅"派发（EventBus.cpp:9）。因此**不要**用 `subscribe(EventType::UserDefined, …)` 表达"只订自定义事件"——那等于订了所有事件。

> ⚠ **回调内禁止再调总线**：`publish` 同步执行且全程持锁——在事件回调里再调同一总线的 `publish` / `subscribe` / `unsubscribe` 会**死锁**（非递归锁）。

---

## 3. 怎么用（极简示例）

### 用 `Result` 返回结果

```cpp
#include "base/types.h"
using namespace Scanner;

Result doSomething() {
    if (/* 出错 */) return Result::fail(1001, "参数非法");
    if (/* 能用但打折 */) return Result::degraded("光线不足");
    return Result::ok();           // 正常成功
}
```

### 用 `EventBus` 收发通知

```cpp
#include "base/EventBus.h"
using namespace Scanner::infra;

EventBus bus;

// 订阅"扫描开始"事件
auto id = bus.subscribe(EventType::ScanStarted, [](const Event& e) {
    // e.param1 / e.param2 可带简单参数
});

// 发布事件
bus.publish(Event{ EventType::ScanStarted, /*sourceId*/0, /*ts*/0, 0, 0 });

// 不用了就退订
bus.unsubscribe(id);
```

---

## 4. 铁规（必须遵守）

- **只放**：公共类型（types）+ 事件总线（EventBus）。
- **禁止**：往里塞任何业务代码。base/ 一旦混进业务，所有模块都会被动拖累。
- **命名空间**：类型用 `Scanner::`，事件总线用 `Scanner::infra::`。
- **依赖**：只依赖标准库，不依赖第三方（OpenCV/CUDA 等一律不许进）。

---

## 5. 怎么接进工程

`base` 编译成一个**静态库**（CMake 目标名就叫 `base`），只编译 `EventBus.cpp`，`types.h` 是纯头文件。

在你的模块的 `CMakeLists.txt` 里链接它即可：

```cmake
target_link_libraries(你的目标 PRIVATE base)
```

链接后即可 `#include "base/types.h"` 和 `#include "base/EventBus.h"`。

> **include 解析归属（勿误解）**：`#include "base/xxx.h"` 这种带 `base/` 前缀的写法靠的是**工程根的全局 include 路径**（根 `CMakeLists.txt` 的 `include_directories(${CMAKE_SOURCE_DIR})`），并非 base 库导出；base 库自身 `target_include_directories(... PUBLIC base/)` 导出的是 `base/` 目录本身（即不带前缀的 `#include "EventBus.h"` 也解析）。本工程内所有目标都吃到根路径全局 include，故两种写法皆可用。

---

## 6. 文件清单

```
base/
├── types.h        # 公共类型（Result/Pose/Event/...），命名空间 Scanner
├── EventBus.h     # 事件总线声明，命名空间 Scanner::infra
├── EventBus.cpp   # 事件总线实现
└── CMakeLists.txt # 静态库 base 构建配置
```
