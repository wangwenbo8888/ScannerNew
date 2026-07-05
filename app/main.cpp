#include <iostream>
#include "framework/common/quality_flag.h"
#include "framework/common/types.h"
#include "framework/ui/IView.h"
#include "framework/workflow/IWorkflow.h"
#include "framework/service/IService.h"
#include "framework/algorithm/operator_convention.h"
#include "framework/data/IDataStore.h"
#include "framework/hal/ICamera.h"
#include "framework/infra/EventBus.h"
#include "framework/crosscut/IAuth.h"
#include "sdk/IScannerSDK.h"

int main() {
    std::cout << "JEAMMWARE v0.1.0 skeleton\n";
    std::cout << "Scanner::QualityFlag Normal=" << static_cast<int>(Scanner::QualityFlag::Normal) << "\n";
    std::cout << "5 layers / 11 modules / sdk / app  —  skeleton OK\n";
    return 0;
}
