#include "include/api/model.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) return 1;
  auto context = std::make_shared<mindspore::Context>();
  context->SetThreadNum(2);
  context->MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());
  mindspore::Model model;
  auto result = model.Build(argv[1], mindspore::kMindIR_Lite, context);
  if (result != mindspore::kSuccess) { std::cerr << result.ToString(); return 2; }
  auto inputs = model.GetInputs();
  if (inputs.size() != 2) return 3;
  std::ifstream examples(argv[2], std::ios::binary);
  if (!examples) return 4;
  std::array<int32_t, 128> data{};
  while (examples.read(reinterpret_cast<char *>(data.data()), sizeof(data))) {
    for (auto &tensor : inputs) {
      const auto name = tensor.Name();
      if ((name != "input_ids" && name != "attention_mask") ||
          tensor.DataType() != mindspore::DataType::kNumberTypeInt32 ||
          tensor.Shape() != std::vector<int64_t>({1,64}) ||
          tensor.DataSize() != 64 * sizeof(int32_t)) {
        std::cerr << name << " dtype " << static_cast<int>(tensor.DataType()) << " bytes " << tensor.DataSize();
        return 5;
      }
      std::memcpy(tensor.MutableData(), data.data() + (name == "input_ids" ? 0 : 64), tensor.DataSize());
    }
    std::vector<mindspore::MSTensor> outputs;
    result = model.Predict(inputs, &outputs);
    if (result != mindspore::kSuccess || outputs.size() != 1 ||
        outputs[0].DataType() != mindspore::DataType::kNumberTypeFloat32 ||
        outputs[0].ElementNum() != 5) { std::cerr << result.ToString(); return 6; }
    auto values = static_cast<const float *>(outputs[0].Data().get());
    for (int i = 0; i < 5; ++i) std::cout << std::setprecision(9) << values[i] << ' ';
    std::cout << '\n';
  }
  return examples.eof() && examples.gcount() == 0 ? 0 : 7;
}
