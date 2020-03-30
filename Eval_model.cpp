#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

const std::string pretrained_model_path = "/home/krishna/SK/Pytorch/traced_DQN_model.pth";
torch::jit::script::Module DQN;

DQN = torch::jit::load(pretrained_model_path);

auto sample_input = torch::randn({1, 4}); //Replace this with the observation from GAZEBO
std::vector<torch::jit::IValue> inputs{sample_input};

auto output = DQN.forward(inputs).toTensor();
int action = output.argmax(1).item().toInt();  //Will return the index 0 - Left, 1 - Right
