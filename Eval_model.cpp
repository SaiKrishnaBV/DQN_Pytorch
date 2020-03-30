#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

const std::string pretrained_model_path = "/../.../..../traced_DQN_model.pth";
//Replace "/../.../..../traced_DQN_model.pth" with the absolute path to the traced model of the trained neural network
torch::jit::script::Module DQN;

DQN = torch::jit::load(pretrained_model_path);

auto sample_input = torch::randn({1, 4}); //Replace this with the observation from environment
std::vector<torch::jit::IValue> inputs{sample_input};

auto output = DQN.forward(inputs).toTensor();
int action = output.argmax(1).item().toInt();  //Will return the index 0 - Left, 1 - Right
