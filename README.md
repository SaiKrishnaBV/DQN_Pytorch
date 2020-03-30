# DQN_Pytorch

The file Qget_data.py contains the code to save the experience of the agent with the environment, which is further used to train the neural network.

The file Qtrain_network.py contains the code to train a neural network the data for which can be loaded using a .npy file

The file Qnetwork_eval.py contains the code to evaluate the neural network. 

The file Model_Py_to_C.py contains the code to convert a trained model in Python to a compiler convertible form. So that the evaluation can be using Libtorch (Pytorch C++ API).

The file Eval_model.cpp contains the code to evaluate the traced model (Python model converted to compiler compatible form).
