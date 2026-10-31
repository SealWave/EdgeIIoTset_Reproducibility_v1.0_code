This study examines quantized tree-based models and a lightweight neural network in their ability to predict and characterize the software resources of intrusion detection in resource-limited Internet of Things (IoT) applications.

The study compares a conventional decision tree with a depth-constrained compact decision tree, a uint8-quantized version of the compact tree, and a small multilayer perceptron (MLP) using a subset of the Edge-IIoTset cybersecurity dataset. The experimental approach consists of dataset auditing, the removal of data duplicates, feature selection, stratified training and testing, and reproducible evaluation on a shared held-out test set.

The manuscript provides classification accuracy, macro precision, macro recall, macro F1-score, confusion matrices, serialized model size and laptop software inference times. It discusses the trade-offs between classification performance, model size and numerical quantization.

The results show that the size reduction or integer representation of the model does not guarantee to improve classification accuracy and deployment efficiency. This study only addresses an evaluation conducted with a laptop setup and doesn't report direct measurements of the energy consumption in the IoT hardware or continuous deployment results on the device.

Keywords: IoT intrusion detection, Edge-IIoTset, TinyML, quantized decision trees, cybersecurity, edge computing, reproducible machine learning.
