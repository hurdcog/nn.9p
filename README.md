nn.9p - Neural Network Library for Lua
========================================

A neural network library for Lua, inspired by [torch/nn](https://github.com/torch/nn).

This repository combines:
- **nn**: A modular neural network library
- **9p**: A 9P client written in Lua (depends on [luadata](https://github.com/lneto/luadata))

## Neural Network (nn) Usage

### Quick Start

```lua
local nn = require('init')

-- Create a simple network
local net = nn.Sequential.new()
net:add(nn.Linear.new(10, 20))    -- Linear layer: 10 -> 20
net:add(nn.ReLU.new())             -- ReLU activation
net:add(nn.Linear.new(20, 10))    -- Linear layer: 20 -> 10
net:add(nn.Tanh.new())             -- Tanh activation

-- Forward pass
local input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
local output = net:forward(input)

-- Compute loss
local criterion = nn.MSECriterion.new()
local target = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
local loss = criterion:forward(output, target)

-- Backward pass
local gradOutput = criterion:backward(output, target)
net:backward(input, gradOutput)
```

### Available Modules

#### Containers
- **Sequential**: Chain modules in sequence
- **Parallel**: Process inputs through multiple modules in parallel
- **Concat**: Concatenate outputs from multiple modules

#### Layers
- **Linear**: Fully connected layer (weight matrix + bias)
- **Identity**: Pass-through layer
- **Reshape**: Reshape input tensors

#### Activations
- **ReLU**: Rectified Linear Unit
- **Tanh**: Hyperbolic tangent
- **Sigmoid**: Sigmoid function

#### Loss Functions
- **MSECriterion**: Mean Squared Error

### Examples

See `example_regression.lua` for a complete training example:
```bash
lua example_regression.lua
```

### Testing

Run the test suite:
```bash
lua test_nn.lua
lua test_nn_extended.lua
```

### Architecture

All modules inherit from `Module`, which provides:
- `forward(input)`: Forward pass computation
- `backward(input, gradOutput)`: Backward pass (gradient computation)
- `parameters()`: Get learnable parameters
- `zeroGradParameters()`: Zero out gradients

## 9P Client TODO

* auth

