-- SPDX-License-Identifier: MIT
-- Sigmoid.lua - Sigmoid activation function

local Module = require('Module')

local Sigmoid = setmetatable({}, {__index = Module})
Sigmoid.__index = Sigmoid

function Sigmoid.new()
    local self = setmetatable(Module.new(), Sigmoid)
    self._type = 'Sigmoid'
    return self
end

-- Forward pass: 1 / (1 + exp(-x))
function Sigmoid:updateOutput(input)
    self.output = {}
    for i = 1, #input do
        self.output[i] = 1 / (1 + math.exp(-input[i]))
    end
    return self.output
end

-- Backward pass: gradient is sigmoid(x) * (1 - sigmoid(x))
function Sigmoid:updateGradInput(input, gradOutput)
    self.gradInput = {}
    for i = 1, #input do
        local sigmoid_val = self.output[i]
        self.gradInput[i] = sigmoid_val * (1 - sigmoid_val) * gradOutput[i]
    end
    return self.gradInput
end

return Sigmoid
