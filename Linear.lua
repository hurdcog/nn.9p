-- SPDX-License-Identifier: MIT
-- Linear.lua - Fully connected linear transformation

local Module = require('Module')

local Linear = setmetatable({}, {__index = Module})
Linear.__index = Linear

function Linear.new(inputSize, outputSize)
    local self = setmetatable(Module.new(), Linear)
    self.inputSize = inputSize
    self.outputSize = outputSize
    self._type = 'Linear'
    
    -- Initialize weights and biases with small random values
    self.weight = {}
    self.bias = {}
    self.gradWeight = {}
    self.gradBias = {}
    
    -- Weight matrix: outputSize x inputSize
    for i = 1, outputSize do
        self.weight[i] = {}
        self.gradWeight[i] = {}
        for j = 1, inputSize do
            self.weight[i][j] = (math.random() - 0.5) * 0.1
            self.gradWeight[i][j] = 0
        end
    end
    
    -- Bias vector: outputSize
    for i = 1, outputSize do
        self.bias[i] = (math.random() - 0.5) * 0.1
        self.gradBias[i] = 0
    end
    
    return self
end

-- Forward pass: output = weight * input + bias
function Linear:updateOutput(input)
    self.output = {}
    for i = 1, self.outputSize do
        local sum = self.bias[i]
        for j = 1, self.inputSize do
            sum = sum + self.weight[i][j] * input[j]
        end
        self.output[i] = sum
    end
    return self.output
end

-- Backward pass: compute gradInput
function Linear:updateGradInput(input, gradOutput)
    self.gradInput = {}
    for j = 1, self.inputSize do
        local sum = 0
        for i = 1, self.outputSize do
            sum = sum + self.weight[i][j] * gradOutput[i]
        end
        self.gradInput[j] = sum
    end
    return self.gradInput
end

-- Accumulate parameter gradients
function Linear:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    
    -- Gradient w.r.t. weight
    for i = 1, self.outputSize do
        for j = 1, self.inputSize do
            self.gradWeight[i][j] = self.gradWeight[i][j] + scale * gradOutput[i] * input[j]
        end
    end
    
    -- Gradient w.r.t. bias
    for i = 1, self.outputSize do
        self.gradBias[i] = self.gradBias[i] + scale * gradOutput[i]
    end
end

-- Zero out gradients
function Linear:zeroGradParameters()
    for i = 1, self.outputSize do
        self.gradBias[i] = 0
        for j = 1, self.inputSize do
            self.gradWeight[i][j] = 0
        end
    end
end

-- Get parameters
function Linear:parameters()
    return {{self.weight, self.bias}, {self.gradWeight, self.gradBias}}
end

-- String representation
function Linear:__tostring()
    return 'nn.Linear(' .. self.inputSize .. ' -> ' .. self.outputSize .. ')'
end

return Linear
