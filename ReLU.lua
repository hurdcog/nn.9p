-- SPDX-License-Identifier: MIT
-- ReLU.lua - Rectified Linear Unit activation

local Module = require('Module')

local ReLU = setmetatable({}, {__index = Module})
ReLU.__index = ReLU

function ReLU.new()
    local self = setmetatable(Module.new(), ReLU)
    self._type = 'ReLU'
    return self
end

-- Forward pass: max(0, x)
function ReLU:updateOutput(input)
    self.output = {}
    for i = 1, #input do
        self.output[i] = math.max(0, input[i])
    end
    return self.output
end

-- Backward pass: gradient is 1 if input > 0, else 0
function ReLU:updateGradInput(input, gradOutput)
    self.gradInput = {}
    for i = 1, #input do
        self.gradInput[i] = (input[i] > 0) and gradOutput[i] or 0
    end
    return self.gradInput
end

return ReLU
