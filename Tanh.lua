-- SPDX-License-Identifier: MIT
-- Tanh.lua - Hyperbolic tangent activation

local Module = require('Module')

local Tanh = setmetatable({}, {__index = Module})
Tanh.__index = Tanh

function Tanh.new()
    local self = setmetatable(Module.new(), Tanh)
    self._type = 'Tanh'
    return self
end

-- Forward pass: tanh(x)
function Tanh:updateOutput(input)
    self.output = {}
    for i = 1, #input do
        self.output[i] = math.tanh(input[i])
    end
    return self.output
end

-- Backward pass: gradient is (1 - tanh(x)^2)
function Tanh:updateGradInput(input, gradOutput)
    self.gradInput = {}
    for i = 1, #input do
        local tanh_val = self.output[i]
        self.gradInput[i] = (1 - tanh_val * tanh_val) * gradOutput[i]
    end
    return self.gradInput
end

return Tanh
