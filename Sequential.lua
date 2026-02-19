-- SPDX-License-Identifier: MIT
-- Sequential.lua - Sequential container that chains modules

local Container = require('Container')

local Sequential = setmetatable({}, {__index = Container})
Sequential.__index = Sequential

function Sequential.new()
    local self = setmetatable(Container.new(), Sequential)
    self._type = 'Sequential'
    return self
end

-- Forward pass through all modules in sequence
function Sequential:updateOutput(input)
    local currentOutput = input
    for i = 1, #self.modules do
        currentOutput = self.modules[i]:forward(currentOutput)
    end
    self.output = currentOutput
    return self.output
end

-- Backward pass through all modules in reverse
function Sequential:updateGradInput(input, gradOutput)
    local currentGradOutput = gradOutput
    for i = #self.modules, 1, -1 do
        local currentModule = self.modules[i]
        local previousOutput = (i == 1) and input or self.modules[i-1].output
        currentGradOutput = currentModule:backward(previousOutput, currentGradOutput)
    end
    self.gradInput = currentGradOutput
    return self.gradInput
end

-- Accumulate gradients (handled by backward in each module)
function Sequential:accGradParameters(input, gradOutput, scale)
    -- Already handled in updateGradInput via backward()
end

return Sequential
