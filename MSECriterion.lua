-- SPDX-License-Identifier: MIT
-- MSECriterion.lua - Mean Squared Error loss

local Criterion = require('Criterion')

local MSECriterion = setmetatable({}, {__index = Criterion})
MSECriterion.__index = MSECriterion

function MSECriterion.new()
    local self = setmetatable(Criterion.new(), MSECriterion)
    self._type = 'MSECriterion'
    self.sizeAverage = true
    return self
end

-- Forward pass: compute MSE loss
function MSECriterion:updateOutput(input, target)
    local sum = 0
    local n = #input
    
    for i = 1, n do
        local diff = input[i] - target[i]
        sum = sum + diff * diff
    end
    
    if self.sizeAverage then
        self.output = sum / n
    else
        self.output = sum
    end
    
    return self.output
end

-- Backward pass: gradient is 2 * (input - target) / n
function MSECriterion:updateGradInput(input, target)
    local n = #input
    self.gradInput = {}
    
    for i = 1, n do
        local grad = 2 * (input[i] - target[i])
        if self.sizeAverage then
            grad = grad / n
        end
        self.gradInput[i] = grad
    end
    
    return self.gradInput
end

return MSECriterion
