-- SPDX-License-Identifier: MIT
-- Criterion.lua - Base class for loss functions

local Criterion = {}
Criterion.__index = Criterion

function Criterion.new()
    local self = setmetatable({}, Criterion)
    self.output = 0
    self.gradInput = nil
    self._type = 'Criterion'
    return self
end

-- Forward pass: compute loss
function Criterion:forward(input, target)
    return self:updateOutput(input, target)
end

-- Backward pass: compute gradient
function Criterion:backward(input, target)
    return self:updateGradInput(input, target)
end

-- Update loss output
function Criterion:updateOutput(input, target)
    return 0
end

-- Update gradient
function Criterion:updateGradInput(input, target)
    return self.gradInput
end

-- String representation
function Criterion:__tostring()
    return 'nn.' .. self._type
end

return Criterion
