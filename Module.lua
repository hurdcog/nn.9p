-- SPDX-License-Identifier: MIT
-- Module.lua - Base class for all neural network modules

local Module = {}
Module.__index = Module

function Module.new()
    local self = setmetatable({}, Module)
    self.output = nil
    self.gradInput = nil
    self._type = 'Module'
    return self
end

-- Forward pass: compute output from input
function Module:updateOutput(input)
    return input
end

-- Backward pass: compute gradient w.r.t. input
function Module:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end

-- Accumulate gradient w.r.t. parameters (if any)
function Module:accGradParameters(input, gradOutput, scale)
    -- Default: no parameters to update
end

-- Forward pass (calls updateOutput)
function Module:forward(input)
    return self:updateOutput(input)
end

-- Backward pass (calls updateGradInput and accGradParameters)
function Module:backward(input, gradOutput, scale)
    scale = scale or 1
    self:updateGradInput(input, gradOutput)
    self:accGradParameters(input, gradOutput, scale)
    return self.gradInput
end

-- Zero out gradients
function Module:zeroGradParameters()
    -- Default: no parameters
end

-- Get learnable parameters and their gradients
function Module:parameters()
    return {}
end

-- Get the module type
function Module:type()
    return self._type
end

-- String representation
function Module:__tostring()
    return 'nn.' .. self._type
end

return Module
