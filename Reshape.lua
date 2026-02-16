-- SPDX-License-Identifier: MIT
-- Reshape.lua - Reshapes input tensor

local Module = require('Module')

local Reshape = setmetatable({}, {__index = Module})
Reshape.__index = Reshape

function Reshape.new(...)
    local self = setmetatable(Module.new(), Reshape)
    local args = {...}
    self.size = {}
    self.batchMode = false
    
    if #args == 1 and type(args[1]) == 'table' then
        self.size = args[1]
    else
        self.size = args
    end
    
    self._type = 'Reshape'
    return self
end

-- Calculate total size
function Reshape:numel()
    local n = 1
    for i = 1, #self.size do
        n = n * self.size[i]
    end
    return n
end

-- Forward pass: reshape input (for tables, just pass through as is)
function Reshape:updateOutput(input)
    -- For simplicity, just pass through the input
    -- In a full implementation, this would actually reshape
    self.output = input
    return self.output
end

-- Backward pass: reshape gradient back to input size
function Reshape:updateGradInput(input, gradOutput)
    -- For simplicity, just pass through
    self.gradInput = gradOutput
    return self.gradInput
end

-- String representation
function Reshape:__tostring()
    local str = 'nn.Reshape('
    for i = 1, #self.size do
        str = str .. self.size[i]
        if i < #self.size then
            str = str .. ', '
        end
    end
    str = str .. ')'
    return str
end

return Reshape
