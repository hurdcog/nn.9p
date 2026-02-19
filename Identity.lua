-- SPDX-License-Identifier: MIT
-- Identity.lua - Identity transformation (pass-through)

local Module = require('Module')

local Identity = setmetatable({}, {__index = Module})
Identity.__index = Identity

function Identity.new()
    local self = setmetatable(Module.new(), Identity)
    self._type = 'Identity'
    return self
end

-- Forward pass: return input as-is
function Identity:updateOutput(input)
    self.output = input
    return self.output
end

-- Backward pass: return gradOutput as-is
function Identity:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end

return Identity
