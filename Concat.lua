-- SPDX-License-Identifier: MIT
-- Concat.lua - Concatenates outputs of multiple modules along a dimension

local Container = require('Container')

local Concat = setmetatable({}, {__index = Container})
Concat.__index = Concat

function Concat.new(dimension)
    local self = setmetatable(Container.new(), Concat)
    self.dimension = dimension or 1
    self._type = 'Concat'
    return self
end

-- Forward pass: apply all modules to same input, concatenate outputs
function Concat:updateOutput(input)
    self.output = {}
    
    -- Apply each module to the same input
    for i = 1, #self.modules do
        local moduleOutput = self.modules[i]:forward(input)
        
        -- Concatenate outputs
        if type(moduleOutput) == 'table' then
            for j = 1, #moduleOutput do
                table.insert(self.output, moduleOutput[j])
            end
        else
            table.insert(self.output, moduleOutput)
        end
    end
    
    return self.output
end

-- Backward pass: sum gradients from all modules
function Concat:updateGradInput(input, gradOutput)
    local currentPos = 1
    
    -- Initialize gradInput
    if type(input) == 'table' then
        self.gradInput = {}
        for i = 1, #input do
            self.gradInput[i] = 0
        end
    else
        self.gradInput = 0
    end
    
    -- Accumulate gradients from each module
    for i = 1, #self.modules do
        -- Determine output size of this module
        local moduleOutputSize
        if self.modules[i].output then
            moduleOutputSize = type(self.modules[i].output) == 'table' and #self.modules[i].output or 1
        else
            moduleOutputSize = 1
        end
        
        -- Extract gradient slice for this module
        local moduleGradOutput = {}
        for j = 1, moduleOutputSize do
            moduleGradOutput[j] = gradOutput[currentPos]
            currentPos = currentPos + 1
        end
        
        -- Backward through module
        local moduleGradInput = self.modules[i]:backward(input, moduleGradOutput)
        
        -- Accumulate gradients
        if type(moduleGradInput) == 'table' then
            for j = 1, #moduleGradInput do
                self.gradInput[j] = self.gradInput[j] + moduleGradInput[j]
            end
        else
            self.gradInput = self.gradInput + moduleGradInput
        end
    end
    
    return self.gradInput
end

return Concat
