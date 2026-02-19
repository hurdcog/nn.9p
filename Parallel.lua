-- SPDX-License-Identifier: MIT
-- Parallel.lua - Parallel container that processes input through multiple modules

local Container = require('Container')

local Parallel = setmetatable({}, {__index = Container})
Parallel.__index = Parallel

function Parallel.new(inputDimension, outputDimension)
    local self = setmetatable(Container.new(), Parallel)
    self.inputDimension = inputDimension
    self.outputDimension = outputDimension
    self._type = 'Parallel'
    return self
end

-- Forward pass: apply each module to corresponding slice of input, concatenate outputs
function Parallel:updateOutput(input)
    self.output = {}
    
    -- Apply each module and collect outputs
    for i = 1, #self.modules do
        local moduleOutput = self.modules[i]:forward(input[i])
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

-- Backward pass
function Parallel:updateGradInput(input, gradOutput)
    self.gradInput = {}
    local currentPos = 1
    
    for i = 1, #self.modules do
        local moduleInput = input[i]
        -- Determine size of gradient for this module
        local moduleOutputSize
        if self.modules[i].output then
            moduleOutputSize = type(self.modules[i].output) == 'table' and #self.modules[i].output or 1
        else
            moduleOutputSize = 1
        end
        
        -- Extract gradient slice
        local moduleGradOutput = {}
        for j = 1, moduleOutputSize do
            moduleGradOutput[j] = gradOutput[currentPos]
            currentPos = currentPos + 1
        end
        
        -- Backward through module
        self.gradInput[i] = self.modules[i]:backward(moduleInput, moduleGradOutput)
    end
    
    return self.gradInput
end

return Parallel
