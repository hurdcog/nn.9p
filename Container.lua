-- SPDX-License-Identifier: MIT
-- Container.lua - Base class for module containers

local Module = require('Module')

local Container = setmetatable({}, {__index = Module})
Container.__index = Container

function Container.new()
    local self = setmetatable(Module.new(), Container)
    self.modules = {}
    self._type = 'Container'
    return self
end

-- Add a module to the container
function Container:add(module)
    table.insert(self.modules, module)
    return self
end

-- Get a module by index
function Container:get(index)
    return self.modules[index]
end

-- Get the number of modules
function Container:size()
    return #self.modules
end

-- Zero gradients for all modules
function Container:zeroGradParameters()
    for i = 1, #self.modules do
        self.modules[i]:zeroGradParameters()
    end
end

-- Get all parameters from contained modules
function Container:parameters()
    local params = {}
    local gradParams = {}
    
    for i = 1, #self.modules do
        local p, gp = table.unpack(self.modules[i]:parameters())
        if p then
            for j = 1, #p do
                table.insert(params, p[j])
                table.insert(gradParams, gp[j])
            end
        end
    end
    
    return {params, gradParams}
end

-- String representation
function Container:__tostring()
    local str = 'nn.' .. self._type .. ' {\n'
    for i = 1, #self.modules do
        str = str .. '  [' .. i .. ']: ' .. tostring(self.modules[i]) .. '\n'
    end
    str = str .. '}'
    return str
end

return Container
