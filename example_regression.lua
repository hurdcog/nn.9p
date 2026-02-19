#!/usr/bin/env lua
-- SPDX-License-Identifier: MIT
-- example_regression.lua - Simple regression example

local nn = require('init')

print("Simple Regression Example")
print("==========================")
print()
print("Task: Learn the function y = 2*x + 3")
print()

-- Training data: y = 2*x + 3
local X = {{1}, {2}, {3}, {4}, {5}}
local Y = {{5}, {7}, {9}, {11}, {13}}

-- Build simple network
local net = nn.Sequential.new()
local layer = nn.Linear.new(1, 1)
net:add(layer)

print("Network: Linear(1->1) - learning y = w*x + b")
print()

local criterion = nn.MSECriterion.new()
local lr = 0.01

print("Training for 100 epochs...")

for epoch = 1, 100 do
    local totalLoss = 0
    
    for i = 1, #X do
        -- Forward
        local output = net:forward(X[i])
        local loss = criterion:forward(output, Y[i])
        totalLoss = totalLoss + loss
        
        -- Backward
        net:zeroGradParameters()
        local gradOutput = criterion:backward(output, Y[i])
        net:backward(X[i], gradOutput)
        
        -- Update
        layer.weight[1][1] = layer.weight[1][1] - lr * layer.gradWeight[1][1]
        layer.bias[1] = layer.bias[1] - lr * layer.gradBias[1]
    end
    
    if epoch % 20 == 0 then
        print(string.format("Epoch %3d, Loss: %.6f, w=%.4f, b=%.4f",
            epoch, totalLoss / #X, layer.weight[1][1], layer.bias[1]))
    end
end

print()
print("Final parameters:")
print(string.format("  Weight (w): %.4f (target: 2.0)", layer.weight[1][1]))
print(string.format("  Bias (b):   %.4f (target: 3.0)", layer.bias[1]))
print()

print("Testing:")
for i = 1, #X do
    local output = net:forward(X[i])
    print(string.format("  x=%d -> predicted=%.2f, actual=%d",
        X[i][1], output[1], Y[i][1]))
end

print()
print("✓ Regression example complete!")
