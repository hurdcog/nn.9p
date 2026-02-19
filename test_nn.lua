#!/usr/bin/env lua
-- SPDX-License-Identifier: MIT
-- test_nn.lua - Test suite for the nn library

local nn = require('init')

print("Testing nn library...")
print()

-- Test 1: Linear module
print("Test 1: Linear module")
local linear = nn.Linear.new(3, 2)
local input = {1, 2, 3}
local output = linear:forward(input)
print("  Input: {1, 2, 3}")
print("  Output: {" .. output[1] .. ", " .. output[2] .. "}")
assert(#output == 2, "Linear output size should be 2")
print("  ✓ Linear module works")
print()

-- Test 2: ReLU activation
print("Test 2: ReLU activation")
local relu = nn.ReLU.new()
local input2 = {-1, 0, 1, 2}
local output2 = relu:forward(input2)
print("  Input: {-1, 0, 1, 2}")
print("  Output: {" .. output2[1] .. ", " .. output2[2] .. ", " .. output2[3] .. ", " .. output2[4] .. "}")
assert(output2[1] == 0, "ReLU(-1) should be 0")
assert(output2[2] == 0, "ReLU(0) should be 0")
assert(output2[3] == 1, "ReLU(1) should be 1")
assert(output2[4] == 2, "ReLU(2) should be 2")
print("  ✓ ReLU activation works")
print()

-- Test 3: Tanh activation
print("Test 3: Tanh activation")
local tanh = nn.Tanh.new()
local input3 = {0, 1, -1}
local output3 = tanh:forward(input3)
print("  Input: {0, 1, -1}")
print("  Output: {" .. output3[1] .. ", " .. output3[2] .. ", " .. output3[3] .. "}")
assert(math.abs(output3[1] - 0) < 0.0001, "tanh(0) should be ~0")
assert(math.abs(output3[2] - math.tanh(1)) < 0.0001, "tanh(1) should match")
print("  ✓ Tanh activation works")
print()

-- Test 4: Sigmoid activation
print("Test 4: Sigmoid activation")
local sigmoid = nn.Sigmoid.new()
local input4 = {0, 1, -1}
local output4 = sigmoid:forward(input4)
print("  Input: {0, 1, -1}")
print("  Output: {" .. output4[1] .. ", " .. output4[2] .. ", " .. output4[3] .. "}")
assert(math.abs(output4[1] - 0.5) < 0.0001, "sigmoid(0) should be ~0.5")
print("  ✓ Sigmoid activation works")
print()

-- Test 5: Sequential container
print("Test 5: Sequential container")
local mlp = nn.Sequential.new()
mlp:add(nn.Linear.new(2, 3))
mlp:add(nn.ReLU.new())
mlp:add(nn.Linear.new(3, 1))

local input5 = {1, 2}
local output5 = mlp:forward(input5)
print("  Network: Linear(2->3) -> ReLU -> Linear(3->1)")
print("  Input: {1, 2}")
print("  Output: {" .. output5[1] .. "}")
assert(#output5 == 1, "Sequential output should have size 1")
print("  ✓ Sequential container works")
print()

-- Test 6: MSE Criterion
print("Test 6: MSE Criterion")
local criterion = nn.MSECriterion.new()
local pred = {1, 2, 3}
local target = {1.5, 2.5, 3.5}
local loss = criterion:forward(pred, target)
local expectedLoss = ((0.5*0.5) + (0.5*0.5) + (0.5*0.5)) / 3
print("  Prediction: {1, 2, 3}")
print("  Target: {1.5, 2.5, 3.5}")
print("  Loss: " .. loss)
print("  Expected: " .. expectedLoss)
assert(math.abs(loss - expectedLoss) < 0.0001, "MSE loss should match expected")
print("  ✓ MSE Criterion works")
print()

-- Test 7: Backward pass
print("Test 7: Backward pass")
local net = nn.Sequential.new()
net:add(nn.Linear.new(2, 2))
net:add(nn.ReLU.new())

local input7 = {1, 1}
local target7 = {2, 2}
local output7 = net:forward(input7)
local crit = nn.MSECriterion.new()
local loss7 = crit:forward(output7, target7)
local gradOutput7 = crit:backward(output7, target7)
local gradInput7 = net:backward(input7, gradOutput7)

print("  Input: {1, 1}")
print("  Target: {2, 2}")
print("  Loss: " .. loss7)
print("  GradInput: {" .. gradInput7[1] .. ", " .. gradInput7[2] .. "}")
assert(#gradInput7 == 2, "GradInput should have size 2")
print("  ✓ Backward pass works")
print()

-- Test 8: Parameter updates (simple gradient descent)
print("Test 8: Parameter updates")
local simple_net = nn.Linear.new(1, 1)
simple_net.weight[1][1] = 2.0  -- Set initial weight
simple_net.bias[1] = 0.0        -- Set initial bias

local x = {3.0}
local y = {10.0}  -- Target: 2*3 + 4 = 10, so ideal bias = 4

-- Forward pass
local pred = simple_net:forward(x)
print("  Initial prediction: " .. pred[1] .. " (target: " .. y[1] .. ")")

-- Compute loss
local loss_fn = nn.MSECriterion.new()
local loss_val = loss_fn:forward(pred, y)
print("  Initial loss: " .. loss_val)

-- Backward pass
local grad = loss_fn:backward(pred, y)
simple_net:backward(x, grad)

-- Check gradients exist
print("  Weight gradient: " .. simple_net.gradWeight[1][1])
print("  Bias gradient: " .. simple_net.gradBias[1])
assert(simple_net.gradWeight[1][1] ~= 0 or simple_net.gradBias[1] ~= 0, "Gradients should be computed")
print("  ✓ Parameter gradients computed")
print()

print("================================")
print("All tests passed! ✓")
print("================================")
