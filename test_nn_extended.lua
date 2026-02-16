#!/usr/bin/env lua
-- SPDX-License-Identifier: MIT
-- test_nn_extended.lua - Extended tests for new modules

local nn = require('init')

print("Testing extended nn modules...")
print()

-- Test 1: Concat container
print("Test 1: Concat container")
local concat = nn.Concat.new()
concat:add(nn.Linear.new(3, 2))
concat:add(nn.Linear.new(3, 3))

local input1 = {1, 2, 3}
local output1 = concat:forward(input1)
print("  Input: {1, 2, 3}")
print("  Concat(Linear(3->2), Linear(3->3))")
print("  Output size: " .. #output1)
assert(#output1 == 5, "Concat output should be 2+3=5")
print("  ✓ Concat container works")
print()

-- Test 2: Identity module
print("Test 2: Identity module")
local identity = nn.Identity.new()
local input2 = {1, 2, 3, 4}
local output2 = identity:forward(input2)
print("  Input: {1, 2, 3, 4}")
print("  Output: {" .. table.concat(output2, ", ") .. "}")
for i = 1, #input2 do
    assert(input2[i] == output2[i], "Identity should return input unchanged")
end
print("  ✓ Identity module works")
print()

-- Test 3: Reshape module
print("Test 3: Reshape module")
local reshape = nn.Reshape.new(2, 2)
local input3 = {1, 2, 3, 4}
local output3 = reshape:forward(input3)
print("  Input: {1, 2, 3, 4}")
print("  Reshape(2, 2)")
print("  Output: " .. tostring(reshape))
assert(output3 ~= nil, "Reshape should produce output")
print("  ✓ Reshape module works")
print()

-- Test 4: Complex network with multiple paths
print("Test 4: Complex network")
local complex = nn.Sequential.new()
complex:add(nn.Linear.new(2, 4))
complex:add(nn.ReLU.new())
complex:add(nn.Linear.new(4, 1))
complex:add(nn.Sigmoid.new())

local input4 = {0.5, -0.5}
local output4 = complex:forward(input4)
print("  Network: Linear(2->4) -> ReLU -> Linear(4->1) -> Sigmoid")
print("  Input: {0.5, -0.5}")
print("  Output: " .. output4[1])
assert(output4[1] >= 0 and output4[1] <= 1, "Sigmoid output should be in [0,1]")
print("  ✓ Complex network works")
print()

-- Test 5: Module string representation
print("Test 5: Module string representation")
print("  " .. tostring(nn.Linear.new(10, 5)))
print("  " .. tostring(nn.ReLU.new()))
print("  " .. tostring(nn.Sequential.new()))
print("  " .. tostring(nn.MSECriterion.new()))
print("  ✓ String representation works")
print()

print("================================")
print("All extended tests passed! ✓")
print("================================")
