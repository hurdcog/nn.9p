-- SPDX-License-Identifier: MIT
-- init.lua - Neural Network library for Lua
-- Based on torch/nn (https://github.com/torch/nn)

local nn = {}

-- Core classes
nn.Module = require('Module')
nn.Container = require('Container')

-- Containers
nn.Sequential = require('Sequential')

-- Layers
nn.Linear = require('Linear')

-- Activations
nn.ReLU = require('ReLU')
nn.Tanh = require('Tanh')
nn.Sigmoid = require('Sigmoid')

-- Criterions (Loss functions)
nn.Criterion = require('Criterion')
nn.MSECriterion = require('MSECriterion')

return nn
