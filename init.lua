-- SPDX-License-Identifier: MIT
-- init.lua - Neural Network library for Lua
-- Based on torch/nn (https://github.com/torch/nn)

local nn = {}

-- Core classes
nn.Module = require('Module')
nn.Container = require('Container')

-- Containers
nn.Sequential = require('Sequential')
nn.Parallel = require('Parallel')
nn.Concat = require('Concat')

-- Layers
nn.Linear = require('Linear')
nn.Identity = require('Identity')
nn.Reshape = require('Reshape')

-- Activations
nn.ReLU = require('ReLU')
nn.Tanh = require('Tanh')
nn.Sigmoid = require('Sigmoid')

-- Criterions (Loss functions)
nn.Criterion = require('Criterion')
nn.MSECriterion = require('MSECriterion')

return nn
