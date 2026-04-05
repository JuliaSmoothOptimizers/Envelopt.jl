module Envelopt

using LinearAlgebra
using Logging
using Printf

using ProximalOperators

using ADNLPModels
using JSOSolvers
using LinearOperators
using NCL
using NLPModels
using NLPModelsModifiers
using SolverCore

include("envelopt_model.jl")
include("envelopt_solver.jl")

include("eq_feasibility.jl")
include("remove_eqs.jl")

include("penalty.jl")

end
