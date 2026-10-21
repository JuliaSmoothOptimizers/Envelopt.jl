module Envelopt

using LinearAlgebra
using Logging
using Printf

using ProximalOperators

using ADNLPModels
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

# Extensions
export KnitroEnveloptSubSolver

function KnitroEnveloptSubSolver(args...; kwargs...)
  error(
    "Knitro support is not loaded. Install and load KNITRO.jl and NLPModelsKnitro.jl to use KnitroEnveloptSubSolver.",
  )
end

export IPOPTEnveloptSubSolver

function IPOPTEnveloptSubSolver(args...; kwargs...)
  error(
    "IPOPT support is not loaded. Install and load NLPModelsIpopt.jl to use IPOPTEnveloptSubSolver.",
  )
end

export TronEnveloptSubSolver, TrunkEnveloptSubSolver

function TronEnveloptSubSolver(args...; kwargs...)
  error(
    "TRON support is not loaded. Install and load JSOSolvers.jl to use TronEnveloptSubSolver.",
  )
end

function TrunkEnveloptSubSolver(args...; kwargs...)
  error(
    "TRUNK support is not loaded. Install and load JSOSolvers.jl to use TrunkEnveloptSubSolver.",
  )
end

end
