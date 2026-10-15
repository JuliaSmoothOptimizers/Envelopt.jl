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

# Extensions
export KnitroEnveloptSubSolver, R2NEnveloptSubSolver, R2EnveloptSubSolver

function KnitroEnveloptSubSolver(args...; kwargs...)
  error(
    "Knitro support is not loaded. Install and load KNITRO.jl and NLPModelsKnitro.jl to use KnitroEnveloptSubSolver.",
  )
end

"""
    R2NEnveloptSubSolver(env_model::EnveloptNLPModel, regularizer; kwargs...)

An Envelopt subproblem solver based on R2N from RegularizedOptimization.jl.
Together with `envelopt`, it solves

    minimize f(x) + h(F(x)) + regularizer(x),

where `regularizer` is a proper, lower semi-continuous and prox-bounded function,
possibly nonconvex, supported by ShiftedProximalOperators.jl,
and `h` is the convex function of `env_model`.
Each subproblem

    minimize f(x) + h_μ(F(x) + μy) + regularizer(x)

is solved by R2N, which treats `regularizer` through its proximal operator.
`env_model` must not have constraints c(x) ∈ C; all constraints must be part of F.

This solver is defined in an extension: load RegularizedOptimization.jl and
RegularizedProblems.jl to use it.

## Keyword arguments

* `hessian_approximation::Symbol = :exact_f`: the Hessian model used by R2N:
  - `:exact_f`: ∇²f(x) + 0, i.e., the curvature of the envelope term is ignored;
  - `:lbfgs`, `:lsr1`: a quasi-Newton approximation of the Hessian of the whole smooth objective;
* `mem::Int = 5`: memory of the quasi-Newton approximation;
* `subsolver = R2Solver`, `m_monotone::Int = 1`: passed to `R2NSolver`;
* `selected = 1:nvar`: the variables to which `regularizer` applies.

Any other keyword argument is passed to R2N at each subproblem solve,
e.g., `max_iter`, `verbose` or `σmin`.
"""
function R2NEnveloptSubSolver(args...; kwargs...)
  error(
    "R2N support is not loaded. Install and load RegularizedOptimization.jl and RegularizedProblems.jl to use R2NEnveloptSubSolver.",
  )
end

"""
    R2EnveloptSubSolver(env_model::EnveloptNLPModel, regularizer; kwargs...)

An Envelopt subproblem solver based on R2 from RegularizedOptimization.jl,
a first-order quadratic regularization method that only uses the objective and gradient
of `env_model`.
It solves the same problems and subproblems as [`R2NEnveloptSubSolver`](@ref),
under the same assumptions.

This solver is defined in an extension: load RegularizedOptimization.jl and
RegularizedProblems.jl to use it.

## Keyword arguments

* `selected = 1:nvar`: the variables to which `regularizer` applies.

Any other keyword argument is passed to R2 at each subproblem solve,
e.g., `max_iter`, `verbose` or `σmin`.
"""
function R2EnveloptSubSolver(args...; kwargs...)
  error(
    "R2 support is not loaded. Install and load RegularizedOptimization.jl and RegularizedProblems.jl to use R2EnveloptSubSolver.",
  )
end

end
