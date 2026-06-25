export R2EnveloptSubSolver

using RegularizedProblems, RegularizedOptimization

# R2 subproblem solver
mutable struct R2EnveloptSubSolver <: AbstractEnveloptSubSolver
  solver::RegularizedOptimization.R2Solver
  stats::GenericExecutionStats
  name::String
end

# ... constructor
function R2EnveloptSubSolver(env_model::EnveloptNLPModel)
  @debug "initializing R2 subproblem solver"
  reg_nlp = RegularizedNLPModel(env_model, env_model.g)
  solver = RegularizedOptimization.R2Solver(reg_nlp, m_monotone = 10)
  stats = RegularizedExecutionStats(reg_nlp)
  return R2EnveloptSubSolver(solver, stats, "R2")
end

# TODO add fixed options

# ... solve
function (sub::R2EnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  outer_iter::Int;
  tol::Float64 = 1.0e-6,
  kwargs...,
)
  reg_nlp = RegularizedNLPModel(env_model, env_model.g)
  RegularizedOptimization.solve!(
    sub.solver,
    reg_nlp,
    sub.stats,
    x = x0,
    atol = tol,
    rtol = 0.0,
    max_iter = 10_000_000,
    max_time = 600.0,
  )
  return sub.stats
end
