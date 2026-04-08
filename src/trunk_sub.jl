export TrunkEnveloptSubSolver

using JSOSolvers, SolverCore

# Trunk subproblem solver
mutable struct TrunkEnveloptSubSolver <: AbstractEnveloptSubSolver
  solver::TrunkSolver
  stats::GenericExecutionStats
  name::String
end

# ... constructor
function TrunkEnveloptSubSolver(env_model::EnveloptNLPModel)
  @debug "initializing Trunk subproblem solver"
  unconstrained(env_model) || error("Trunk only supports unconstrained problems")
  solver = TrunkSolver(EnveloptLSR1Model(env_model))  # at this point, only the problem dimensions are used
  stats = GenericExecutionStats(env_model)
  return TrunkEnveloptSubSolver(solver, stats, "Trunk")
end

# ... solve
function (M::TrunkEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  args...;
  tol::Float64 = 1.0e-6,
  kwargs...,
)
  return JSOSolvers.solve!(
    M.solver,
    EnveloptLSR1Model(env_model),
    M.stats;
    atol = tol,  # TODO: play with tolerances
    rtol = 0.0,
    verbose = 0,
    kwargs...,
  )
end
