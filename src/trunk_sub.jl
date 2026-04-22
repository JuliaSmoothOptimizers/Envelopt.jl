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

const trunk_fixed_options = Dict(:max_iter => 1000)

# ... solve
function (M::TrunkEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  args...;
  tol::Float64 = 1.0e-6,
  kwargs...,
)
  return JSOSolvers.solve!(
    M.solver,
    EnveloptLSR1Model(env_model),
    M.stats;
    atol = 0.0,  # TODO: play with tolerances
    rtol = tol,
    x = x0,
    trunk_fixed_options...,
    kwargs...,
  )
end
