export TronEnveloptSubSolver

# TRON subproblem solver
mutable struct TronEnveloptSubSolver <: AbstractEnveloptSubSolver
  solver::TronSolver
  stats::GenericExecutionStats
  name::String
end

# ... constructor
function TronEnveloptSubSolver(env_model::EnveloptNLPModel)
  @debug "initializing Tron subproblem solver"
  unconstrained(env_model) ||
    bound_constrained(env_model) ||
    error("Tron only supports unconstrained and bound-constrained problems")
  solver = TronSolver(EnveloptLSR1Model(env_model))  # at this point, only the problem dimensions are used
  stats = GenericExecutionStats(env_model)
  return TronEnveloptSubSolver(solver, stats, "Tron")
end

const tron_fixed_options = Dict(:max_iter => 1000, :use_only_objgrad => true, :max_time => 600.0)

# ... solve
function (M::TronEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  args...;
  tol::Float64 = 1.0e-6,
  kwargs...,
)
  JSOSolvers.solve!(
    M.solver,
    EnveloptLSR1Model(env_model),
    M.stats;
    atol = tol,  # TODO: play with tolerances
    rtol = 0.0,
    x = x0,
    tron_fixed_options...,
    kwargs...,
  )
  return M.stats
end
