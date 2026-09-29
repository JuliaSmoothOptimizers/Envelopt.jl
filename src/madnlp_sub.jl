export MadNLPEnveloptSubSolver

using MadNLP

# MadNLP subproblem solver
mutable struct MadNLPEnveloptSubSolver <: AbstractEnveloptSubSolver
  solver::MadNLPSolver
  stats::MadNLP.MadNLPExecutionStats
  name::String
end

# ... constructor
function MadNLPEnveloptSubSolver(env_model::EnveloptNLPModel)
  @debug "initializing MADNLP subproblem solver"
  solver =
    MadNLPSolver(env_model, hessian_approximation = MadNLP.CompactLBFGS, print_level = MadNLP.ERROR)
  stats = MadNLP.MadNLPExecutionStats(solver)
  return MadNLPEnveloptSubSolver(solver, stats, "MadNLP")
end

const madnlp_fixed_options = Dict(:max_iter => 1000, :dual_initialized => true)

# ... solve
# FIXME: replace outer_iter and madnlp_y with the Envelopt solver object
function (M::MadNLPEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  outer_iter::Int;
  tol::Float64 = 1.0e-6,
  kwargs...,
)
  # FIXME: initialize solver outside the loop only and reuse.
  # See https://github.com/MadNLP/MadNLP.jl/discussions/552
  # MadNLP._reset!(solver.kkt.quasi_newton)
  M.solver =
    MadNLPSolver(env_model, hessian_approximation = MadNLP.CompactLBFGS, print_level = MadNLP.ERROR)

  # prepare for warm start
  # TODO: try solver.mu from the previous solve
  mu_init = 1.0e-1
  bound_push = 1.0e-2
  if outer_iter == 2
    mu_init = 1e-3
    bound_push = 1e-3
  elseif outer_iter == 4
    mu_init = 1e-5
    bound_push = 1e-5
  elseif outer_iter == 6
    mu_init = 1e-6
    bound_push = 1e-6
  elseif outer_iter == 8
    mu_init = 1e-7
    bound_push = 1e-7
  elseif outer_iter >= 10
    mu_init = 1e-8
    bound_push = 1e-8
  end

  # MadNLP uses info from the problem itself to warm start.
  # The problem is stored inside the solver.
  copyto!(get_x0(M.solver.nlp), x0)  # to warm start the next outer iteration
  copyto!(get_y0(M.solver.nlp), M.stats.multipliers)

  MadNLP.solve!(
    M.solver,
    M.stats;
    mu_init = mu_init,
    bound_push = bound_push,
    tol = tol,
    madnlp_fixed_options...,
    kwargs...,
  )
  return M.stats
end

const status_map = Dict(
  MadNLP.SOLVE_SUCCEEDED => :first_order,
  MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL => :acceptable,
  MadNLP.INFEASIBLE_PROBLEM_DETECTED => :infeasible,
  MadNLP.MAXIMUM_ITERATIONS_EXCEEDED => :max_iter,
  MadNLP.MAXIMUM_WALLTIME_EXCEEDED => :max_time,
  MadNLP.DIVERGING_ITERATES => :unbounded,
  MadNLP.INVALID_NUMBER_DETECTED => :exception,
  MadNLP.ERROR_IN_STEP_COMPUTATION => :small_step,
  MadNLP.INTERNAL_ERROR => :exception,
  MadNLP.USER_REQUESTED_STOP => :user,
)
get_substat(stats::MadNLP.MadNLPExecutionStats) = get(status_map, stats.status, :unknown)
failed(stats::MadNLP.MadNLPExecutionStats) = stats.status != MadNLP.SOLVE_SUCCEEDED
first_order(stats::MadNLP.MadNLPExecutionStats) = stats.status == MadNLP.SOLVE_SUCCEEDED
