export exact_penalty_solver

# TODO: return proper stats
function exact_penalty_solver(
  nlp::AbstractNLPModel,
  penalty = NormL2;
  verbose = true,
  sub_verbose = true,
)
  (unconstrained(nlp) || bound_constrained(nlp)) &&
    error("exact penalty solver only supports constrained problems")

  # convert inequalities to equalities via slack variables, if needed
  if has_inequalities(nlp)
    snlp = SlackModel(nlp)
  else
    snlp = nlp
  end
  @assert has_equalities(snlp)

  # objective and bounds
  model = EqualityLessModel(snlp)
  @assert !has_equalities(model)

  # F(x)
  Fmodel = EqualityFeasibilityModel(snlp)
  @assert equality_constrained(Fmodel)

  # select subsolver, depending on constraints left in model
  # if has_inequalities(model)
  EnveloptSubSolver = MadNLPEnveloptSubSolver
  # elseif has_bounds(model)
  #   EnveloptSubSolver = TronEnveloptSubSolver
  # else
  #   EnveloptSubSolver = TrunkEnveloptSubSolver
  # end

  if penalty == NormL1
    dual_norm = NormLinf(1.0)
  elseif penalty == NormLinf
    dual_norm = NormL1(1.0)
  elseif penalty == NormL2
    dual_norm = NormL2(1.0)
  else
    error("unsupported penalty function")
  end

  # initializations
  τ = 10.0  # penalty parameter
  iter = 0
  max_iter = 20
  dfeas_tol = 1.0e-6
  subtol = 1.0e-2
  pfeas_measure = penalty(1.0)
  cval = cons(Fmodel, get_x0(Fmodel))
  pfeas = pfeas_measure(cval)
  pfeas_tol = 1.0e-6 * (1 + pfeas)
  x = get_x0(model)

  if verbose
    @info @sprintf "%-4s  %-7s  %-7s  %-7s  %-7s  %-7s\n" "outer" "penalty" "subtol" "dfeas" "pfeas" "‖step‖"
    @info @sprintf "%-4d  %-7.1e  %-7.1e  %-7s  %-7.1e  %-7s\n" iter τ subtol "" pfeas ""
  end

  first_order = false
  tired = iter >= max_iter
  finished = first_order || tired

  local stats
  while !finished
    # penalty = regularizer
    h = penalty(τ)

    # Envelopt model
    env_model = EnveloptNLPModel(model, Fmodel, h, x0 = x)

    # call solver
    stats, status, u = envelopt(
      env_model,
      verbose = sub_verbose,
      subsolver = EnveloptSubSolver(env_model),
      dtol_min = subtol,
      ptol_min = subtol,
      indent = "  ",
    )

    # check progress towards feasibility
    step_norm = pfeas_measure(stats.solution - x)
    x = stats.solution
    kkt = max(stats.dual_feas, stats.primal_feas)
    cons!(Fmodel, x, cval)
    # @info "" x cval
    pfeas_next = pfeas_measure(cval)
    if pfeas_next > 0.9 * pfeas
      τ = max(
        τ * 5,
        dual_norm(stats.multipliers),
        dual_norm(stats.multipliers_L),
        dual_norm(stats.multipliers_U),
      )
      subtol = max(min(dfeas_tol, pfeas_tol) / 2, min(kkt, subtol / 2))
    else
      subtol = max(min(dfeas_tol, pfeas_tol) / 2, min(kkt, subtol / 5))
    end
    pfeas = pfeas_next
    iter += 1

    if iter == 1
      dfeas_tol *= (1 + stats.dual_feas)
    end

    if verbose
      @info @sprintf "%-4d  %-7.1e  %-7.1e  %-7.1e  %-7.1e  %-7.1e\n" iter τ subtol kkt pfeas step_norm
    end

    first_order = stats.dual_feas ≤ dfeas_tol && stats.primal_feas ≤ pfeas_tol && pfeas ≤ pfeas_tol
    tired = iter >= max_iter
    finished = first_order || tired
  end

  return stats, pfeas
end
