@testmodule R2NHelpers begin
  using ADNLPModels, ProximalOperators

  # minimize ½‖x - a‖² + β‖x‖₁ + λ‖x‖₁,
  # where β‖x‖₁ is the convex function of Envelopt (with F(x) = x)
  # and λ‖x‖₁ is the regularizer handled by R2N.
  # The solution is the soft-thresholding of a at level λ + β.
  const a = [3.0, -0.5, 1.5, -2.0]
  const β = 0.6
  const λ = 0.4
  const xstar = sign.(a) .* max.(abs.(a) .- (λ + β), 0)  # [2.0, 0.0, 0.5, -1.0]
  const objstar = sum((xstar .- a) .^ 2) / 2 + (λ + β) * sum(abs, xstar)

  soft_threshold_model() = ADNLPModel(x -> sum((x .- a) .^ 2) / 2, zeros(length(a)))
end

@testitem "R2N subsolver: single subproblem solve" tags=[:solver, :unconstrained, :r2n] begin
  using ADNLPModels, NLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  env_model = EnveloptNLPModel(model, NormL1(1.0))  # F(x) = x and μ = 1 by default
  r2n_solver = R2NEnveloptSubSolver(env_model, NormL1(0.1))
  stats = r2n_solver(env_model, get_x0(env_model), 0, tol = 1.0e-4)
  @test Envelopt.first_order(stats)
end

@testitem "R2N subsolver: envelopt solve" setup=[R2NHelpers] tags=[:solver, :unconstrained, :r2n] begin
  using NLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  for hessian_approximation in (:exact_f, :lbfgs, :lsr1)
    env_model = EnveloptNLPModel(R2NHelpers.soft_threshold_model(), NormL1(R2NHelpers.β))
    subsolver = R2NEnveloptSubSolver(
      env_model,
      NormL1(R2NHelpers.λ);
      hessian_approximation = hessian_approximation,
    )
    stats, status, u =
      envelopt(env_model; subsolver = subsolver, verbose = false, dtol_min = 1e-5, ptol_min = 1e-5)
    @test status == :first_order
    @test isapprox(stats.solution, R2NHelpers.xstar, atol = 1e-3)
    @test isapprox(u, R2NHelpers.xstar, atol = 1e-3)
    # the reported objective includes the regularizer
    @test isapprox(stats.objective, R2NHelpers.objstar, atol = 1e-3)
  end
end

@testitem "R2N subsolver: invalid inputs" tags=[:solver, :r2n] begin
  using ADNLPModels, NLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  env_model = EnveloptNLPModel(model, NormL1(1.0))
  @test_throws ArgumentError R2NEnveloptSubSolver(
    env_model,
    NormL1(0.1);
    hessian_approximation = :foo,
  )

  # constraints c(x) ∈ C are not supported
  cmodel = ADNLPModel(
    x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0],
    x -> [x[1] + x[2]],
    [-Inf],
    [1.0],
  )
  cenv_model = EnveloptNLPModel(cmodel, NormL1(1.0))
  @test_throws ErrorException R2NEnveloptSubSolver(cenv_model, NormL1(0.1))

  # the subsolver is tied to the model it was built for
  r2n_solver = R2NEnveloptSubSolver(env_model, NormL1(0.1))
  other_env_model = EnveloptNLPModel(model, NormL1(1.0))
  @test_throws ErrorException r2n_solver(other_env_model, get_x0(other_env_model), 0)
end
