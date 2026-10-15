@testitem "R2 subsolver: single subproblem solve" setup=[R2NHelpers] tags=[
  :solver,
  :unconstrained,
  :r2,
] begin
  using NLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  env_model = EnveloptNLPModel(R2NHelpers.soft_threshold_model(), NormL1(R2NHelpers.β))
  r2_solver = R2EnveloptSubSolver(env_model, NormL1(R2NHelpers.λ))
  stats = r2_solver(env_model, get_x0(env_model), 0, tol = 1.0e-4)
  @test Envelopt.first_order(stats)
end

@testitem "R2 subsolver: envelopt solve" setup=[R2NHelpers] tags=[:solver, :unconstrained, :r2] begin
  using NLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  env_model = EnveloptNLPModel(R2NHelpers.soft_threshold_model(), NormL1(R2NHelpers.β))
  subsolver = R2EnveloptSubSolver(env_model, NormL1(R2NHelpers.λ))
  stats, status, u =
    envelopt(env_model; subsolver = subsolver, verbose = false, dtol_min = 1e-5, ptol_min = 1e-5)
  @test status == :first_order
  @test isapprox(stats.solution, R2NHelpers.xstar, atol = 1e-3)
  @test isapprox(u, R2NHelpers.xstar, atol = 1e-3)
  # the reported objective includes the regularizer
  @test isapprox(stats.objective, R2NHelpers.objstar, atol = 1e-3)
end

@testitem "R2 subsolver: invalid inputs" tags=[:solver, :r2] begin
  using ADNLPModels, NLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  env_model = EnveloptNLPModel(model, NormL1(1.0))

  # constraints c(x) ∈ C are not supported
  cmodel = ADNLPModel(
    x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0],
    x -> [x[1] + x[2]],
    [-Inf],
    [1.0],
  )
  cenv_model = EnveloptNLPModel(cmodel, NormL1(1.0))
  @test_throws ErrorException R2EnveloptSubSolver(cenv_model, NormL1(0.1))

  # the subsolver is tied to the model it was built for
  r2_solver = R2EnveloptSubSolver(env_model, NormL1(0.1))
  other_env_model = EnveloptNLPModel(model, NormL1(1.0))
  @test_throws ErrorException r2_solver(other_env_model, get_x0(other_env_model), 0)
end
