@testmodule KnitroHelper begin
  using KNITRO

  knitro_available = try
    # KNITRO.has_knitro() returns true even if there is no valid license, and seems fairly useless
    KNITRO.KN_free(KNITRO.KN_new())
    true
  catch
    false
  end

  if knitro_available
    println("KNITRO is available. Running Knitro tests.")
  else
    println("KNITRO is not available. Skipping Knitro tests.")
  end
end

@testitem "Knitro not available by default" tags=[:solver, :knitro] begin
  @test_throws ErrorException KnitroEnveloptSubSolver()
end

@testitem "single subproblem solve with Knitro" setup=[KnitroHelper] tags=[
  :solver,
  :unconstrained,
  :knitro,
] begin
  if KnitroHelper.knitro_available
    using ADNLPModels, KNITRO, NLPModels, NLPModelsKnitro, ProximalOperators
    model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
    h = NormL1(1.0)
    env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
    knitro_solver = KnitroEnveloptSubSolver(env_model)
    stats = knitro_solver(env_model, get_x0(env_model), 0, tol = 1.0e-2)
    @test Envelopt.first_order(stats)
  end
end

@testitem "simple unconstrained envelopt solve with Knitro" setup=[KnitroHelper] tags=[
  :solver,
  :unconstrained,
  :knitro,
] begin
  if KnitroHelper.knitro_available
    using ADNLPModels, KNITRO, NLPModelsKnitro, ProximalOperators
    model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
    h = NormL1(1.0)
    env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
    stats, status, u =
      envelopt(env_model; subsolver = KnitroEnveloptSubSolver(env_model), verbose = false)
    @test Envelopt.first_order(stats)
  end
end

@testitem "simple bound-constrained envelopt solve with Knitro" setup=[KnitroHelper] tags=[
  :solver,
  :boundconstrained,
  :knitro,
] begin
  if KnitroHelper.knitro_available
    using ADNLPModels, KNITRO, NLPModelsKnitro, ProximalOperators
    using OptimizationProblems, OptimizationProblems.ADNLPProblems
    model = hs1()
    h = NormL1(1.0)
    env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
    stats, status, u =
      envelopt(env_model; subsolver = KnitroEnveloptSubSolver(env_model), verbose = false)
    @test Envelopt.first_order(stats)
  end
end

@testitem "test constrained problem with Knitro" setup=[KnitroHelper] tags=[
  :solver,
  :constrained,
  :knitro,
] begin
  if KnitroHelper.knitro_available
    using ADNLPModels, KNITRO, NLPModelsKnitro, ProximalOperators
    using OptimizationProblems, OptimizationProblems.ADNLPProblems
    model = hs13()
    h = NormL1(1.0)
    env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
    stats, status, u =
      envelopt(env_model; subsolver = KnitroEnveloptSubSolver(env_model), verbose = false)
    @test Envelopt.first_order(stats)
  end
end
