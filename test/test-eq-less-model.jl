@testitem "equality-less model of model without equalities" tags=[:modifier] begin
  using ADNLPModels
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  @test_throws ErrorException EqualityLessModel(model)
end

@testitem "equality-less model of model with equalities only" tags=[:modifier] begin
  using ADNLPModels, NLPModels, NLPModelsTest
  model = ADNLPModel(
    x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0],
    x -> [x[1] + x[2] - 2.0],
    [0.0],
    [0.0],
  )
  eq_less_model = EqualityLessModel(model)
  @test get_nvar(eq_less_model) == get_nvar(model)
  @test get_ncon(eq_less_model) == 0
  @test get_nnzj(eq_less_model) == 0
  @test all(get_x0(eq_less_model) .== get_x0(model))
  @test obj(eq_less_model, eq_less_model.meta.x0) == obj(model, model.meta.x0)
  @test all(grad(eq_less_model, eq_less_model.meta.x0) .== grad(model, model.meta.x0))
  g_errs = gradient_check(eq_less_model)
  @test length(g_errs) == 0
end

@testitem "equality-less model of model with equalities and inequalities" tags=[:modifier] begin
  using ADNLPModels, NLPModels, NLPModelsTest
  model = ADNLPModel(
    x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0],
    x -> [x[1] + x[2] - 2.0, x[1] - 0.5],
    [0.0, 0.0],
    [0.0, Inf],
  )
  eq_less_model = EqualityLessModel(model)
  @test get_nvar(eq_less_model) == get_nvar(model)
  @test get_ncon(eq_less_model) == 1
  @test get_nnzj(eq_less_model) == 1
  @test all(get_x0(eq_less_model) .== get_x0(model))
  @test obj(eq_less_model, eq_less_model.meta.x0) == obj(model, model.meta.x0)
  @test all(grad(eq_less_model, eq_less_model.meta.x0) .== grad(model, model.meta.x0))
  g_errs = gradient_check(eq_less_model)
  @test length(g_errs) == 0
end
