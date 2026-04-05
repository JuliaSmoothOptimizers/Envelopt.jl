@testitem "eq-feasibility model of unconstrained model" tags=[:modifier] begin
  using ADNLPModels
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  @test_throws ErrorException EqualityFeasibilityModel(model)
end

@testitem "eq-feasibility model of equality-constrained model" tags=[:modifier] begin
  using ADNLPModels, NLPModels, NLPModelsTest
  model = ADNLPModel(
    x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0],
    x -> [x[1] + x[2] - 1.0],
    [0.0],
    [0.0],
  )
  feas_model = EqualityFeasibilityModel(model)
  @test equality_constrained(feas_model)
  @test get_ncon(feas_model) == 1
  @test all(cons(feas_model, get_x0(feas_model)) .== cons(model, get_x0(model)))
  @test all(get_lcon(feas_model) .== get_lcon(model))
  @test all(get_ucon(feas_model) .== get_ucon(model))
  @test obj(feas_model, get_x0(feas_model)) == 0.0
  @test all(grad(feas_model, get_x0(feas_model)) .== 0.0)
end

@testitem "eq-feasibility model of model with equalities and inequalities" tags=[:modifier] begin
  using ADNLPModels, NLPModels, NLPModelsTest
  model = ADNLPModel(
    x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0],
    x -> [x[1] + x[2] - 2.0, x[1] - 0.5],
    [0.0, 0.0],
    [0.0, Inf],
  )
  feas_model = EqualityFeasibilityModel(model)
  @test equality_constrained(feas_model)
  @test get_ncon(feas_model) == 1
  jfix = get_jfix(model)
  @test all(cons(feas_model, get_x0(feas_model)) .== cons(model, get_x0(model))[jfix])
  @test all(get_lcon(feas_model) .== get_lcon(model)[jfix])
  @test all(get_ucon(feas_model) .== get_ucon(model)[jfix])
  @test all(get_y0(feas_model) .== get_y0(model)[jfix])
  @test obj(feas_model, get_x0(feas_model)) == 0.0
  @test all(grad(feas_model, get_x0(feas_model)) .== 0.0)
end
