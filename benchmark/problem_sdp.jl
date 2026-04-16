using ProximalOperators, ADNLPModels
using Random
using JuMP
import Clarabel

using Envelopt

include("vectorized_proximable.jl")

Random.seed!(123456789)

# problem format
# min <c,x>
# wrt x
#  st Ax - b in K
abstract type AbstractConicProblem end

struct ConicProblem <: AbstractConicProblem
  b::AbstractArray
  A::AbstractArray
  c::AbstractArray
  K
end

function solve_conic_problem(prob; tol = 1e-6)
  nvar = length(prob.c)
  model = Model(Clarabel.Optimizer)
  @variable(model, x[1:nvar])
  @objective(model, Min, sum(prob.c .* x) + x[2]^2)
  @constraint(model, sum(prob.A[:, :, i] .* x[i] for i = 1:nvar) - prob.b in prob.K)
  set_attribute(model, "tol_gap_abs", tol)
  set_attribute(model, "tol_gap_rel", tol)
  set_attribute(model, "tol_feas", tol)
  optimize!(model)
  xval = value.(x)
  objval = sum(prob.c .* x)
  return xval, objval
end

###############
# M. V. Ramana. An exact duality theory for semidefinite programming and its complexity
# implications. Mathematical Programming, 77(1):129–162, 1997. doi:10.1007/BF02614433.
#
# Example 4:
#   minimize        -s
#   subject to      [1-s, 0, 0]
#                   [0, -t, -s] ∈ PSDCone
#                   [0, -s,  0]
# Optimal value = 0.
###############
function example_4_Ramana()
  nvar = 2
  nmat = 3
  nF = nmat * nmat
  eval_F!(Fx, x) = begin
    Fx .= 0.0
    Fx[1] = 1 - x[2]
    Fx[5] = -x[1]
    Fx[6] = -x[2]
    Fx[8] = -x[2]
    Fx
  end
  f(x) = -x[2] + x[2]^2
  x0 = randn(nvar)
  model = ADNLPModel(f, x0)
  hmat = IndPSD()
  h = VectorizedProximable(hmat, nmat)
  Fmodel = ADNLPModel!(x -> 0.0, zeros(nvar), eval_F!, zeros(nF), zeros(nF))
  env_model = EnveloptNLPModel(model, Fmodel, h)
  return env_model
end

function example_4_Ramana_JuMP()
  nvar = 2
  c = zeros(nvar)
  c[2] = -1
  K = PSDCone()
  A = zeros(3, 3, nvar)
  A[:, :, 1] = [0 0 0; 0 -1 0; 0 0 0]
  A[:, :, 2] = [-1 0 0; 0 0 -1; 0 -1 0]
  b = [-1 0 0; 0 0 0; 0 0 0]
  prob = ConicProblem(b, A, c, K)
  return prob
end

# solve with Envelopt
env_model = example_4_Ramana()
stats, status, u = envelopt(
  env_model,
  verbose = true,
  max_outer = 10000,
  # subsolver = MadNLPEnveloptSubSolver(env_model),
  subsolver = TronEnveloptSubSolver(env_model),
)
@assert status == "first_order"
x = stats.solution
display(x)

# solve with Clarabel
problem = example_4_Ramana_JuMP()
x, fx = solve_conic_problem(problem)
display(x)
