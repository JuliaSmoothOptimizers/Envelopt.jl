using ProximalOperators, ADNLPModels
using Random
using JuMP
import Clarabel
import SCS

using Envelopt

include("vectorized_proximable.jl")

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

function solve_conic_problem(prob; tol = 1e-6, optimizer = Clarabel.Optimizer)
  nvar = length(prob.c)
  model = Model(optimizer)
  @variable(model, x[1:nvar])
  @objective(model, Min, sum(prob.c .* x))
  @constraint(model, sum(prob.A[:, :, i] .* x[i] for i = 1:nvar) - prob.b in prob.K)
  if optimizer == Clarabel.Optimizer
    set_attribute(model, "tol_gap_abs", tol)
    set_attribute(model, "tol_gap_rel", 0.0)
    set_attribute(model, "tol_feas", 0.0)
  elseif optimizer == SCS.Optimizer
    set_attribute(model, "eps_abs", tol)
    set_attribute(model, "eps_rel", 0.0)
    set_attribute(model, "eps_infeas", 0.0)
  else
    @warn "Unknown optimizer $(optimizer)"
  end
  optimize!(model)
  xval = value.(x)
  objval = sum(prob.c .* xval)
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
  f(x) = -x[2]
  x0 = randn(nvar)
  model = ADNLPModel(f, x0)
  hmat = IndPSD()
  h = VectorizedProximable(hmat, nmat)
  Fmodel = ADNLPModel!(x -> 0.0, zeros(nvar), eval_F!, zeros(nF), zeros(nF))
  emodel = EnveloptNLPModel(model, Fmodel, h)
  return emodel
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

function process_output(stats, status, inner_iter)
  if status != "first_order"
    @warn "failed problem"
  end
  display(stats.solution)
  println("ENV iterations $(stats.iter)")
  println("NLP iterations $(inner_iter)")
end

Random.seed!(123) # seed for reproducibility

TOL = 1e-6

# solve with SCS
x, fx = solve_conic_problem(problem, tol = TOL, optimizer = SCS.Optimizer)
display(x)

# solve with Clarabel
problem = example_4_Ramana_JuMP()
x, fx = solve_conic_problem(problem, tol = TOL, optimizer = Clarabel.Optimizer)
display(x)

# solve with Envelopt+Ipopt
emodel = example_4_Ramana()
stats, status, u, inner_iter =
  envelopt(emodel, dtol_min = TOL, ptol_min = TOL, subsolver = IPOPTEnveloptSubSolver(emodel))
process_output(stats, status, inner_iter)

# solve with Envelopt+MadNLP
emodel = example_4_Ramana()
stats, status, u, inner_iter =
  envelopt(emodel, dtol_min = TOL, ptol_min = TOL, subsolver = MadNLPEnveloptSubSolver(emodel))
process_output(stats, status, inner_iter)