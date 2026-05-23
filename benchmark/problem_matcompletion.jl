using ProximalOperators, ADNLPModels
using LinearAlgebra, Random, Statistics

using Envelopt

include("vectorized_proximable.jl")

function sampled_distance_matrix(N::Int, nobs::Int, l::Int)
  @assert l >= 2
  @assert N > 0
  nsym = Int(N * (N - 1) / 2)
  @assert 0 < nobs < (N * N - nsym)
  # sample points
  X = randn(N, l)
  # distance matrix
  D = zeros(N, N)
  for i = 1:N
    for j = (i + 1):N
      D[i, j] = sum((X[i, :] - X[j, :]) .^ 2)
      D[j, i] = D[i, j]
    end
  end
  # sample observations
  idx = randperm(N * N)[1:nobs]
  sort!(idx)
  # extract indices
  IJ = CartesianIndices(D)[idx]
  iobs = zeros(Int, nobs)
  jobs = zeros(Int, nobs)
  for k = 1:nobs
    iobs[k] = IJ[k][1]
    jobs[k] = IJ[k][2]
  end
  vobs = D[IJ]
  return iobs, jobs, vobs
end

function obs_constraints(N, iobs, jobs, vobs)
  nobs = length(iobs)
  @assert length(jobs) == nobs
  @assert length(vobs) == nobs
  nvar = N * N
  Aobs = zeros(nobs, nvar)
  bobs = zeros(nobs)
  lindex = LinearIndices((N, N))
  for k = 1:nobs
    i = iobs[k]
    j = jobs[k]
    Aobs[k, lindex[i, i]] = 1
    Aobs[k, lindex[i, j]] = -1
    Aobs[k, lindex[j, i]] = -1
    Aobs[k, lindex[j, j]] = 1
    bobs[k] = vobs[k]
  end
  return Aobs, bobs
end

function sym_constraints(N)
  nvar = N * N
  nsym = Int(N * (N - 1) / 2)
  Asym = zeros(nsym, nvar)
  lindex = LinearIndices((N, N))
  k = 0
  for j = 1:N
    for i = (j + 1):N
      k += 1
      Asym[k, lindex[i, j]] = 1
      Asym[k, lindex[j, i]] = -1
    end
  end
  @assert k == nsym
  return Asym
end

function eval_constraints!(cx, x, nobs, nsym, Aobs, bobs, Asym)
  cx[1:nobs] .= Aobs * x - bobs # observations
  cx[(nobs + 1):(nobs + nsym)] .= Asym * x # symmetry
  return cx
end

Random.seed!(123) # seed for reproducibility

N = 10
dim = 5
nvar = N * N
nsym = Int(N * (N - 1) / 2)
nobs = Int(floor((nvar - nsym) / 3))
ncon = nobs + nsym

ntrials = 100
res = (
  iter = zeros(ntrials),
  inner_iter = zeros(ntrials),
  rank_x = zeros(ntrials),
  rank_u = zeros(ntrials),
)

f(x) = 0
gmat = NuclearNorm(1.0)
g = VectorizedProximable(gmat, N, N)

for i = 1:ntrials
  println("--------------------")
  println("problem #$(i) / $(ntrials)")
  x0 = randn(nvar)
  iobs, jobs, vobs = sampled_distance_matrix(N, nobs, dim)
  Aobs, bobs = obs_constraints(N, iobs, jobs, vobs)
  Asym = sym_constraints(N)
  c!(cx, x) = eval_constraints!(cx, x, nobs, nsym, Aobs, bobs, Asym)
  model = ADNLPModel!(f, x0, c!, zeros(ncon), zeros(ncon))

  env_model = EnveloptNLPModel(model, g)
  stats, status, u, tot_inner_iter = envelopt(env_model, verbose = true, max_outer = 30)
  x = stats.solution

  res.iter[i] = stats.iter
  res.inner_iter[i] = tot_inner_iter
  if status == "first_order"
    xmat = reshape(x, (N, N))
    rank_x = rank(xmat)
    println("Rank of x = $(rank_x)")
    res.rank_x[i] = rank_x
    umat = reshape(u, (N, N))
    rank_u = rank(umat)
    println("Rank of u = $(rank_u)")
    res.rank_u[i] = rank_u
  else
    @warn "failed problem"
    res.rank_x[i] = NaN
    res.rank_u[i] = NaN
  end
end

is_problem_solved = .!(isnan.(res.rank_u))
println("$(ntrials) trials")
println("- improved rank $(sum((res.rank_u[is_problem_solved] .< N)))")
println("- failed $(ntrials-sum(is_problem_solved))")
println("Median iterations: $(median(res.iter[is_problem_solved]))")
println("Median inner iterations: $(median(res.inner_iter[is_problem_solved]))")