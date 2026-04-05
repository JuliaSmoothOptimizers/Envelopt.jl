using ProximalOperators, ADNLPModels
using Random

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

N = 10
dim = 2
nvar = N * N
nsym = Int(N * (N - 1) / 2)
nobs = Int(floor((nvar - nsym) / 3))
ncon = nobs + nsym

f(x) = 0
x0 = randn(nvar)
iobs, jobs, vobs = sampled_distance_matrix(N, nobs, dim)
Aobs, bobs = obs_constraints(N, iobs, jobs, vobs)
Asym = sym_constraints(N)
c!(cx, x) = eval_constraints!(cx, x, nobs, nsym, Aobs, bobs, Asym)
model = ADNLPModel!(f, x0, c!, zeros(ncon), zeros(ncon))
gmat = NuclearNorm(1.0)
g = VectorizedProximable(gmat, N, N)

env_model = EnveloptNLPModel(model, g)
stats, status, u = envelopt(env_model, verbose = true, max_outer = 30)
@assert status == "first_order"
x = stats.solution
xmat = reshape(x, (N, N))
println("Rank of x = $(rank(xmat))")
umat = reshape(u, (N, N))
println("Rank of u = $(rank(umat))")
