using ProximalOperators, ManualNLPModels, NLPModels
using Random
using FFTW, LinearAlgebra, AbstractOperators
using MAT, Images, SparseArrays

using Envelopt

include("vectorized_proximable.jl")

REDUCE_PROBLEM_SIZE = true
pipa_data_folder = normpath(joinpath(@__DIR__, "PIPA-master", "data_geo_text"))

function make_pipa_F(nrows, ncols)
  # edge detection operator
  Laplacian = [0 1 0; 1 -4 1; 0 1 0]
  W = zeros(nrows, ncols)
  im = Int(ceil(nrows / 2))
  in = Int(ceil(ncols / 2))
  W[im:(im + 2), in:(in + 2)] .= Laplacian
  W = ones(nrows, ncols) .- real(fft(fftshift(W)))
  N = nrows * ncols
  # convolution operator
  F(x) = reshape(W .* real(fft(reshape(x, nrows, ncols))), N) ./ nrows
  # adjoint
  FT(z) = reshape(real(ifft(W .* reshape(z, nrows, ncols))), N) .* nrows
  return F, FT
end

function make_pipa_H()
  # read Radon operator
  tmp = matread(joinpath(pipa_data_folder, "H.mat"))
  H = tmp["H"]
  return H
end

function load_image(sample_name)
  sample_length = 415
  if sample_name == "glass"
    tmp = matread(joinpath(pipa_data_folder, "glass1024.mat"))
    tmpX = tmp["I"]
    tmpXX = imresize(view(tmpX, 286:(286 + sample_length), 87:(87 + sample_length)), (128, 128))
    weightTV = 0.25
  elseif sample_name == "agaricus"
    tmpX = load(joinpath(pipa_data_folder, "agari0330_8bit.tif"))
    tmpXX = imresize(view(tmpX, 420:(420 + sample_length), 420:(420 + sample_length)), (128, 128))
    weightTV = 0.5
  end
  @assert size(tmpXX) == (128, 128)
  # disk mask to simulate a CT acquisition
  tmp = matread(joinpath(pipa_data_folder, "disk61.mat"))
  disk = tmp["disk"]
  disk[disk .< maximum(disk[:]) / 2] .= 0
  disk[disk .> 0] .= 1
  disk_full = zeros(128, 128)
  disk_full[3:125, 3:125] .= disk
  # apply mask
  X = disk_full .* tmpXX
  # normalization
  X = X ./ maximum(X[:])
  return X, weightTV
end

function noisy_CT_image(X, H)
  # apply observation and degradation model for CT
  y = H * X[:]                      # tomographic data
  chi = 0.02 * maximum(abs.(y))              # amplitude of uniform noise 2% of max value
  y = y .+ chi .* (-1 .+ 2 .* rand(size(H, 1))) # noisy sinogram
  return y, chi
end

Random.seed!(123) # For reproducibility

sample_name = "glass"
println("sample:           $(sample_name)\n")
Xgroundt, weightTV = load_image(sample_name)
pipa_H = make_pipa_H() # Radon operator
if REDUCE_PROBLEM_SIZE
  @warn "Problem with reduced size for quicker testing"
  pipa_H = pipa_H[1:50, :]
end
y, chi = noisy_CT_image(Xgroundt, pipa_H)
alpha = 1 / 3

nrows, ncols = size(Xgroundt)
N = nrows * ncols
nmeas = length(y)

# meta
nvar = 2 * N
ncon = N + nmeas
lvar = [-alpha * ones(N); -Inf * ones(N)]
uvar = [alpha * ones(N); Inf * ones(N)]
lcon = [zeros(N); y .- chi]
ucon = [ones(N); y .+ chi]

# smooth objective (NLS structure)
pipa_F, pipa_FT = make_pipa_F(nrows, ncols)

function resid!(rx, x)
  @assert length(x) == nvar
  @assert length(rx) == N
  rx .= pipa_F(x[1:N])
  rx
end

function resid_jtprod!(u, x, v)
  @assert length(u) == nvar
  @assert length(x) == nvar
  @assert length(v) == N
  u[1:N] .= pipa_FT(v)
  u[(N + 1):nvar] .= 0
  u
end

# TODO improve allocations
function nlsobj(x)
  rx = zeros(N)
  resid!(rx, x)
  return dot(rx, rx) / 2
end

function nlsgrad!(gx, x)
  @assert length(gx) == nvar
  @assert length(x) == nvar
  rx = zeros(N)
  resid!(rx, x)
  gx[1:N] .= pipa_FT(rx)
  gx[(N + 1):nvar] .= 0
  gx
end

# standard constraints (linear)
pipa_H_sparse = sparse(pipa_H)
Ac = spzeros(ncon, nvar)
Ac[1:N, 1:N] .= sparse(Matrix(1.0I, N, N))
Ac[1:N, (N + 1):nvar] .= sparse(Matrix(1.0I, N, N))
Ac[(N + 1):ncon, 1:N] .= pipa_H_sparse
Ac[(N + 1):ncon, (N + 1):nvar] .= pipa_H_sparse
Ac_sparse = sparse(Ac)
Ac_i, Ac_j, Ac_v = findnz(Ac_sparse)

c!(cx, x) = begin
  cx .= Ac_sparse * x
  cx
end
c_jprod!(jv, x, v) = begin
  jv .= Ac_sparse * v
  jv
end
c_jtprod!(jtv, x, v) = begin
  jtv .= Ac_sparse' * v
  jtv
end
Ac_vals!(vals, x) = begin
  vals .= Ac_v
  vals
end

# composition function (linear)
V_operator = Variation((nrows, ncols)) # from AbstractOperators
eval_V!(u, x) = begin
  @assert length(u) == 2 * N
  @assert length(x) == nvar
  inp = reshape(x[1:N], nrows, ncols)
  Vx = V_operator * inp
  u .= reshape(Vx, 2 * N, 1)
  u
end

eval_Vadjoint!(u, z) = begin
  @assert length(u) == nvar
  @assert length(z) == 2 * N
  inp = reshape(z, N, 2)
  Vtz = V_operator' * inp
  u[1:N] .= reshape(Vtz, N, 1)
  u[(N + 1):nvar] .= 0.0
  u
end

V_jprod!(jv, x, v) = eval_V!(jv, v)

V_jtprod!(jtv, x, v) = eval_Vadjoint!(jtv, v)

zeroobj(x) = 0.0
zerograd!(gx, x) = begin
  gx .= 0.0
  gx
end

# nonsmooth objective
hmat = NormL21(weightTV, 2)
h = VectorizedProximable(hmat, N, 2)

# buils models
x0 = Random.randn(nvar)

# TODO specify model as NLSModel
# TODO specify constraints are linear
model = NLPModel(
  x0,
  nlsobj,
  grad = nlsgrad!,
  lvar = lvar,
  uvar = uvar,
  cons = (c!, lcon, ucon),
  jprod = c_jprod!,
  jtprod = c_jtprod!,
  jac_coord = (Ac_i, Ac_j, Ac_vals!),
)

# for debugging
#solver = MadNLPSolver(model, hessian_approximation=MadNLP.CompactLBFGS)
#out = solve!(solver)

# TODO specify constraints are linear
Fmodel = NLPModel(
  x0,
  zeroobj,
  grad = zerograd!,
  cons = (eval_V!, zeros(2 * N), zeros(2 * N)),
  jprod = V_jprod!,
  jtprod = V_jtprod!,
)

env_model = EnveloptNLPModel(model, Fmodel, h)

# for debugging
#solver = MadNLPSolver(env_model, hessian_approximation=MadNLP.CompactLBFGS)
#out = solve!(solver)

# TODO add callback to monitor SNR as in the PIPA paper
TOL = 1e-3
runtime = @elapsed stats, status, u =
  envelopt(env_model, verbose = true, max_outer = 30, dtol_min = TOL, ptol_min = TOL)
display(stats)
display(runtime)
