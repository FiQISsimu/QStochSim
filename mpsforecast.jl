using LinearAlgebra,KrylovKit,Distributions,HMMBase

const MPSTensor = Vector{Array{ComplexF64,2}}

macro ass(cond,err) 
	return :( $(esc(cond)) || throw(ArgumentError("MPS: " * $(esc(err)))) )
end

" compute eigen density matrix"
function eigendm(K::Vector{T}) where T <: AbstractMatrix
	a0 = rand(ComplexF64,size(K[1])...)
    x0 = a0*a0' # initial guess for the steady state
	f(x) = sum(K[s]*x*K[s]' for s=eachindex(K))
    v,x = eigsolve(f,x0/tr(x0),1,:LM)
	ρ = Hermitian(x[1]/x[1][1,1])
	v[1],ρ/tr(ρ)
end

" spectral decomposition of the steady state of an MPS"
function steadystatedecom(A::Vector{T}) where T <: AbstractMatrix
    v,ρ = eigendm(A)
    @ass v ≈ 1 "wrong largest eigenvalue"
    F = eigen(ρ) 
    @ass all(F.values .> 0) "wrong steady state"
    cond = sum(A[s]*ρ*A[s]' for s=eachindex(A)) ≈ ρ 
    @ass cond "wrong steady state"
    F.values,F.vectors
end

" steady state of a Markov transition matrix "
function steadymarkov(A::Array{T,2}) where T
    D = size(A,1)
    M = Matrix{T}(undef,D+1,D)
    b = zeros(D+1)
    M[1:D,1:D] = A-I
    M[D+1,1:D] .= 1
    b[D+1] = 1
    M\b
end

abstract type MPSGeneral end

" Normalized MPS Quantum simulator from Kraus operators "
struct MPSQuantum <: MPSGeneral
    A::MPSTensor
    λ::Vector{Float64}
    W::Matrix{ComplexF64}

	function MPSQuantum(A::Vector{Array{T,2}}) where T
        λ,W = steadystatedecom(A)
		@ass λ ≈ sort(λ) "wrong ordering"
        new(convert(MPSTensor,A),λ/sum(λ),W)
	end
end

" Normalized MPS Classical simulator from Krauss operators"
struct MPSClassical <: MPSGeneral
    A::MPSTensor
    λ::Vector{Float64}

	function MPSClassical(A::Vector{Array{T,2}}) where T
        M = sum(abs2.(A[i]) for i=eachindex(A))
        @ass all(sum(M,dims=1) .≈ 1.0) "wrong Markov matrix"
        new(convert(MPSTensor,A),steadymarkov(M))
    end
end

function MPSQuantum(ψ::MPSClassical)
    λ,S = steadystatedecom(adjoint.(ψ.A))
    W = Diagonal(sqrt.(λ))*S'
    iW = S*Diagonal(inv.(sqrt.(λ)))
    MPSQuantum([W*A*iW for A in ψ.A])
end


MPSClassical(ψ::MPSQuantum) = MPSClassical(ψ.A)
markovmatrix(ψ::MPSGeneral) = sum(abs2.(A) for A in ψ.A)

memory(ψ::MPSQuantum)    = ComplexF64.(sqrt.(ψ.λ))
memory(ψ::MPSClassical)  = ψ.λ
memsize(ψ::MPSGeneral)   = length(ψ.λ)
krauss(ψ::MPSGeneral)    = ψ.A
nkrauss(ψ::MPSGeneral)   = length(ψ.A)
Base.size(ψ::MPSGeneral) = size(ψ.A[1],1)

Base.show(io::IO, ψ::MPSQuantum)   = print(io, string("MPS Quantum Simulator with memory ", 
                                                    memsize(ψ), " and entropy ", entropy(ψ)))
Base.show(io::IO, ψ::MPSClassical) = print(io, string("MPS Classical Simulator with memory ", 
                                                    memsize(ψ), " and entropy ", entropy(ψ)))

entropy(ψ::MPSGeneral) = shannonentropy(ψ.λ)
randmps(D::Int, d::Int) = MPSQuantum(mpsnormalized([randn(ComplexF64,D,D) for i=1:d]))

steadystate(ψ::MPSQuantum) = ψ.W*Diagonal(ψ.λ)*ψ.W'


" compress an MPS simulator onto a new memory D "
function compress(ψ::MPSClassical,D::Int,sigma=nothing)
    D0,d = memsize(ψ),nkrauss(ψ)
	@ass D < D0 "size cannot be bigger"
    if sigma == :left || sigma == :right
        return _spectralcompress(ψ,D,D0,d,sigma)
    end
    T, Ti = zeros(D0,D), zeros(D,D0)
    n = sum(ψ.λ[i] for i=1:D)
    for i=1:D
        T[i,i] = n
        for j=D+1:D0
            T[j,i]  = ψ.λ[j]
        end
        for j=1:D0
            Ti[i,j] = T[j,i]*ψ.λ[i]/n/ψ.λ[j] # Bayes rule
        end
    end
    MPSClassical([sqrt.(Ti*abs2.(a)*T) for a in ψ.A])
end
    

" compress an MPS simulator onto a new memory D "
function _spectralcompress(ψ::MPSClassical,D,D0,d,sigma=:left)
    P = sum(abs2.(A) for A in ψ.A)
    Q = Matrix{Float64}(undef,d,D0)
    for x=1:d, α=1:D0
        Q[x,α] = sum(abs2(ψ.A[x][β,α]) for β=1:D)
    end
    U0,S0,V0 = svd(P*Diagonal(ψ.λ))
    U,S,V = U0[:,1:D], Diagonal(S0[1:D]), V0[:,1:D]
    posit(x::Float64) = x>0.0 ? x : 0.0
    if sigma==:left
        res = [ posit.(sum(Q[x,β]*(S*V')[:,β]*U[β,:]' for β=1:D0)) for x=1:d]
    elseif sigma==:right
        res = [ posit.(sum(Q[x,β]*V[β,:]*((U*S)[β,:])' for β=1:D0)) for x=1:d]
    else
        res = [ posit.(sum(Q[x,β]*(sqrt.(S)*V')[:,β]*((U*sqrt.(S))[β,:])' for β=1:D0)) for x=1:d]
    end
    n =  sum(sum(res),dims=1)
    MPSClassical([sqrt.(a ./ n) for a in res])
end


" left normalize Kraus operators "
function mpsnormalized(B0::Vector{Array{T,2}}) where T
	μ,ρ = eigendm(adjoint.(B0))
	F = eigen(ρ)
	@ass all(F.values .>= -1e14) "error in normalization"
    f(x) = x>0.0 ? sqrt(x) : 0.0 
	M = Diagonal(f.(F.values))*F.vectors'
	iM = inv(M)
	B = [M*X*iM/sqrt(μ) for X in B0]
	id = sum(X'*X for X in B)
	@ass id ≈ I "wrong Kraus operators"
	B
end

" compress an MPS simulator onto a new memory D "
function compress(ψ::MPSQuantum,D::Int)
	D0 = memsize(ψ)
	@ass D < D0 "size cannot be bigger"
	Wc = ψ.W[:,D0-D+1:D0]
    MPSQuantum(mpsnormalized([Wc'*A*Wc for A in ψ.A]))
end

" Optimize a compressed state ϕ to match ψ "
function optimize(ψ::MPSQuantum,ϕ::MPSQuantum,iters=10,η=0.01)
	@ass nkrauss(ψ) == nkrauss(ϕ) "different number of Krauss operators"
	@ass nkrauss(ψ) > 0 "wrong input"
    A, B = ψ.A, deepcopy(ϕ.A)
    for i=1:iters
	    x0 = rand(ComplexF64,size(ϕ),size(ψ))
	    f(x) = sum(B[s]*x*A[s]' for s=1:nkrauss(ψ))
	    v,x = eigsolve(f,x0,1,:LM)
        L = x[1]
        for s=1:nkrauss(ψ)
            B[s] += η/v[1] * (L*A[s]'*L')
        end
        B = mpsnormalized((B))
    end
    MPSQuantum(B)
end

" calculate the asymptotic fidelity decay rate and other stuff "
function fidrate(ψ::MPSQuantum,ϕ::MPSQuantum,L=0)
	@ass nkrauss(ψ) == nkrauss(ϕ) "different number of Krauss operators"
	@ass nkrauss(ψ) > 0 "wrong input"
	x0 = rand(ComplexF64,size(ψ),size(ϕ))
	f(x) = sum(ψ.A[s]*x*ϕ.A[s]' for s=1:nkrauss(ψ))
	v,x = eigsolve(f,x0,1,:LM)
	if L>0
		U,d,V = svd(x[1])
	    op = U[:,1]*V[:,1]'
		y = vec(ψ.W*Diagonal(sqrt.(ψ.λ))*ψ.W')*vec(ϕ.W*Diagonal(sqrt.(ϕ.λ))*ϕ.W')'
        iψ = zeros(ComplexF64,size(ψ),size(ψ))+I
        iϕ = zeros(ComplexF64,size(ϕ),size(ϕ))+I
        fs = zeros(L)
        fc = zeros(L)
        for ℓ=1:L
            op = f(op)
		    y = sum(kron(ψ.A[s],iψ)*y*kron(ϕ.A[s]',iϕ) for s=1:nkrauss(ψ))
            fs[ℓ] = sum(svdvals(op))
            fc[ℓ] = sum(svdvals(y))
        end
        return abs(v[1]), fs, fc
	end
    abs(v[1]), [0.0], [0.0]
end

transition(ψ::MPSClassical) = [abs2.(A) for A in ψ.A]
transition(ψ::MPSQuantum) = [kron(A,conj(A)) for A in ψ.A]


" similarity rate between probability distrubitions "
function similarity(ψ::MPSGeneral,ϕ::MPSGeneral)
	@ass nkrauss(ψ) == nkrauss(ϕ) "different number of Krauss operators"
	@ass nkrauss(ψ) > 0 "wrong input"
    T1, T2 = transition(ψ), transition(ϕ)
    x0 = rand(ComplexF64,size(T1[1],2),size(T2[1],2))
    x1 = rand(ComplexF64,size(T1[1],2),size(T1[1],2))
    x2 = rand(ComplexF64,size(T2[1],2),size(T2[1],2))
	f(x) = sum(T1[s]*x*T2[s]' for s=1:nkrauss(ψ))
	f1(x) = sum(T1[s]*x*T1[s]' for s=1:nkrauss(ψ))
	f2(x) = sum(T2[s]*x*T2[s]' for s=1:nkrauss(ψ))
	v,_ = eigsolve(f,x0,1,:LM)
	v1,_ = eigsolve(f1,x1,1,:LM)
	v2,_ = eigsolve(f2,x2,1,:LM)
    abs(v[1])/sqrt(abs(v1[1])*abs(v2[1]))
end

function shannonentropy(p::Vector{Float64})
	f(x) = (x > 0 && x < 1) ? -x*log2(x) : 0.0
	sum(f.(p))
end

tmat(::MPSQuantum,K)    = K
tmat(::MPSClassical,K)  = abs2.(K)
deno(::MPSQuantum,m)    = norm(m)
deno(::MPSClassical,m)  = sum(m)
prob(::MPSQuantum,m)    = norm(m)^2
prob(::MPSClassical,m)  = sum(m)

#function sample!(ψ::MPSGeneral,n::Int,mem::Vector{ComplexF64})
function sample!(ψ::MPSGeneral,n::Int,mem::Vector{T}) where T
    res = zeros(Int,n)
    for i=1:n
        next = [tmat(ψ,K)*mem for K in ψ.A]
        s = rand(Categorical([prob(ψ,m) for m in next]))
        res[i] = s-1
        mem = next[s]/deno(ψ,next[s])
    end
    res
end

sample(ψ::MPSGeneral,n::Int) = sample!(ψ,n,memory(ψ))

function memory(ψ::MPSGeneral,samples::Vector{Int})
    mem = memory(ψ)
    for x in samples
        mem = tmat(ψ,ψ.A[x+1])*mem/deno(ψ,mem)
    end
    mem/deno(ψ,mem)
end

function prob(ψ::MPSGeneral,xs::Vector{Int},mem0::Vector{T}) where T
    mem = deepcopy(mem0)
    for x in xs
        mem = tmat(ψ,ψ.A[x+1])*mem
    end
    prob(ψ,mem)
end


bhatta(p1::Float64,p2::Float64)   = sqrt(p1*p2)
chi2(p1::Float64,p2::Float64)     = p1^2/p2
relentro(p1::Float64,p2::Float64) = (p1 ≈ 0.) ? 0.0 : p1*log2(p1/p2)

for op = (:bhatta, :chi, :relentro)
    @eval function $op(ψ1::T1,ψ2::T2,L::Int,history::Vector{Int}; 
                       verbose=false) where {T1<:MPSGeneral,T2<:MPSGeneral}
        mem1 = memory(ψ1,history)
        mem2 = memory(ψ2,history)
        @ass nkrauss(ψ1) == nkrauss(ψ2) "simulators with different alphabets"
        res = 0.0
        for it in Iterators.product([0:nkrauss(ψ1)-1 for i=1:L]...)
            x = [it...]
            p1 = prob(ψ1,x,mem1)
            p2 = prob(ψ2,x,mem2)
            verbose && println("prob ", it, ": ", p1, ", ", p2)
            res += $op(p1,p2)
        end
        return res
    end
end

function discrete_renewal(n)
	A = [zeros(n,n) for i=1:2]
    for k=1:n-1
		A[1][k+1,k] = 1
		A[2][k,n] = 1/sqrt(n)
    end
	A[2][n,n] = 1/sqrt(n)
    A
end

function discrete_renewal_cl(n)
	A = [zeros(n,n) for i=1:2]
    for k=1:n-1
        A[1][k+1,k] = sqrt((n-k)/(n+1-k))
        A[2][1,k] = 1/sqrt(n+1-k)
    end
    A[2][1,n] = 1.0
    A
end

function fitclassical(D,observations)
    d = length(unique(observations))
    h = HMM(randtransmat(D),[Categorical(ones(d) ./ d) for i=1:D])
    hmm, hist = fit_mle(h,observations .+1; maxiter=10^5)
    @ass hist.converged "Baum-Welch not convergent"
    B = hcat(map(x->x.p,hmm.B)...)
    A = [ zeros(D,D) for i=1:d ]
    for x=1:d
        for j=1:D, k=1:D
            A[x][j,k] = sqrt(B[x,k]*hmm.A[k,j])
        end
    end
    return MPSClassical(A)
end

function fitquantum(D,x; maxiter=1000, tol=1e-4, η=0.1, c=0.999, train=0, verbose=false)
    T = length(x)
    d = length(unique(x))
    A = reshape(vcat(randmps(D,d).A...),d,D,D)
    F = Vector{Vector{ComplexF64}}(undef,T)
    B = Vector{Vector{ComplexF64}}(undef,T)
    F[1] = zeros(ComplexF64,D)
    F[1][1]=1
    like = 2.0
    G = deepcopy(A)
    Ax(t) = A[x[t]+1,:,:]
    llhist = zeros(maxiter)
    lold = -1.0
    @ass train < T "wrong training set"

    for iter=1:maxiter
        n = 0.0
        for t=1:T-1
            F[t+1] = A[x[t]+1,:,:]*F[t] 
            n += -log(norm(F[t+1]))
            normalize!(F[t+1])
        end
        B[T] = A[x[T]+1,:,:]*F[T]
        n += -log(norm(B[T]))
        normalize!(B[T])
        for t=reverse(2:T)
            B[t-1] = A[x[t]+1,:,:]'*B[t]
            normalize!(B[t-1])
        end
        verbose && println("log-likelihood ",  n)
        llhist[iter] = n
        if abs(n-lold) < tol
            return MPSQuantum([A[i,:,:] for i=1:d]), llhist[1:iter]
        end
        lold = n
        G .= 0.0
        for t=1:T
            G[x[t]+1,:,:] -= B[t]*F[t]' / (F[t]'*A[x[t]+1,:,:]'*B[t])
        end
        # training
        if train > 0 
            B[train] = A[x[train]+1,:,:]*F[train]
            normalize!(B[train])
            for t=reverse(2:train)
                B[t-1] = A[x[t]+1,:,:]'*B[t]
                normalize!(B[t-1])
            end
            for t=1:train
                G[x[t]+1,:,:] += B[t]*F[t]' / (F[t]'*A[x[t]+1,:,:]'*B[t])
            end
        end
        # training
        A0 = reshape(A,d*D,D)
        verbose && println("norm ", norm(G))
        M = η/2*reshape(G,d*D,D)*A0'
        M -= M'
        A = reshape(inv(I+M)*(I-M)*A0,d,D,D)
        η *= c
    end
    #verbose || throw(ArgumentError("MPS: fit not converged"))
    verbose || println("MPS: fit not converged")
    MPSQuantum([A[i,:,:] for i=1:d]), llhist
end
        

function fitquantum2(D,x; maxiter=1000, η=0.1, tol=1e-4, train=0, verbose=false)
    T = length(x)
    d = length(unique(x))
    A = randmps(D,d).A
    F = Vector{Vector{ComplexF64}}(undef,T)
    B = Vector{Vector{ComplexF64}}(undef,T)
    F[1] = zeros(ComplexF64,D)
    F[1][D]=1
    lold = -1.0
    llhist = zeros(maxiter)
    @ass train < T "wrong training set"

    for iter=1:maxiter
        n=0.0
        for t=1:T-1
            F[t+1] = A[x[t]+1]*F[t] 
            n += -log(norm(F[t+1]))
            normalize!(F[t+1])
        end
        B[T] = A[x[T]+1]*F[T]
        n += -log(norm(F[T]))
        normalize!(B[T])
        for t=reverse(2:T)
            B[t-1] = A[x[t]+1]'*B[t]
            normalize!(B[t-1])
        end
        verbose && println("log-likelihood ", n)
        llhist[iter] = n
        if abs(n-lold) < tol
            return MPSQuantum(A), llhist[1:iter]
        end
        lold = n
        A0 = deepcopy(A)
        for t=1:T
            A0[x[t]+1] += η*B[t]*F[t]' / (F[t]'*A[x[t]+1]'*B[t])
        end
        if train > 0 
            B[train] = A[x[train]+1]*F[train]
            normalize!(B[train])
            for t=reverse(2:train)
                B[t-1] = A[x[t]+1]'*B[t]
                normalize!(B[t-1])
            end
            for t=1:train
                A0[x[t]+1] -= η*B[t]*F[t]' / (F[t]'*A[x[t]+1]'*B[t])
            end
        end
        A = mpsnormalized(A0)
    end
    #verbose || throw(ArgumentError("MPS: fit not converged"))
    verbose || println("MPS: fit not converged")
    if out 
        return MPSQuantum(A), llhist
    end
    MPSQuantum(A)
end

        

#include("../../SicPOVM/src/sicpovm.jl")

function sic(ψ::MPSQuantum)
    D,d = size(ψ),nkrauss(ψ)
    p = SicPOVM(D)
    P = Vector{Matrix{Float64}}(undef,d)
    for i=1:d
        P[i] = Matrix{Float64}(undef,D*D,D*D)
        for α=eachindex(p), β=eachindex(p)
            v = tr(Π(p,α)*ψ.A[i]*K(p,β)*ψ.A[i]')
            if real(v) > 0
                P[i][α,β] = real(v)
            else
                P[i][α,β] = 0
            end
        end
    end
    n =  sum(sum(P),dims=1)
    MPSClassical([sqrt.(a ./ n) for a in P])
end


function finetune(ψ::MPSClassical,η::MPSClassical) 
    D1,d1 = size(ψ),nkrauss(ψ)
    D2,d2 = size(η),nkrauss(η)
    @ass d1==d2 "wrong classical simulators"
    Xt = Array{Float64}(undef,D1,D2,D1,D2)
    for i=1:D1, j=1:D1, k=1:D2, l=1:D2
        Xt[i,k,j,l] = sum(abs2(ψ.A[x][i,j])*abs2(η.A[x][k,l]) for x=1:d1)
    end
    X = reshape(Xt,D1*D2,D1*D2)
end


using Plots,LaTeXStrings    # remove if you don't run the example
if false
	pgfplotsx()
    function plotta()
        Ns = [16, 32, 64]
        first = true
        p = nothing
        for N in Ns
            ψ = MPSQuantum(discrete_renewal(N))
            qs = zeros(N)
            fs = zeros(N)
            qs[N] = entropy(ψ)
            fs[N] = 1.0
            for n=2:N-1
                ϕ = compress(ψ,n) 
                qs[n] = entropy(ϕ)
                fs[n], _, _ = fidrate(ψ,ϕ)
            end
            if first
                p = plot(-2log.(fs[2:N]),qs[2:N]; label="N=$N", linewidth=2.5, legend=:topright,
                         size=(400,300), title="Discrete renewal", 
                         xaxis=L"\lim_{L\to\infty} D_{1/2}(\mathrm{exact},\mathrm{compressed})/L", 
                         yaxis=L"\mathrm{entropy~of~compressed~memory}")
                first = false
            else
                p = plot!(-2log.(fs[2:N]),qs[2:N]; label="N=$N", linewidth=2.5)
            end
        end
        p
    end
    plot(plotta())
#   savefig("fcastentrodiv.pdf")
end




