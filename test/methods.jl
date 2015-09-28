# test simple methods
#---------------------
# test tensorcopy
A=randn((3,5,4,6))
p=randperm(4)
C1=permutedims(A,p)
C2=tensorcopy(A,1:4,p)
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)
@test_throws TensorOperations.IndexError tensorcopy(A,1:3,1:4)
@test_throws TensorOperations.IndexError tensorcopy(A,[1,2,2,4],1:4)

# test tensoradd
B=randn((5,6,3,4))
p=[3,1,4,2]
C1=tensoradd(A,p,B,1:4)
C2=A+permutedims(B,p)
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)
@test_throws DimensionMismatch tensoradd(A,1:4,B,1:4)

# test tensortrace
A=randn(50,100,100)
C1=tensortrace(A,[:a,:b,:b])
C2=zeros(50)
for i=1:50
    for j=1:100
        C2[i]+=A[i,j,j]
    end
end
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)
A=randn(3,20,5,3,20,4,5)
C1=tensortrace(A,[:a,:b,:c,:d,:b,:e,:c],[:e,:a,:d])
C2=zeros(4,3,3)
for i1=1:4, i2=1:3, i3=1:3
    for j1=1:20,j2=1:5
        C2[i1,i2,i3]+=A[i2,j1,j2,i3,j1,i1,j2]
    end
end
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)

# test tensorcontract
A=randn(3,20,5,3,4)
B=randn(5,6,20,3)
C1=tensorcontract(A,[:a,:b,:c,:d,:e],B,[:c,:f,:b,:g],[:a,:g,:e,:d,:f];method=:BLAS)
C2=tensorcontract(A,[:a,:b,:c,:d,:e],B,[:c,:f,:b,:g],[:a,:g,:e,:d,:f];method=:native)
C3=zeros(3,3,4,3,6)
for a=1:3, b=1:20, c=1:5, d=1:3, e=1:4, f=1:6, g=1:3
    C3[a,g,e,d,f] += A[a,b,c,d,e]*B[c,f,b,g]
end
@test vecnorm(C1-C3)<eps()*sqrt(length(C1))*vecnorm(C1+C3)
@test vecnorm(C2-C3)<eps()*sqrt(length(C1))*vecnorm(C2+C3)
@test_throws TensorOperations.IndexError tensorcontract(A,[:a,:b,:c,:d],B,[:c,:f,:b,:g])
@test_throws TensorOperations.IndexError tensorcontract(A,[:a,:b,:c,:a,:e],B,[:c,:f,:b,:g])

# test tensorproduct
A=randn(5,5,5,5)
B=rand(Complex128,(5,5,5,5))
C1=reshape(tensorproduct(A,[1,2,3,4],B,[5,6,7,8],[1,2,5,6,3,4,7,8]),(5*5*5*5,5*5*5*5))
C2=kron(reshape(B,(25,25)),reshape(A,(25,25)))
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)
@test_throws TensorOperations.IndexError tensorproduct(A,[:a,:b,:c,:d],B,[:d,:e,:f,:g])
@test_throws TensorOperations.IndexError tensorproduct(A,[:a,:b,:c,:d],B,[:e,:f,:g,:h],[:a,:b,:c,:d,:e,:f,:g,:i])

# test index notation
#---------------------
# Da=10
# Db=15
# Dc=4
# Dd=8
# De=6
# Df=7
# Dg=3
# Dh=2
# A=rand(Complex128,(Da,Dc,Df,Da,De,Db,Db,Dg))
# B=rand(Complex128,(Dc,Dh,Dg,De,Dd))
# C=rand(Complex128,(Dd,Dh,Df))
# D=rand(Complex128,(Dd,Df,Dh))
# D[l"d,f,h"]=A[l"a,c,f,a,e,b,b,g"]*B[l"c,h,g,e,d"]+0.5*C[l"d,h,f"]
# @test_approx_eq(vecnorm(D),sqrt(abs(scalar(D[l"d,f,h"]*conj(D[l"d,f,h"])))))
# @test_throws IndexError D[l"a,a,a"]
# @test_throws IndexError D[l"a,b,c,d"]
# @test_throws IndexError D[l"a,b"]

# test in-place methods
#-----------------------
# test different versions of in-place methods,
# with changing element type and with nontrivial strides

# tensorcopy!
Abig=randn((30,30,30,30))
A=sub(Abig,1+3*(0:9),2+2*(0:6),5+4*(0:6),4+3*(0:8))
p=[3,1,4,2]
Cbig=zeros(Complex128,(50,50,50,50))
C=sub(Cbig,13+(0:6),11+4*(0:9),15+4*(0:8),4+3*(0:6))
Acopy=tensorcopy(A,1:4,1:4)
Ccopy=tensorcopy(C,1:4,1:4)
TensorOperations.tensorcopy!(A,1:4,C,p)
TensorOperations.tensorcopy!(Acopy,1:4,Ccopy,p)
@test vecnorm(C-Ccopy)<eps()*sqrt(length(C))*vecnorm(C+Ccopy)
@test_throws TensorOperations.IndexError TensorOperations.tensorcopy!(A,1:3,C,p)
@test_throws DimensionMismatch TensorOperations.tensorcopy!(A,p,C,p)
@test_throws TensorOperations.IndexError TensorOperations.tensorcopy!(A,1:4,C,[1,1,2,3])

# tensoradd!
Cbig=zeros(Complex128,(50,50,50,50))
C=sub(Cbig,13+(0:6),11+4*(0:9),15+4*(0:8),4+3*(0:6))
Acopy=tensorcopy(A,1:4,p)
Ccopy=tensorcopy(C,1:4,1:4)
alpha=randn()
beta=randn()
TensorOperations.tensoradd!(alpha,A,1:4,beta,C,p)
Ccopy=beta*Ccopy+alpha*Acopy
@test vecnorm(C-Ccopy)<eps()*sqrt(length(C))*vecnorm(C+Ccopy)
@test_throws TensorOperations.IndexError TensorOperations.tensoradd!(1.2,A,1:3,0.5,C,p)
@test_throws DimensionMismatch TensorOperations.tensoradd!(1.2,A,p,0.5,C,p)
@test_throws TensorOperations.IndexError TensorOperations.tensoradd!(1.2,A,1:4,0.5,C,[1,1,2,3])

# tensortrace!
Abig=rand((30,30,30,30))
A=sub(Abig,1+3*(0:8),2+2*(0:14),5+4*(0:6),7+2*(0:8))
Bbig=rand(Complex128,(50,50))
B=sub(Bbig,13+(0:14),3+5*(0:6))
Acopy=tensorcopy(A,1:4)
Bcopy=tensorcopy(B,1:2)
alpha=randn()
beta=randn()
TensorOperations.tensortrace!(alpha,A,[:a,:b,:c,:a],beta,B,[:b,:c])
Bcopy=beta*Bcopy
for i=1+(0:8)
    Bcopy+=alpha*slice(A,i,:,:,i)
end
@test vecnorm(B-Bcopy)<eps()*vecnorm(B+Bcopy)*sqrt(length(B))
@test_throws TensorOperations.IndexError TensorOperations.tensortrace!(alpha,A,[:a,:b,:c],beta,B,[:b,:c])
@test_throws DimensionMismatch TensorOperations.tensortrace!(alpha,A,[:a,:b,:c,:a],beta,B,[:c,:b])
@test_throws TensorOperations.IndexError TensorOperations.tensortrace!(alpha,A,[:a,:b,:a,:a],beta,B,[:c,:b])
@test_throws DimensionMismatch TensorOperations.tensortrace!(alpha,A,[:a,:b,:a,:c],beta,B,[:c,:b])

# tensorcontract!
Abig=rand((30,30,30,30))
A=sub(Abig,1+3*(0:8),2+2*(0:14),5+4*(0:6),7+2*(0:8))
Bbig=rand(Complex128,(50,50,50))
B=sub(Bbig,3+5*(0:6),7+2*(0:7),13+(0:14))
Cbig=rand(Complex64,(40,40,40))
C=sub(Cbig,3+2*(0:8),13+(0:8),7+3*(0:7))
Acopy=tensorcopy(A,1:4)
Bcopy=tensorcopy(B,1:3)
Ccopy=tensorcopy(C,1:3)
alpha=randn()
beta=randn()
Ccopy=beta*Ccopy
for d=1+(0:8),a=1+(0:8),e=1+(0:7)
    for b=1+(0:14),c=1+(0:6)
        Ccopy[d,a,e]+=alpha*A[a,b,c,d]*conj(B[c,e,b])
    end
end
TensorOperations.tensorcontract!(alpha,A,[:a,:b,:c,:d],'N',B,[:c,:e,:b],'C',beta,C,[:d,:a,:e];method=:BLAS)
@test vecnorm(C-Ccopy)<eps(Float32)*vecnorm(C+Ccopy)*sqrt(length(C))
Cbig=rand(Complex64,(40,40,40))
C=sub(Cbig,3+2*(0:8),13+(0:8),7+3*(0:7))
Ccopy=tensorcopy(C,1:3)
Ccopy=beta*Ccopy
for d=1+(0:8),a=1+(0:8),e=1+(0:7)
    for b=1+(0:14),c=1+(0:6)
        Ccopy[d,a,e]+=alpha*A[a,b,c,d]*conj(B[c,e,b])
    end
end
TensorOperations.tensorcontract!(alpha,A,[:a,:b,:c,:d],'N',B,[:c,:e,:b],'C',beta,C,[:d,:a,:e];method=:native)
@test vecnorm(C-Ccopy)<eps(Float32)*vecnorm(C+Ccopy)*sqrt(length(C))
@test_throws TensorOperations.IndexError TensorOperations.tensorcontract!(alpha,A,[:a,:b,:c,:a],'N',B,[:c,:e,:b],'N',beta,C,[:d,:a,:e])
@test_throws TensorOperations.IndexError TensorOperations.tensorcontract!(alpha,A,[:a,:b,:c,:d],'N',B,[:c,:b],'N',beta,C,[:d,:a,:e])
@test_throws TensorOperations.IndexError TensorOperations.tensorcontract!(alpha,A,[:a,:b,:c,:d],'N',B,[:c,:e,:b],'N',beta,C,[:d,:e])
@test_throws DimensionMismatch TensorOperations.tensorcontract!(alpha,A,[:a,:b,:c,:d],'N',B,[:c,:e,:b],'N',beta,C,[:d,:e,:a])
