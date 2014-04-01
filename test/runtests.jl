using TensorOperations
using Base.Test  

# LEVEL 1 OPERATIONS:
# tensorcopy tests
A=randn((3,5,4,6))
p=randperm(4)
B1=permutedims(A,p)
B2=tensorcopy(p,A,1:4)
@test vecnorm(B1-B2)<eps()*vecnorm(B1)
@test_throws tensorcopy(1:3,A,1:4)
@test_throws tensorcopy([1,2,2,4],A,1:4)

# test tensoradd
B=randn((5,6,3,4))
p=[3,1,4,2]
C1=tensoradd(A,p,B,1:4)
C2=A+permutedims(B,p)
@test_approx_eq(vecnorm(C1-C2),0)
C3=0.8*A+1.2*permutedims(B,p)
C4=similar(C3)
fill!(C4,0)
TensorOperations.tensoradd!(0.0,C4,1:4,1.0,A,1:4)
TensorOperations.tensoradd!(0.8,C4,p,1.2,B,1:4)
@test vecnorm(C3-C4)<eps()*vecnorm(C3)
@test_throws tensoradd(A,1:4,B,1:4)

# test tensordot
c1=tensordot(A,p,B,1:4)
c2=dot(reshape(A,length(A)),reshape(permutedims(B,p),length(B)))
@test_approx_eq(c1,c2)

# test different versions of level 1 methods,
# with changing element type and with strides
Abig=randn((30,30,30,30))
A=sub(Abig,1:3:30,2:2:15,5:4:30,4:3:30)
p=[3,1,4,2]
Bbig=zeros(Complex128,(50,50,50,50))
B=sub(Bbig,13:19,11:4:50,15:4:50,4:3:24)
Acopy=tensorcopy(1:4,A,1:4)
Bcopy=tensorcopy(1:4,B,1:4)
TensorOperations.tensorcopy!(B,p,A,1:4)
TensorOperations.tensorcopy!(Bcopy,p,Acopy,1:4)
@test vecnorm(B-Bcopy)<eps()*vecnorm(B)
Bbig=rand(Complex128,(50,50,50,50))
B=sub(Bbig,13:19,11:4:50,15:4:50,4:3:24)
Bcopy=tensorcopy(1:4,B,1:4)
Acopy=tensorcopy(p,A,1:4)
TensorOperations.tensoradd!(0.5,B,p,1.2,A,1:4)
@test vecnorm(B-0.5*Bcopy-1.2*Acopy)<eps()*vecnorm(B)
Bcopy=0.5*Bcopy+1.2*Acopy
@test_approx_eq(tensordot(B,p,A,1:4),dot(reshape(Bcopy,length(Bcopy)),reshape(Acopy,length(Acopy))))
