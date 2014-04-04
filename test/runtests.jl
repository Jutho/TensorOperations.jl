using TensorOperations
using Base.Test  

# test simple methods
#---------------------
# test tensorcopy
A=randn((3,5,4,6))
p=randperm(4)
B1=permutedims(A,p)
B2=tensorcopy(A,1:4,p)
@test vecnorm(B1-B2)<eps()*sqrt(length(B1))*vecnorm(B1+B2)
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
TensorOperations.tensoradd!(1.0,A,1:4,0.0,C4,1:4)
TensorOperations.tensoradd!(1.2,B,1:4,0.8,C4,p)
@test vecnorm(C3-C4)<eps()*sqrt(length(C3))*vecnorm(C3+C4)
@test_throws tensoradd(A,1:4,B,1:4)

# test tensortrace
A=randn(50,100,100)
B=tensortrace(A,[:a,:b,:b])
C=zeros(50)
for i=1:50
    for j=1:100
        C[i]+=A[i,j,j]
    end
end
@test vecnorm(B-C)<eps()*vecnorm(B)
A=randn(3,20,5,3,20,4,5)
B=tensortrace(A,[:a,:b,:c,:d,:b,:e,:c],[:e,:a,:d])
C=zeros(4,3,3)
for i1=1:4, i2=1:3, i3=1:3
    for j1=1:20,j2=1:5
        C[i1,i2,i3]+=A[i2,j1,j2,i3,j1,i1,j2]
    end
end
@test vecnorm(B-C)<eps()*sqrt(length(B))*vecnorm(B+C)

# test tensorcontract
A=randn(3,20,5,3,4)
B=randn(5,6,20,3)
C=tensorcontract(A,[:a,:b,:c,:d,:e],B,[:c,:f,:b,:g],[:a,:g,:e,:d,:f];method=:BLAS)
D=tensorcontract(A,[:a,:b,:c,:d,:e],B,[:c,:f,:b,:g],[:a,:g,:e,:d,:f];method=:native)
@test vecnorm(C-D)<eps()*sqrt(length(C))*vecnorm(C+D)
@test_throws tensorcontract(A,[:a,:b,:c,:d],B,[:c,:f,:b,:g])
@test_throws tensorcontract(A,[:a,:b,:c,:a,:e],B,[:c,:f,:b,:g])

# test index notation
#---------------------
Da=10
Db=15
Dc=4
Dd=8
De=6
Df=7
Dg=3
Dh=2
A=randn(Da,Dc,Df,Da,De,Db,Db,Dg)
B=randn(Dc,Dh,Dg,De,Dd)
C=randn(Dd,Dh,Df)
D=randn(Dd,Df,Dh)
D[l"d,f,h"]=A[l"a,c,f,a,e,b,b,g"]*B[l"c,h,g,e,d"]+0.5*C[l"d,h,f"]

# test in-place methods
#-----------------------
# test different versions of in-place methods,
# with changing element type and with nontrivial strides
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
@test_approx_eq(tensordot(B,p,'C',A,1:4,'N'),dot(reshape(Bcopy,length(Bcopy)),reshape(Acopy,length(Acopy))))
@test_throws tensordot(B,p,'C',A,1:4,'T')