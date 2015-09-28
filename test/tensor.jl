# test index notation using @tensor macro
#-----------------------------------------
A=randn((3,5,4,6))
p=randperm(4)
C1=permutedims(A,p)
@eval @tensor C2[$(p...)] := A[1,2,3,4]
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)
@test_throws TensorOperations.IndexError begin
    @tensor C[1,2,3,4] := A[1,2,3]
end
@test_throws TensorOperations.IndexError begin
    @tensor C[1,2,3,4] := A[1,2,2,4]
end

B=randn((5,6,3,4))
p=[3,1,4,2]
@tensor C1[3,1,4,2] := A[3,1,4,2] + B[1,2,3,4]
C2=A+permutedims(B,p)
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)
@test_throws DimensionMismatch begin
    @tensor C[1,2,3,4] := A[1,2,3,4] + B[1,2,3,4]
end

A=randn(50,100,100)
@tensor C1[a] := A[a,b,b]
C2=zeros(50)
for i=1:50
    for j=1:100
        C2[i]+=A[i,j,j]
    end
end
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)
A=randn(3,20,5,3,20,4,5)
@tensor C1[e,a,d] := A[a,b,c,d,b,e,c]
C2=zeros(4,3,3)
for i1=1:4, i2=1:3, i3=1:3
    for j1=1:20,j2=1:5
        C2[i1,i2,i3]+=A[i2,j1,j2,i3,j1,i1,j2]
    end
end
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)

A=randn(3,20,5,3,4)
B=randn(5,6,20,3)
@tensor C1[a,g,e,d,f] := A[a,b,c,d,e]*B[c,f,b,g]
C2=zeros(3,3,4,3,6)
for a=1:3, b=1:20, c=1:5, d=1:3, e=1:4, f=1:6, g=1:3
    C2[a,g,e,d,f] += A[a,b,c,d,e]*B[c,f,b,g]
end
@test_throws TensorOperations.IndexError begin
    @tensor A[a,b,c,d]*B[c,f,b,g]
end

A=randn(5,5,5,5)
B=rand(Complex128,(5,5,5,5))
@tensor C1[1,2,5,6,3,4,7,8] := A[1,2,3,4]*B[5,6,7,8]
C2=reshape(kron(reshape(B,(25,25)),reshape(A,(25,25))),(5,5,5,5,5,5,5,5))
@test vecnorm(C1-C2)<eps()*sqrt(length(C1))*vecnorm(C1+C2)
@test_throws TensorOperations.IndexError begin
    @tensor C[a,b,c,d,e,f,g,i] := A[a,b,c,d]*B[e,f,g,h]
end

Da=10
Db=15
Dc=4
Dd=8
De=6
Df=7
Dg=3
Dh=2
A=rand(Complex128,(Da,Dc,Df,Da,De,Db,Db,Dg))
B=rand(Complex128,(Dc,Dh,Dg,De,Dd))
C=rand(Complex128,(Dd,Dh,Df))
@tensor D1[d,f,h] := A[a,c,f,a,e,b,b,g]*B[c,h,g,e,d]+0.5*C[d,h,f]
D2=zeros(Complex128,(Dd,Df,Dh))
for d=1:Dd, f=1:Df, h=1:Dh
    D2[d,f,h] += 0.5*C[d,h,f]
    for a=1:Da, b=1:Db, c=1:Dc, e=1:De, g=1:Dg
        D2[d,f,h] += A[a,c,f,a,e,b,b,g]*B[c,h,g,e,d]
    end
end
@test vecnorm(D1-D2)<eps()*sqrt(length(D1))*vecnorm(D1+D2)
@test_approx_eq(vecnorm(D1),sqrt(abs(@tensor scalar(D1[d,f,h]*conj(D1[d,f,h])))))

Abig=randn((30,30,30,30))
A=sub(Abig,1+3*(0:9),2+2*(0:6),5+4*(0:6),4+3*(0:8))
p=[3,1,4,2]
Cbig=zeros(Complex128,(50,50,50,50))
C=sub(Cbig,13+(0:6),11+4*(0:9),15+4*(0:8),4+3*(0:6))
Acopy = copy(A)
Ccopy = copy(C)
@tensor C[3,1,4,2] = A[1,2,3,4]
@tensor Ccopy[3,1,4,2] = Acopy[1,2,3,4]
@test vecnorm(C-Ccopy)<eps()*sqrt(length(C))*vecnorm(C+Ccopy)
@test_throws TensorOperations.IndexError begin
    @tensor C[3,1,4,2] = A[1,2,3]
end
@test_throws DimensionMismatch begin
    @tensor C[3,1,4,2] = A[3,1,4,2]
end
@test_throws TensorOperations.IndexError begin
    @tensor C[1,1,2,3] = A[1,2,3,4]
end

Cbig=zeros(Complex128,(50,50,50,50))
C=sub(Cbig,13+(0:6),11+4*(0:9),15+4*(0:8),4+3*(0:6))
Acopy=tensorcopy(A,1:4,p)
Ccopy=copy(C)
alpha=randn()
beta=randn()
@tensor C[3,1,4,2] = beta*C[3,1,4,2] + alpha*A[1,2,3,4]
Ccopy=beta*Ccopy+alpha*Acopy
@test vecnorm(C-Ccopy)<eps()*sqrt(length(C))*vecnorm(C+Ccopy)
@test_throws TensorOperations.IndexError begin
    @tensor C[3,1,4,2] = 0.5*C[3,1,4,2] + 1.2*A[1,2,3]
end
@test_throws DimensionMismatch  begin
    @tensor C[3,1,4,2] = 0.5*C[3,1,4,2] + 1.2*A[3,1,2,4]
end
@test_throws TensorOperations.IndexError  begin
    @tensor C[1,1,2,3] = 0.5*C[1,1,2,3] + 1.2*A[1,2,3,4]
end

Abig=rand((30,30,30,30))
A=sub(Abig,1+3*(0:8),2+2*(0:14),5+4*(0:6),7+2*(0:8))
Bbig=rand(Complex128,(50,50))
B=sub(Bbig,13+(0:14),3+5*(0:6))
Acopy=copy(A)
Bcopy=copy(B)
alpha=randn()
@tensor B[b,c] += alpha*A[a,b,c,a]
for i=1+(0:8)
    Bcopy+=alpha*slice(A,i,:,:,i)
end
@test vecnorm(B-Bcopy)<eps()*vecnorm(B+Bcopy)*sqrt(length(B))
@test_throws TensorOperations.IndexError begin
    @tensor B[b,c] += alpha*A[a,b,c]
end
@test_throws DimensionMismatch begin
    @tensor B[c,b] += alpha*A[a,b,c,a]
end
@test_throws TensorOperations.IndexError begin
    @tensor B[c,b] += alpha*A[a,b,a,a]
end
@test_throws DimensionMismatch begin
    @tensor B[c,b] += alpha*A[a,b,a,c]
end

Abig=rand((30,30,30,30))
A=sub(Abig,1+3*(0:8),2+2*(0:14),5+4*(0:6),7+2*(0:8))
Bbig=rand(Complex128,(50,50,50))
B=sub(Bbig,3+5*(0:6),7+2*(0:7),13+(0:14))
Cbig=rand(Complex64,(40,40,40))
C=sub(Cbig,3+2*(0:8),13+(0:8),7+3*(0:7))
Acopy=copy(A)
Bcopy=copy(B)
Ccopy=copy(C)
alpha=randn()
for d=1+(0:8),a=1+(0:8),e=1+(0:7)
    for b=1+(0:14),c=1+(0:6)
        Ccopy[d,a,e] -=alpha*A[a,b,c,d]*conj(B[c,e,b])
    end
end
@tensor C[d,a,e] -= alpha*A[a,b,c,d]*conj(B[c,e,b])
@test vecnorm(C-Ccopy)<eps(Float32)*vecnorm(C+Ccopy)*sqrt(length(C))
Cbig=rand(Complex64,(40,40,40))
C=sub(Cbig,3+2*(0:8),13+(0:8),7+3*(0:7))
Ccopy=copy(C)

for d=1+(0:8),a=1+(0:8),e=1+(0:7)
    for b=1+(0:14),c=1+(0:6)
        Ccopy[d,a,e] += alpha*A[a,b,c,d]*conj(B[c,e,b])
    end
end
@tensor C[d,a,e] += alpha*A[a,b,c,d]*conj(B[c,e,b])
@test vecnorm(C-Ccopy)<eps(Float32)*vecnorm(C+Ccopy)*sqrt(length(C))
@test_throws TensorOperations.IndexError begin
    @tensor C[d,a,e] += alpha*A[a,b,c,a]*B[c,e,b]
end
@test_throws TensorOperations.IndexError begin
    @tensor C[d,a,e] += alpha*A[a,b,c,d]*B[c,b]
end
@test_throws TensorOperations.IndexError begin
    @tensor C[d,e] += alpha*A[a,b,c,d]*B[c,e,b]
end
@test_throws DimensionMismatch begin
    @tensor C[d,e,a] += alpha*A[a,b,c,d]*B[c,e,b]
end
