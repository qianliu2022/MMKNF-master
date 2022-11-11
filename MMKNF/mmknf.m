
function [S] = mmknf(X, c, k)
t0 = tic;
order = 2;
no_dim=c;

ITER = 30;
num = size(X,1);
r = -1;
beta = 0.8;
D_Kernels = multipleK(X);
clear X
alphaK = 1/(size(D_Kernels,3))*ones(1,size(D_Kernels,3));%对w初始化
distX = mean(D_Kernels,3);
[distX1, idx] = sort(distX,2);% sort(A):对一维或二维矩阵进行升序排序,并返回排序后的矩阵;当A为二维矩阵时,对矩阵的每一列分别进行排序。
A = zeros(num);%返回一个全0数组
di = distX1(:,2:(k+2));%a(:,4)则表示取第4列所有数据 此处是取dix1的2到12列的数据
rr = 0.5*(k*di(:,k+1)-sum(di(:,1:k),2));
id = idx(:,2:k+2);
temp = (repmat(di(:,k+1),1,size(di,2))-di)./repmat((k*di(:,k+1)-sum(di(:,1:k),2)+eps),1,size(di,2));
a = repmat([1:num]',1,size(id,2));
A(sub2ind(size(A),a(:),id(:)))=temp(:);
if r <= 0
    r = mean(rr);
end
lambda = max((mean(rr)),0);
A(isnan(A))=0;
S0 = max(max(distX))-distX;
S0 = Network_Diffusion(S0,k);
S0 = NE_dn(S0,'ave');%每列加起来等于1
S= (S0 + S0')/2;
D0 = diag(sum(S,order));
L0= D0-S;
[F, temp, evs]=eig1(L0, c, 0);
F = NE_dn(F,'ave');
for iter = 1:NITER
    distf = L2_distance_1(F',F');
    A = zeros(num);
    b = idx(:,2:end);
    a = repmat([1:num]',1,size(b,2));
    inda = sub2ind(size(A),a(:),b(:));
    ad = reshape((distX(inda)+lambda*distf(inda))/2/r,num,size(b,2));
    ad = projsplx_c(-ad')';
    A(inda) = ad(:);
    A(isnan(A))=0;
    S = (1-beta)*A+beta*S;
    S = Network_Diffusion(S,k);
    S= (S + S')/2;
    D = diag(sum(S,order));
    L = D - S;
    F_old = F;
    [F, temp, ev]=eig1(L, c, 0);
    F = NE_dn(F,'ave');
    F = (1-beta)*F_old+beta*F;
    evs(:,iter+1) = ev;
    for i = 1:size(D_Kernels,3)
        temp = (eps+D_Kernels(:,:,i)).*(eps+S);
        DD(i) = mean(sum(temp));
    end
    alphaK0 = umkl_bo(DD);
    alphaK0 = alphaK0/sum(alphaK0);
    alphaK = (1-beta)*alphaK + beta*alphaK0;
    alphaK = alphaK/sum(alphaK);
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    converge(iter) = fn2-fn1;
    if iter<10
        if (ev(end) > 0.000001)
            lambda = 1.5*lambda;
            r = r/1.01;
        end
    else
        if (converge(iter)>1.01*converge(iter-1))
            S = S_old;
            if converge(iter-1) > 0.2
                warning('Maybe you should set a larger value of c');
            end
            break;
        end
    end
    S_old = S;
    distX = Kbeta(D_Kernels,alphaK');
    [distX1, idx] = sort(distX,2);% sort(A):对一维或二维矩阵进行升序排序,并返回排序后的矩阵;当A为二维矩阵时,对矩阵的每一列分别进行排序。
end;
LF = F;
D = diag(sum(S,order));
L = D - S;
[U,D] = eig(L);
[m,n]=size(S);
b=eye(m,n);
S=S+1*b;
C=S>=0
S=S.* C
%c 即保留a中大于等于0的元素原来的值，而将原小于0的元素用0代替。
timeOurs = toc(t0); 
end
function thisP = umkl_bo(D,beta)
if nargin<2
    beta = 1/length(D);
end
tol = 1e-4;
u = 20;logU = log(u);
[H, thisP] = Hbeta(D, beta);
betamin = -Inf;
betamax = Inf;
% Evaluate whether the perplexity is within tolerance
Hdiff = H - logU;
tries = 0;
while (abs(Hdiff) > tol) && (tries < 30)
    
    % If not, increase or decrease precision
    if Hdiff > 0
        betamin = beta;
        if isinf(betamax)
            beta = beta * 2;
        else
            beta = (beta + betamax) / 2;
        end
    else
        betamax = beta;
        if isinf(betamin)
            beta = beta / 2;
        else
            beta = (beta + betamin) / 2;
        end
    end
    
    % Recompute the values
    [H, thisP] = Hbeta(D, beta);
    Hdiff = H - logU;
    tries = tries + 1;
end
end

function [H, P] = Hbeta(D, beta)
D = (D-min(D))/(max(D) - min(D)+eps);
P = exp(-D * beta);
sumP = sum(P);
H = log(sumP) + beta * sum(D .* P) / sumP;
P = P / sumP;
end

function D_Kernels = multipleK(x)
N = size(x,1);
%size(X,1),返回矩阵X的行数；
%样本的个数 n
KK = 0;
sigma = [2:-0.25:1];
Diff = (dist2_1(x));%计算X样本间的的欧几里得距离 n×n的
[T,INDEX]=sort(Diff,2);%sort函数是排序 按行排序 INDEX索引名称
[m,n]=size(Diff);% size（）：获取矩阵的行数和列数 m=m=n
allk = 10:2:30;
t=1;
for l = 1:length(allk)%是L不是1
    if allk(l) < (size(x,1)-1)
        TT=mean(T(:,2:(allk(l)+1)),2)+eps;
        %mean（X,2）返回每行的平均值 
        %a(:,4)则表示取第4列所有数据 此处是取dix1的2到12列的数据 EPS是MATLAB中的函数,大约是 2e-16
      
        Sig=(repmat(TT,1,n)+repmat(TT',n,1))/2;
      % B = repmat(A,m,n)，将矩阵 A 复制 m×n 块，
      %即把 A 作为 B 的元素，B 由 m×n 个 A 平铺而成。
      %B 的维数是 [size(A,1)*m, size(A,2)*n] 。
        Sig=Sig.*(Sig>eps)+eps;
        for j = 1:length(sigma)
            W=normpdf(Diff,0,sigma(j)*Sig);
%正态分布
% normpdf：正态概率密度函数
% 
% Y = normpdf(X,mu,sigma)
% 
% mu：均值
% 
% sigma：标准差
% 
% Y：正态概率密度函数在x处的值
            Kernels(:,:,KK+t) = (W + W')/2;
            t = t+1;
        end
    end
end

for i = 1:size(Kernels,3)
    %返回矩阵维度的值:55
    K = Kernels(:,:,i);
    % a是一个三维矩阵，a(:,:,1)表示取a矩阵第一页的所有行和列。
    k = 1./sqrt(diag(K)+1);
    %设A为m×n矩阵 diag(A) 函数用于提取矩阵A主对角线元素，产生一个具有min(m,n)[且syu的大小等于x较小的维数]个元素的列向量。
    %sqrt() 计算 X 的每个元素的平方根。然后取倒数
    G = K;
    D_Kernels(:,:,i) = (repmat(diag(G),1,length(G)) +repmat(diag(G)',length(G),1) - 2*G)/2;
    D_Kernels(:,:,i) = D_Kernels(:,:,i) - diag(diag(D_Kernels(:,:,i)));
  
end

end




function W = Network_Diffusion(A, K)
A = A-diag(diag(A));
%调用 diag 两次 将返回一个 包含 原始矩阵的对角线上元素 的 对角矩阵
P = (dominateset(double(abs(A)),min(K,length(A)-1))).*sign(A);
%sign(整数)=1; 
%sign(负数)=-1; 
%sign（零）=0
%abs(A) 取绝对值 double转化为浮点类型的数据
DD = sum(abs(P'));
P = P + (eye(length(P))+diag(sum(abs(P'))));
P = (TransitionFields(P));
[U,D] = eig(P);
d = real((diag(D))+eps);
alpha = 0.8;
beta = 2;
d = (1-alpha)*d./(1-alpha*d.^beta);


D = diag(real(d));
W = U*D*U';

W = (W.*(1-eye(length(W))))./repmat(1-diag(W),1,length(W));
D=sparse(1:length(DD),1:length(DD),DD);
W=D*(W);
W = (W+W')/2;

end


