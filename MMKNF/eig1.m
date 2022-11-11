function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

if nargin < 2
    c = size(A,1);% c = a的行数
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A'); %和原来的矩阵一样？？？
end;
[v,d] = eig(A);
% E=eig(A)：求矩阵A的全部特征值，构成列向量E。
% [v,d] = eig(A) 求矩阵A的全部特征值，构成对角阵d，并产生矩阵v，v各列是相应的特征向量
d = diag(d);
if isMax == 0
    [d1, idx] = sort(d);
else
    [d1, idx] = sort(d,'descend');
end;

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = real(v(:,idx1));
eigval_full = d(idx);