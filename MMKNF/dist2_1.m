function n2 = dist2(x, c)
%DIST2	Calculates squared distance between two sets of points.
%
%	Description
%	D = DIST2(X, C) takes two matrices of vectors and calculates the
%	squared Euclidean distance between them.  Both matrices must be of
%	the same column dimension.  If X has M rows and N columns, and C has
%	L rows and N columns, then the result has M rows and L columns.  The
%	I, Jth entry is the  squared distance from the Ith row of X to the
%	Jth row of C.
%
%	See also
%	GMMACTIV, KMEANS, RBFFWD
%
%   DIST2计算两组点之间的平方距离。

%

%描述

%D=DIST2（X，C）取两个向量矩阵并计算

%它们之间的欧几里得距离的平方。两个矩阵都必须是

%相同的列维度。如果X有M行和N列，而C有

%L行和N列，则结果有M行和L列。这个

%I，Jth entry是从X的第I行到

%第j排C。

%

%另见

%GMMACTIV、KMEANS、RBFwd

%
%	Copyright (c) Ian T Nabney (1996-2001)

if nargin<2
    c = x;
end
[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
	error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
  ones(ndata, 1) * sum((c.^2)',1) - ...
  2.*(x*(c'));

% Rounding errors occasionally cause negative entries in n2
if any(any(n2<0))
  n2(n2<0) = 0;
  %DIST2计算两组点之间的平方距离。欧式距离KNN

%

%描述

%D=DIST2（X，C）取两个向量矩阵并计算

%它们之间的欧几里得距离的平方。两个矩阵都必须是

%相同的列维度。如果X有M行和N列，而C有

%L行和N列，则结果有M行和L列。这个

%I，Jth entry是从X的第I行到

%第j排C。

%

%另见

%GMMACTIV、KMEANS、RBFwd

%版权所有（c）Ian T Nabney（1996-2001）

%舍入错误有时会导致n2中出现负值条目
end