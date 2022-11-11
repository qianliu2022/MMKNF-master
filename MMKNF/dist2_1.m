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
%   DIST2���������֮���ƽ�����롣

%

%����

%D=DIST2��X��C��ȡ�����������󲢼���

%����֮���ŷ����þ����ƽ�����������󶼱�����

%��ͬ����ά�ȡ����X��M�к�N�У���C��

%L�к�N�У�������M�к�L�С����

%I��Jth entry�Ǵ�X�ĵ�I�е�

%��j��C��

%

%���

%GMMACTIV��KMEANS��RBFwd

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
  %DIST2���������֮���ƽ�����롣ŷʽ����KNN

%

%����

%D=DIST2��X��C��ȡ�����������󲢼���

%����֮���ŷ����þ����ƽ�����������󶼱�����

%��ͬ����ά�ȡ����X��M�к�N�У���C��

%L�к�N�У�������M�к�L�С����

%I��Jth entry�Ǵ�X�ĵ�I�е�

%��j��C��

%

%���

%GMMACTIV��KMEANS��RBFwd

%��Ȩ���У�c��Ian T Nabney��1996-2001��

%���������ʱ�ᵼ��n2�г��ָ�ֵ��Ŀ
end