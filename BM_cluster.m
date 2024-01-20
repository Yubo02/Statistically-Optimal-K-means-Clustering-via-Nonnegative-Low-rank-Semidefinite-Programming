% KMEANS Cluster multivariate data using the BM method.
% Authors:      Yubo Zhuang (yubo2@illinois.edu)
% Filename:     BM_cluster.m
% Version:      2024-01-20
% Description:  This function uses BM formulation to solve the K-means 
%               clustering problem [1].
% Inputs:       
%               -X: 
% 
%               A d x n array of data points where d denotes the
%               dimension of the data and n denotes the number of points to 
%               cluster. Each column of X corresponds with a data point.
% 
%               -K: 
% 
%               The number of clusters.
%
% Outputs:
%               -U:
%               
%               A n x r matrix corresponding to the solution of 
%               kmeans BM solution [1]. 
% 
% References:
% 
% [1] Y. Zhuang, X. Chen, Y. Yang, R. Y. Zhang. Statistically Optimal 
% K-means Clustering via Nonnegative Low-rank Semidefinite Programming. 
% -------------------------------------------------------------------------
function U_out = BM_cluster(X,K)


n = size(X,2); % Sample size
r = 2*K; % Rank of the matrix in BM
beta = 10; % Coeffient of the augmented term
alpha = 1e-6; % Step size
tol = 1e-6; % Tolerance of stopping criteria
maxiter = 50000; % Maximum of iterations

%% Projection operator
proj = @(V) max(V,0)/norm(max(V,0),'fro') * sqrt(K) ;

%% Gradient of Augmented Lagrangian
one = ones(n,1);

ynew = @(U,y) y + beta*(U*(U'*one) - one);

gradAugL = @(U,y) -2*X'*(X*U) + ynew(U,y)*(one'*U) + one*(ynew(U,y)'*U);


%% Implement algorithm

y = zeros(n,1); % Intialization for dual variables
U_0 = abs(randn(n,r)); U = sqrt(K)/norm(U_0,'fro')*U_0; % Random intialization


for iter = 1:maxiter

    % Primal update
    G = gradAugL(U,y);
    Unew = proj(U - alpha*G);
    
    % Evaluate iterate
    infeas = Unew*(Unew'*one) - one;

    % Dual update
    rdiff = norm(Unew - U,'fro')/sqrt(K);
    if rdiff < 1e-3
        U = Unew;
        y = y + beta*infeas;
    else
        U = Unew;
    end


    % Stopping criteria    
    if max(rdiff, norm(infeas)/norm(one))<tol
       break
    end


end

% Output matrix

U_out = U;

end


