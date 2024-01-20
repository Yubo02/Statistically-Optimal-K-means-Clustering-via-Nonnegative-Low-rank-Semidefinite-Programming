

function U_out = NNMF_cluster(X,K)
%% Initialization

n = size(X,2); % Sample size
nmX = norm(X,'fro'); % Norm of matrix X
r = 2*K; % Rank of the matrix in NNMF
alpha = 1e-6; % Step size
tol = 1e-6; % Tolerance of stopping criteria
maxiter = 50000; % Maximum of iterations

% Projection operator
proj = @(V) max(V,0) ;

%% Gradient of function
grad = @(U) -4*X'*(X*U) + 4*U*(U'*U); 

%% Implement algorithm
U_p = abs(randn(n,r));  U = U_p/norm(U_p,'fro')*nmX; % Random intialization

for iter = 1:maxiter

    G = grad(U);
    Unew = proj(U - alpha*G);
    
    % Evaluate iterate
    rdiff = norm(Unew - U,'fro')/norm(U,'fro');
    
    % Update the variable
    U = Unew;

% Stopping criteria
if rdiff<tol
    break
end


end

% Output matrix
U_out = U;

end
