function grad_renyi = finite_diff(gRho,zRho,keyProj, alpha)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
beta = (2-1/alpha)^(-1);
Sz = size(gRho);
eps = 1e-9;
grad_renyi = zeros(Sz);
Trace = trace((gRho^beta)*(zRho^(1-beta)));

for row = 1:Sz(1)
    for col = 1:Sz(2)
        temp = zeros(Sz);
        temp(row,col)=1;
        gRhoprime = gRho + eps*temp;
        zRhoprime = ApplyMap(gRhoprime,keyProj);
        subgrad = elementwise(gRhoprime,gRho,zRhoprime, zRho, eps);
        subtemp = zeros(Sz);
        subtemp(row,col) = 1;
        grad_renyi = grad_renyi + subgrad*subtemp;
    end
end

    function subgrad = elementwise(gRhoprime,gRho,zRhoprime, zRho,eps)
        % subgrad = (trace((gRhoprime^beta)*zRhoprime^(1-beta))-trace((gRho^beta)*zRho^(1-beta)))/eps;
        subgrad = (RenyiEntropy(alpha, gRhoprime, zRhoprime)-RenyiEntropy(alpha, gRho, zRho))/(eps);
    end

grad_renyi = real(grad_renyi);

end