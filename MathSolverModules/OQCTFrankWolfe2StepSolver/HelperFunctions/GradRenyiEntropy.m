function grad_renyi = GradRenyiEntropy(alpha,gRho,zRho,krausOp,keyProj)

Sz = size(gRho);
beta = 1/alpha;
Trace = trace((gRho^beta)*zRho^(1-beta));

% mu = (1-beta)/(2*beta);
% Trace = trace(((zRho^mu)*gRho*zRho^mu)^beta);

perturbation = zeros(size(gRho));
for i = 1:Sz(1)
    if mod(i,2) == 0
        perturbation(i,i)=1e-11;
    else
        perturbation(i,i) = -1e-11;
    end
end

%Renyi
Lambda1 = ApplyMap((1-beta)*(gRho^beta)*(zRho^(-beta)), keyProj);
Lambda2 = beta*(gRho^(beta-1))*(zRho^(1-beta));
temp = trace(gRho)*(1/(beta-1))*((Lambda1+Lambda2)/Trace - eye(Sz)/trace(gRho))...
    + (eye(Sz)/trace(gRho))*RenyiEntropy(alpha,gRho,zRho);

grad_renyi = ApplyMap(temp,DualMap(krausOp));

end

