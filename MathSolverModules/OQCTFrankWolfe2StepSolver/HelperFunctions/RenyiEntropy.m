function renyi_ent = RenyiEntropy(alpha,gRho,zRho)

beta = (2-1/alpha)^(-1);
mu = (1-beta)/(2*beta);
Trace = trace(((zRho^mu)*gRho*zRho^mu)^beta);

renyi_ent = trace(gRho)*(1/(beta-1))*(log(Trace/trace(gRho)));
% renyi_ent = trace(gRho*logm(gRho)-gRho*logm(zRho));
end

