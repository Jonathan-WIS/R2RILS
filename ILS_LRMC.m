function [X_hat U_hat lambda_hat V_hat, observed_RMSE U1_RMSE] = ILS_LRMC(X,mask,rank)
%function [X_hat U_hat V_hat] = ILS_LRMC(X,mask,rank)
%
%
% WRITTEN BY BAUCH & NADLER / 2019
%
% INPUT: 
% X = true matrix assumed to be low rank
% mask = list of size nv * 2 of pairs (i,j) ; nv=total number of visible
% entries
%
%

TOTAL_ITER = 35; % total number of iterations

[nr nc] = size(X);   %nr,nc = number of rows / colums

% X0 is matrix with 0 at non-observed locations and Xij at observed locations
X0 = zeros(nr,nc); 

nv = size(mask,1);  % number of visible entries
for counter=1:nv
    X0(mask(counter,1),mask(counter,2)) = X(mask(counter,1),mask(counter,2)); 
end

Xmax = max(max(abs(X0)));   %max absolute value of all observed entries
norm_0 = sqrt( sum( sum( X0.^2 ) ) / nv ); 

rhs = zeros(nv,1);   %vector of visible entries in matrix X


fprintf('Inside ILS_LRMC nr= %d nc = %d nv= %d\n',nr,nc,nv); 

for counter=1:nv
    rhs(counter) = X0(mask(counter,1),mask(counter,2)); 
end
[U S V] = svds(X0,rank); % U is of size nr x rank; V is of size nc x rank (both column vectors)

[Utrue Strue Vtrue] = svds(X,rank); 

%[Utrue Strue Vtrue] = svds(X,rank);  % in past used for debugging 

U0 = U; V0 = V; 

m = (nc + nr) * rank;  % total number of variables in single update iteration

% Z^T = [a(coordinate 1)  a(coordinate 2) ... a(coordinate nc) | b(coordinate 1) ... b(coordinate nr) ]
% Z^T = [ Vnew and then Unew ] 

observed_RMSE=zeros(TOTAL_ITER,1); 
U1_RMSE = zeros(TOTAL_ITER,1); 

for loop_idx = 1:TOTAL_ITER
    if 0 fprintf('loop_idx  %d/%d\n',loop_idx,TOTAL_ITER); end

    Z = zeros(m,1);  
    A = zeros(nv,m); 

    %tic; 
    for counter=1:nv
        j = mask(counter,1); k = mask(counter,2); 
        index=rank*(k-1)+1; 
        A(counter,index:index+rank-1) = U(j,:); 
        index = rank*nc + rank*(j-1)+1; 
        A(counter,index:index+rank-1) = V(k,:); 
        if 0 fprintf('counter %d j %d k %d U %f V %f\n',counter,j,k,U(j,1),V(k,1)); end
    end
    %fprintf('Construct A\t'); toc; 
    
    A = sparse(A); 
    
    %tic; 
    if 0    % matrix A is rank deficient. Need to either regularize OR find min norm solution
        BB_epsilon = 1e-12; 
        Z = inv( A' * A + BB_epsilon*eye(m))* A'*rhs; 
    else
      Z = lsqminnorm(A,rhs); 
    end
    %fprintf('Solving System\t'); toc; 
    
    % construct U and V from the entries of the long vector Z 
    
    Unew = zeros(size(U)); Vnew = zeros(size(V)); 
    nc_list = rank* [0:1:(nc-1)]; 
    for i=1:rank
        Vnew(:,i) = Z(i+nc_list); 
    end

    nr_list = rank*[0:1:(nr-1)]; 
    start_idx = rank*nc; 
    for i=1:rank
        Unew(:,i) = Z(start_idx + i + nr_list);
    end
     
    
    
    X_hat = U *Vnew' + Unew * V';   % rank 2*r intermediate result

    observed_RMSE(loop_idx) = sqrt(sum(sum( (abs(X0)>0).*(X_hat-X0).^2) )  / nv); 
    
    normU = sqrt(sum(Unew.^2)); 
    Unew = Unew * diag(1./normU);  

    normV = sqrt(sum(Vnew.^2)); 
    Vnew = Vnew * diag(1./normV);  

    
    if 0 
        Uall = [U Unew]; 
        [Ut,St,TEMP] = svds(Uall,rank);
    
        U = Ut; 
    
        Vall = [V Vnew]; 
        [Vt,St,TEMP] = svds(Vall,rank);
    
        V = Vt; 
    end
    
    if 1
        U = 1/2 * (U + Unew);
        V = 1/2 * (V + Vnew); 
        normU = sqrt(sum(U.^2)); 
        U = U * diag(1./normU); 

        normV = sqrt(sum(V.^2)); 
        V = V * diag(1./normV); 
        
        U1_RMSE(loop_idx) = min(norm(U(:,1)-Utrue(:,1)),norm(U(:,1)+Utrue(:,1)));   
    end

    if 0 
        figure(10); clf; plot(Utrue(:,1),'b.-'); grid on; hold on; plot(U(:,1),'rs-');
        plot(Unew(:,1),'ko-'); 
        title([loop_idx ' ' num2str(U1_RMSE(loop_idx))]); drawnow; 
        pause; 
    end
    
    
    %D = ILS_Estimate_Sigma(Unew,Vnew,mask,rhs);
    %X_b = Unew*diag(D)*Vnew'; 
    %disp(max(max(X_b))); 
    
    %if 0 
    if observed_RMSE(loop_idx) > 10*norm_0
        fprintf('ILS-LRMC early exit loop_idx %d max(Xt) %f max(Xv) %f\n',loop_idx,max(max(abs(X_hat))),Xmax);
        fprintf('ILS-LRMC observed RMSE %8f norm_observed %8f\n',observed_RMSE(loop_idx),norm_0); 
        %pause; 
        break; 
    end
end

[U_hat lambda_hat V_hat] = svds(X_hat,rank); 
