function error = err_rate(L,label1,K) 

n=size(L,1); % number of samples

l_0 = [];
for j = 1:K
    l_0{j} = label1==j;
end

order = 1:K;
Per = perms(order);
error_1 = zeros(1,size(Per,1));

% Calculate all the permutations of 1:K
for i=1:size(Per,1)

    idx_ori=l_0{1}.*Per(i,1);
    for j=2:K
        idx_loop = l_0{j}.*Per(i,j);
        idx_ori = idx_ori + idx_loop; 
    end

error_1(i)=sum(idx_ori~=L)/n; % error rate comparing to L 

end

error = min(error_1); % The minimum of error rates comparing to L 

end

