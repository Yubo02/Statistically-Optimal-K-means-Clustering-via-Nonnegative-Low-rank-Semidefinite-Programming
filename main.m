
%% Load the data
[data_lev,name] = fca_readfcs('Levine_32dim.fcs');

%% Tidy the data

data_lev1 = data_lev((data_lev(:,41)==1),:);
data_lev2 = data_lev1((data_lev1(:,40)>0),:);

data_lev0 = data_lev2(:,5:36); % data points
label_lev0 = data_lev2(:,40); % label

% Arrange the data based on the labels
clus_ind = 1:14;
clus_ind1 = clus_ind;
data1_lev = data_lev0((label_lev0==clus_ind1(1)),:);
label1_lev = label_lev0((label_lev0==clus_ind1(1)),:);
for i = 2:size(clus_ind1,2)
data1_lev = [data1_lev ; data_lev0((label_lev0==clus_ind1(i)),:)];
label1_lev = [label1_lev ; label_lev0((label_lev0==clus_ind1(i)),:)];
end

for i=1:size(clus_ind1,2)
    ind_l=clus_ind1(i);
    label1_lev(label1_lev==ind_l)=i;
end

% Final data matrix and labels
data1_lev; data1_lev = data1_lev';
label1_lev;

%% Choose K=4 clusters
K = 4;

f1 = 2; f2 = 7; f3 = 8; f4 = 9;
% New data matrix
X_t1 = data1_lev(:,label1_lev==f1);
X_t2 = data1_lev(:,label1_lev==f2);
X_t3 = data1_lev(:,label1_lev==f3);
X_t4 = data1_lev(:,label1_lev==f4);

X0 = [X_t1  X_t2 X_t3 X_t4];
n = size(X0,2); % new sample size


% New labels
label0 = [ones(sum(label1_lev==f1),1); 2*ones(sum(label1_lev==f2),1); 3*ones(sum(label1_lev==f3),1);  4*ones(sum(label1_lev==f4),1)];


%% The main part of comparision


% Sub-sample n=1800 many data points

ind_ran = randsample(n,1800);
X = X0(:,ind_ran);
label_true = label0(ind_ran);


%% SDP via SDPNAL+

X_sdp = kmeans_sdp(X, K);

% Rounding process
[U_sdp,~,~] = svd(X_sdp);
U_top_k = U_sdp(:,1:K);

Label_SDP = kmeansplus(U_top_k',K)';  % label
error_SDP = err_rate(Label_SDP,label_true,K); % mis-clustering error

sdp_err = error_SDP;

%% BM method
 
U_bm = BM_cluster(X,K);

% Rounding process
[U_low,~,~] = svd(U_bm,'econ');
U_top_k = U_low(:,1:K);

Label_bm = kmeansplus(U_top_k',K)';  % label
error_bm = err_rate(Label_bm,label_true,K); % mis-clustering error

bm_err = error_bm;

%% K-means++

Label_km = kmeansplus(X,K)';  

error_km = err_rate(Label_km,label_true,K); % mis-clustering error

km_err = error_km;

%% Spectral Clustering
Label_SC = spectralcluster(X',K);

error_SC = err_rate(Label_SC,label_true,K); % mis-clustering error

sc_err = error_SC;

%% NNMF
U_22 = NNMF_cluster(X,K);

% Rounding process
[U_low,~,~] = svd(U_22,'econ');
U_top_k = U_low(:,1:K);

Label_nnmf = kmeansplus(U_top_k',K)';  % label
error_nnmf = err_rate(Label_nnmf,label_true,K); % mis-clustering error

nnmf_err = error_nnmf;

%% Final comparison
Error = [nnmf_err; km_err; bm_err; sc_err; sdp_err];


