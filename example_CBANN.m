% System configuration:
% Install Matlab


% Training data
k=5;
x=linspace(0,k*2*pi,500)';
y=(sin(x)+1)/2;
% Query point
xq=2;
% Algorithmic settings
% (1) Related to training data:
% Random perturbation tolerance
pTol=0;
% Aspect ratio
ar=1000;
% Scaling factor
scF=ar*(max(x(:))-min(x(:)))/(max(y)-min(y));
% Validation ratio
tr=0.1;
% (2) Related to CBANN:
% Number of additional clustering features
nCl = [2,4,6,8];
% Hidden layer size
H=2^8;
% Maximum number of epochs
max_Epochs=2000;

% Step 1.1: Random perturbation of features
nVar=size(x,2);
C=x.*(1+pTol*rand(size(x)));
C0=C;
C1=[C,scF*(y-min(y))];

% Step 1.2: Cluster Boosting (CB)
cInd=cell(numel(nCl),1);
for i=1:numel(nCl)
    num_clusters=nCl(i);
    opts = statset('Display','iter');
    cInd{i} = kmedoids(C1, num_clusters,...
        'Distance', 'euclidean',...
        'Algorithm','large',...
        'Options',opts);
    C=[C,cInd{i}];
end
C=[C,scF*(y-min(y))];

% Step 2.1: Divide dataset into training and validation
n=size(C,1);
nV=round(tr*n);
indV=randperm(n,nV);
indT=setdiff((1:n),indV);
dataT=C(indT,:);
dataV=C(indV,:);
% Extract features and target variable
X_T = dataT(:,1:end-1); % training features
y_T = dataT(:,end); % training labels
X_V = dataV(:,1:end-1); % validation features
y_V = dataV(:,end); % validation labels

% Step 2.2: Train the neural network
layers = [...
    ... % Input layer
    featureInputLayer(size(X_T, 2),...
    'Name', 'input',...
    'Normalization','zscore')
    ... % Hidden layer with tanh activation
    fullyConnectedLayer(H, 'Name', 'fc1')
    sigmoidLayer('Name', 'sig1')
    ... % Output layer with linear activation
    fullyConnectedLayer(1, 'Name', 'fc2')
    regressionLayer('Name', 'regress1')
    ];
options = trainingOptions('adam' ...
    ,'MaxEpochs', max_Epochs ...
    ,'MiniBatchSize', 128 ...
    ,'Verbose', true ...
    ,'InitialLearnRate', 0.02 ...
    ,'Shuffle', 'every-epoch' ...
    ,'ValidationData', {X_V, y_V} ...
    ,'ResetInputNormalization',0);
% Train the ANN
[net,info] = trainNetwork(X_T, y_T, layers, options);
disp('Training complete')

% Step 3: Input vector needed for the ANN for prediction at xq
ind_d=zeros(1,numel(nCl));
for i=1:numel(nCl)
    num_clusters=nCl(i);
    min_d=zeros(num_clusters,1);
    for j=1:num_clusters
        cp=C0(cInd{i}==j,:);
        d=sum((cp-repmat(xq,size(cp,1),1)).^2,2);
        min_d(j)=min(d);
    end
    [~,ind_d(i)]=min(min_d);
end
% Formatting of input feature for prediction
pFeat=[xq,ind_d];

% Step 4: Make prediction on xq
pVal = min(y)+1/scF*predict(net, pFeat);
% Display output
disp(['Input value: ', num2str(pFeat)])
disp(['Prediction: ', num2str(pVal)])
disp(['True value: ', num2str((sin(xq)+1)/2)])
