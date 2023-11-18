clear all;clc;

addpath('.\utils');
addpath('.\tools\libsvm-3.17\matlab');

% parameter
param.dim = 200;
param.C = 1;
tol_eps = 0.000001;

fprintf('loading data....\n');
train_data = load('.\data\dslr');
test_data = load('.\data\amazon');

%XA = train_data.fts';
XA = train_data.fts;
XA = XA - repmat(mean(XA,1),size(XA,1),1);

%XB = test_data.test_fts';
XB = test_data.fts;
XB = XB - repmat(mean(XB,1),size(XB,1),1);

testA = length(XA)
testB = length(XB)
fprintf('performing GFK....\n');
G = train_gfk(XA, XB, tol_eps, param.dim);
K = XA * G * XA';
test_kernel = XB * G * XA';

train_kernel    = [(1:size(K, 1))' K];

para   = sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.C,0,4,1);
model  = svmtrain(train_data.labels, train_kernel, para)

ay      = full(model.sv_coef)*model.Label(1);

idx     = full(model.SVs);
b       = -(model.rho*model.Label(1));
b       = kron(b, ones(25,9))
b = b(1:958,:)
test = size(b)
test = size(test_kernel(:, idx)*ay)
decs    = test_kernel(:, idx)*ay + b

test_labels = [test_data.labels test_data.labels test_data.labels test_data.labels test_data.labels test_data.labels test_data.labels test_data.labels test_data.labels]

test = size(decs)
test = size(test_labels )

ap  = calc_ap(test_labels , decs)





