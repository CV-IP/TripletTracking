function net = make_siam_adapt2(opts)

% src_model = '../models/2016-08-17.net.mat';
src_model = opts.src_model;
net = load(src_model);
net = net.net;
net = dagnn.DagNN.loadobj(net);

net.removeLayer('objective');

net.removeLayer('errdisp');

net.removeLayer('errmax');

% fixed learning rate
% for i = length(net.params)-7
%     net.params(i).learningRate = 0;
% end

end