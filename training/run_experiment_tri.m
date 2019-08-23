function run_experiment_tri(imdb_video)
%RUN_EXPERIMENT
%	contains the parameters that are specific to the current experiment.
% 	These are the parameters that change frequently and should not be committed to
% 	the repository but should be saved with the results of the experiment.


	startup;
	% Parameters that should have no effect on the result.
	opts.prefetch = false;
	opts.train.gpus = 1;

	% Parameters that should be recorded.	
    opts.loss.labelWeight = 'balanced';%'gaussian';%
    opts.numPairs =  5.32e4;
    rng(1)
	if nargin < 1
	    imdb_video = [];
    end
    opts.train.numEpochs = 10;
    opts.expDir = 'triplet_gray';
    opts.train.learningRate = logspace(-4, -5, opts.train.numEpochs);
    
    opts.src_model =  '../models/2016-08-17_gray025.net.mat';
    opts.augment.grayscale = 0.25;
    experiment_triplet(imdb_video, opts);



end

