function opts = env_paths_training(opts)

    opts.rootDataDir = '/media/dxp/DATA/datasets/ILSVRC2015crop/Data/VID/train/';
    opts.imdbVideoPath = '/media/dxp/DATA/datasets/imdb_video.mat';
    opts.imageStatsPath = '/media/dxp/DATA/datasets/ILSVRC2015.stats.mat';
    opts.src_model =  '../models/2016-08-17_gray025.net.mat';%*mf*
end
