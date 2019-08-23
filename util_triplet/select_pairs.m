classdef select_pairs< dagnn.ElementWise
%   In the first form of label_type, C has dimension H x W x 2 x N and specifies a
%   2 categorical label for each spatial location.   C(:,:,1,:) == ones(H,W,N). C(:,:,2,:)== zeros(H,W,N); 
%   In the second form of label_type, C has dimension H x W x 1 x N and specifies a
%   categorical label for each spatial location.

  properties
    label_type = 'form2'
  end

  methods
      % --------------------------------------------------------------------
      function y = zerosLike(obj,x)
          % --------------------------------------------------------------------
          if isa(x,'gpuArray')
              y = gpuArray.zeros(size(x),classUnderlying(x)) ;
          else
              y = zeros(size(x),'like',x) ;
          end
      end
%       
      function y = onesLike(obj, x)
          % --------------------------------------------------------------------
          if isa(x,'gpuArray')
              y = gpuArray.ones(size(x),classUnderlying(x)) ;
          else
              y = ones(size(x),'like',x) ;
          end
      end
      
    function outputs = forward(obj, inputs, params)
        score = inputs{1};
        label = inputs{2};
        ind_pos = label>0;
        ind_neg = label<0;
        ind_pos = ind_pos(:);
        ind_neg = ind_neg(:);
        n_pos = sum(ind_pos);
        n_neg = sum(ind_neg);
        ind_select = ones(n_pos,n_neg)>0;
        pos_use = sum(ind_select,2);
        neg_use = sum(ind_select,1);
%         instance_weight = 1./(pos_use*neg_use+eps);
        instance_weight = ones(n_pos,n_neg);
        n_pairs = sum(ind_select(:));
        sz_in = size(score);
        try
            batch_size = sz_in(4);
        catch
            batch_size = 1;
        end
%         out = 0;
        if isa(score,'gpuArray')
              in0 = gpuArray.zeros([n_pairs,2,batch_size],classUnderlying(score)) ;
              instance_weight =gpuArray(instance_weight );
          else
              in0 = zeros([n_pairs,2,batch_size],'single');
        end  
        
        
        for i = 1:batch_size
            tmp_s = score(:,:,:,i);            
            pos = tmp_s(ind_pos);
            neg = tmp_s(ind_neg);
%             ef1 = exp(pos); ef2 = exp(neg);
            pos_m = repmat(pos,1,n_neg).*instance_weight;
            neg_m = repmat(neg',n_pos,1).*instance_weight;
            pos_v = pos_m(ind_select);
            neg_v = neg_m(ind_select);
            if size(pos_v,1)==1
                pos_v = pos_v'; neg_v = neg_v';
            end
            in0(:,:,i) = [pos_v,neg_v];            
        end   
        in0 = reshape(in0,[1,n_pairs,2,batch_size]);
%         switch obj.label_type
%             case 'form1'
%         label0 = obj.zerosLike(in0);
%         label0(:,:,1,:) = 1;
%             case 'form2'
%                 label0 = ones(1,n_pairs,1,batch_size,'single');
%             otherwise
%                 
%         end
        
        outputs{1} = in0;
%         outputs{2} = label0;
%         sz_in = size(inputs{1});
%         n = prod(sz_in(1:3));
%         ef1 = exp(inputs{1});
%         ef2 = exp(inputs{2});
%         out = 2*(ef2./(ef1+ef2+eps)).^2 ;
%       outputs{1} = sum(out(:))/n;
%       n = obj.numAveraged ;
%       m = n + size(inputs{1},4) ;
%       obj.average = (n * obj.average + gather(outputs{1})) / m ;
%       obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        score = inputs{1};
        label = inputs{2};
        ind_pos = label>0;
        ind_neg = label<0;
        ind_pos = ind_pos(:);
        ind_neg = ind_neg(:);
        n_pos = sum(ind_pos);
        n_neg = sum(ind_neg);
        ind_select0 = ones(n_pos,n_neg,'single');
        ind_select = ind_select0>0;
        pos_use = sum(ind_select,2);
        neg_use = sum(ind_select,1);
%         instance_weight = 1./(pos_use*neg_use+eps);
        instance_weight = ones(n_pos,n_neg);
        if isa(score,'gpuArray')
            ind_select0 = gpuArray(ind_select0);
            instance_weight =gpuArray(instance_weight );
        end
%         n_pairs = sum(ind_select(:));
        sz_in = size(score);
        try
            batch_size = sz_in(4);
        catch
            batch_size = 1;
        end
        der1 = squeeze(derOutputs{1});
        
%         out = 0;
        derIn0 = obj.zerosLike(score);
%         derOutputs{1} = derOutputs{1}/n_pairs;
        for i = 1:batch_size
            tmp = obj.zerosLike(ind_select0);
            tmp(ind_select) = der1(:,1,i);
            tmp = tmp .*instance_weight;
            tmp_s = score(:,:,:,i);
            der_tmp = obj.zerosLike(tmp_s);
            der_tmp(ind_pos) = sum(tmp,2);
            
            tmp = obj.zerosLike(ind_select0);
            tmp(ind_select) = der1(:,2,i);
            tmp = tmp .*instance_weight;
%             der_tmp = obj.zerosLike(tmp_s);
            der_tmp(ind_neg) = sum(tmp,1);
            derIn0(:,:,1,i) = reshape(der_tmp,size(tmp_s));
            
%             pos = tmp_s(ind_pos);
%             neg = tmp_s(ind_neg);
%             ef1 = exp(pos); ef2 = exp(neg);
%             ef1_m = repmat(ef1,1,n_neg).*ind_select;
%             ef2_m = repmat(ef2',n_pos,1).*ind_select;
%             derIn1 =  -4.*ef2_m.^2./((ef1_m+ef2_m).^3+eps).*derOutputs{1};
%             der_tmp(ind_pos) = sum(derIn1,2);
%             derIn2 = derIn1 *(-1).* ef1_m;
%             der_tmp(ind_neg) = sum(derIn2);
%             derIn0(:,:,1,i) = reshape(der_tmp,size(tmp_s));
        end   
        derInputs{1} = derIn0;
        derInputs{2} = [];
%         sz_in = size(inputs{1});
%         n = prod(sz_in(1:3));
%         derOutputs{1} = derOutputs{1}/n;
%         ef1 = exp(inputs{1});
%         ef2 = exp(inputs{2});  
%       derInputs{1} = -4.*ef2.^2./((ef1+ef2).^3+eps).*derOutputs{1};
%       derInputs{2} = derInputs{1}*(-1).* ef1;
      derParams = {} ;
    end

%     function reset(obj)
%       obj.average = 0 ;
%       obj.numAveraged = 0 ;
%     end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = select_pairs(varargin)
      obj.load(varargin) ;
    end
  end
end
