%-------------------------------------
% feature construction for policy functions
% input is a batch of sample
% return of the features of all actions
% the idea of radial basis function is to work on a higher dimensional space
% that is more likely to be seperable than the original space
%-------------------------------------
function phi = all_feature ( sample, center, sigma2, TK ) % action, which_reward, K, TK )
    % size(center) = ( |phi| * |A| * 2 ), K = | phi | );
    X = repmat( horzcat(vertcat(sample.wellness), ...
                        vertcat(sample.tumorsize)), 1, TK );
    C = repmat( reshape(center',1,[]), sample(1).sample_size, 1 );
    XC = X - C;

    %  double_sigma2 = repmat(double_sigma2, sample(1).sample_size ,1); 
    XCD = XC .* XC;
    CXDS = XCD(:, 1:2:end) + XCD(:, 2:2:end);
    inside = CXDS/2; %./ double_sigma2;                    
    phi = exp(-1 * inside);   
    
end