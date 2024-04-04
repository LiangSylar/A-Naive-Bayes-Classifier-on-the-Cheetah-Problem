% using MAP 

%% a1. compute covariance mtr for each class-conditional by sample cov
% load data 
load('TrainingSamplesDCT_subsets_8.mat');
load('Alpha.mat');
strategy1 = load("Prior_1.mat");  % use strategy 1 for prior parameter dtb

% work on D1 training data 
n_FG = size(D1_FG, 1);
n_BG = size(D1_BG, 1);  
sample_mean_FG = transpose(mean(D1_FG, 1)); % the estimated mean 
sample_mean_BG = transpose(mean(D1_BG, 1)); 

sample_cov_FG = zeros(64); % the estimated cov matrix 
for i = 1:n_FG
    x = transpose(D1_FG(i,:)); 
    unbiased_x = x - sample_mean_FG;  
    sample_cov_FG = sample_cov_FG + (unbiased_x) * transpose(unbiased_x);  
end  
sample_cov_FG = sample_cov_FG / (n_FG) ; 

sample_cov_BG = zeros(64);
for i = 1:n_BG
    x = transpose(D1_BG(i,:));
    unbiased_x = x - sample_mean_BG;
    sample_cov_BG = sample_cov_BG + (unbiased_x) * transpose(unbiased_x); 
end
sample_cov_BG = sample_cov_BG / (n_BG) ; 
 
%% a2. posterior parameter distribution 
error_lst = zeros(9, 1);
for idx = 1:9
    alpha_choice = alpha(idx);
    prior_mean_FG = transpose(strategy1.mu0_FG);
    prior_mean_BG = transpose(strategy1.mu0_BG);
    prior_cov= diag(alpha_choice*strategy1.W0); % same cov matrix for both prior mean dtb 
    
    % Q: how to compute posterior, given a multivariate normal and a normal prior  
    % https://stats.stackexchange.com/questions/28744/multivariate-normal-posterior
    posterior_mean_FG = prior_cov/(prior_cov + sample_cov_FG/n_FG)*sample_mean_FG + (sample_cov_FG/n_FG)/(prior_cov + sample_cov_FG/n_FG)*prior_mean_FG; 
    posterior_mean_BG = prior_cov/(prior_cov + sample_cov_BG/n_BG)*sample_mean_BG + (sample_cov_BG/n_BG)/(prior_cov + sample_cov_BG/n_BG)*prior_mean_BG;  
    size(posterior_mean_FG); % 64 1    
    
    posterior_cov_FG = prior_cov/(prior_cov + sample_cov_FG/n_FG)*(sample_cov_FG/n_FG);
    posterior_cov_BG = prior_cov/(prior_cov + sample_cov_BG/n_BG)*(sample_cov_BG/n_BG);    
    %% a3. parameters of the predictive distribution 
    % We use MAP estimation here. Since posterior u dtb is Gaussian, 
    %   then the posterior mean of u maximizes the posteiror dtb. 
    z_u_FG = posterior_mean_FG;
    z_u_BG = posterior_mean_BG;
    z_cov_FG = sample_cov_FG;  
    z_cov_BG = sample_cov_BG;  
    
    %% a5. Maximum likelihood extimate of prior distribution (same as hw2)
    prior_prob = ones(2, 1);
    prior_prob(1) = n_FG/(n_FG+n_BG);
    prior_prob(2) = n_BG/(n_FG+n_BG);

    %% a. Apply the Bayesian model on the test data 
    % load data 
    mask = imread('cheetah_mask.bmp');
    size(mask); % check size: 255   270 
    cheetah = imread('cheetah.bmp'); 
    cheetah = double(cheetah)/255;
    flatten_pattern = load('Zig-Zag Pattern.txt');
    
    % containers for prediction on each pixel
    test_results = zeros(size(cheetah)); 
    
    % interate through each pixel of the test image
    for i = 1:255
        for j = 1:270 
            % cut an 8*8 block from the test img, with pixel(i,j) placed in mid. 
            % cheetah(255, 270) 
            down = i + 3;
            up = i - 4;
            left = j - 4;
            right = j + 3;
            down = (down <= 255)*down + (down > 255)*255;
            up = (up >= 1)*up + (up < 1)*1;
            left = (left >= 1)*left + (left < 1)*1;
            right = (right <= 270)*right + (right > 270)*270; 
            dct_input = cheetah(up:down, left:right);   
            dct_output = dct2(dct_input, [8, 8]); 
    
            % flatten the 8*8 matrix into a 64D vector as dct_output_vec
            dct_output_vec = 1:64; % initialization 
            for a = 1:8
                for b = 1:8
                    pos = flatten_pattern(a, b) + 1;
                    dct_output_vec(pos) = dct_output(a,b);
                end
            end 
    
            % compute the posterior probability 
            xx = transpose(dct_output_vec); 
            predictive_val_fg = mvnpdf(xx, z_u_FG, z_cov_FG);
            posterior_fg = predictive_val_fg*prior_prob(1);
    
            predictive_val_bg = mvnpdf(xx, z_u_BG, z_cov_BG);
            posterior_bg = predictive_val_bg*prior_prob(2); 
            % fprintf("%d %d %d %d\n",predictive_val_bg, predictive_val_fg, posterior_bg, posterior_fg);
            if posterior_fg > posterior_bg
                test_results(i, j) = 255;
            else
                test_results(i, j) = 0; 
            end
        end
    end
    
%     % plot the test result 
%     colormap(gray(255));
%     imagesc(test_results);  
   % compute error rate
   error = sum(test_results ~= mask, "all") / (255*270); 
   error_lst(idx) = error;
end

alpha_x = (-4:4);
plot(alpha_x, error_lst)