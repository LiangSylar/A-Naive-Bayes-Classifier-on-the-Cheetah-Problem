% MLE method for cheetagh problem 
load('TrainingSamplesDCT_subsets_8.mat');
% choosing dataset 
TrainsampleDCT_FG = D4_FG; 
TrainsampleDCT_BG = D4_BG;

%% b1) Fit a multinomial dtb for the prior. Since there are only 
%%   categories, grass and foreground, so the multionomial dtb actually
%%   degrades to a binomial one. 

% plot the histogram for the prior 
n_FG = size(TrainsampleDCT_FG, 1);
n_BG = size(TrainsampleDCT_BG, 1); 
freq_fg = ones(250, 1);
freq_bg = zeros(1053, 1);
freq_y = categorical(cat(1, freq_fg, freq_bg)); 

% estimated parameter by MLE 
X = cat(1, freq_fg, freq_bg);
n = size(X, 1);  
bin_p = sum(X, 'all') / n;
bin_var = bin_p*(1-bin_p) / n;

prior_prob = zeros(2);
prior_prob(1) = bin_p;
prior_prob(2) = 1-bin_p;


%% b2) Use MLE to find estimated parameters for 
%%   Multivariate gaussian distribution. c
sample_mean_FG = mean(TrainsampleDCT_FG, 1); % the estimated mean
size(sample_mean_FG) % 1 64  
sample_mean_BG = mean(TrainsampleDCT_BG, 1); 

% compute estimated cov matrix
sample_cov_FG = zeros(64);
for i = 1:n_FG
    x = TrainsampleDCT_FG(i,:); 
    unbiased_x = x - sample_mean_FG;  
    sample_cov_FG = sample_cov_FG + transpose(unbiased_x) * (unbiased_x);  
end  
sample_cov_FG = sample_cov_FG / (n_FG) ;
size(sample_cov_FG); % 64    64  
sample_cov_BG = zeros(64);
for i = 1:n_BG
    x = TrainsampleDCT_BG(i,:);
    unbiased_x = x - sample_mean_BG;
    sample_cov_BG = sample_cov_BG + transpose(unbiased_x) * (unbiased_x); 
end
sample_cov_BG = sample_cov_BG / (n_BG) ; 
 
%% b3). solve the cheetah problem using 64 features 
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
        
        likelihood_fg = mvnpdf(dct_output_vec, sample_mean_FG, sample_cov_FG);
        posterior_fg = likelihood_fg*prior_prob(1);
    
        likelihood_bg = mvnpdf(dct_output_vec, sample_mean_BG, sample_cov_BG);
        posterior_bg = likelihood_bg*prior_prob(2);

        if posterior_fg > posterior_bg
            test_results(i, j) = 255;
        else
            test_results(i, j) = 0; 
        end
    end
end
  
% % plot the test result 
% colormap(gray(255));
% imagesc(test_results);  

% estimate the probability of errors 
error_mle = sum(test_results ~= mask, "all") / (255*270);
%  0.1450 for 64 features 





%% Task(c) Part II: solve the cheetah problem using extracted 8 features 

