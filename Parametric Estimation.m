%% load train data 
load('TrainingSamplesDCT_8_new.mat');
size(TrainsampleDCT_FG)  % 250    64
size(TrainsampleDCT_BG)  % 1053          64
n_FG = size(TrainsampleDCT_FG, 1);
n_BG = size(TrainsampleDCT_BG, 1); 

%% Task a). Fit a multinomial dtb for the prior. Since there are only 
%%   categories, grass and foreground, so the multionomial dtb actually
%%   degrades to a binomial one. 

% plot the histogram for the prior 
freq_fg = ones(250, 1);
freq_bg = zeros(1053, 1);
freq_y = categorical(cat(1, freq_fg, freq_bg));
% prior_hist = histogram(freq_y); 
% xloc = 1:numel(prior_hist.Categories);
% yloc = prior_hist.Values;
% text(xloc(1),yloc(1),sprintfc('%s', 'foreground'),'vert','bottom','horiz','center');
% text(xloc(2),yloc(2),sprintfc('%s', 'background'),'vert','bottom','horiz','center');
% saveas(gcf, 'prior_hist.jpg'); 

% estimated parameter by MLE 
X = cat(1, freq_fg, freq_bg);
n = size(X, 1);  
bin_p = sum(X, 'all') / n;
bin_var = bin_p*(1-bin_p) / n;

prior_prob = zeros(2);
prior_prob(1) = bin_p;
prior_prob(2) = 1-bin_p;


%% Task b). Part I. Use MLE to find estimated parameters for 
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
%% Task b). Part II. Find the marginal distribution for each feature. 
%%  and plot the marginal distribution.
% % plot the marginal dtb of feature 1, conditioned on FG (total 64 features)
% feature_idx = 21;
% u = sample_mean_FG(feature_idx); 
% var = sample_cov_FG(feature_idx, feature_idx)
% x = linspace(u-5*var, u+5*var, 100)
% y = normpdf(x, u, var);
% plot(x, y);

% % subplot marginal dtb of features conditioned   
% for feature_idx = 1:16
%     subplot(4, 4, feature_idx);
%     u_fg = sample_mean_FG(feature_idx); 
%     var_fg = sample_cov_FG(feature_idx, feature_idx); 
%     x = linspace(u_fg-5*var_fg, u_fg+5*var_fg,100);
%     y_fg = normpdf(x, u_fg, var_fg); 
%     plot(x, y_fg);
%     hold on 
% 
%     u_bg = sample_mean_BG(feature_idx); 
%     var_bg = sample_cov_BG(feature_idx, feature_idx); 
%     y_bg = normpdf(x, u_bg, var_bg); 
%     plot(x, y_bg);
%     hold off
%     title(feature_idx);
% end  
% saveas(gcf, 'FG_marginal_1.jpg');  
% 
% for feature_idx = 17:32
%     subplot(4, 4, feature_idx-16);
%     u_fg = sample_mean_FG(feature_idx); 
%     var_fg = sample_cov_FG(feature_idx, feature_idx); 
%     x = linspace(u_fg-5*var_fg, u_fg+5*var_fg,100);
%     y_fg = normpdf(x, u_fg, var_fg); 
%     plot(x, y_fg);
%     hold on 
% 
%     u_bg = sample_mean_BG(feature_idx); 
%     var_bg = sample_cov_BG(feature_idx, feature_idx); 
%     y_bg = normpdf(x, u_bg, var_bg); 
%     plot(x, y_bg);
%     hold off
%     title(feature_idx);
% end  
% saveas(gcf, 'FG_marginal_2.jpg');  
% 
% for feature_idx = 33:48
%     subplot(4, 4, feature_idx-32);
%     u_fg = sample_mean_FG(feature_idx); 
%     var_fg = sample_cov_FG(feature_idx, feature_idx); 
%     x = linspace(u_fg-5*var_fg, u_fg+5*var_fg,100);
%     y_fg = normpdf(x, u_fg, var_fg); 
%     plot(x, y_fg);
%     hold on 
% 
%     u_bg = sample_mean_BG(feature_idx); 
%     var_bg = sample_cov_BG(feature_idx, feature_idx); 
%     y_bg = normpdf(x, u_bg, var_bg); 
%     plot(x, y_bg);
%     hold off
%     title(feature_idx);
% end  
% saveas(gcf, 'FG_marginal_3.jpg');  
% 
% for feature_idx = 49:64
%     subplot(4, 4, feature_idx-48);
%     u_fg = sample_mean_FG(feature_idx); 
%     var_fg = sample_cov_FG(feature_idx, feature_idx); 
%     x = linspace(u_fg-5*var_fg, u_fg+5*var_fg,100);
%     y_fg = normpdf(x, u_fg, var_fg); 
%     plot(x, y_fg);
%     hold on 
% 
%     u_bg = sample_mean_BG(feature_idx); 
%     var_bg = sample_cov_BG(feature_idx, feature_idx); 
%     y_bg = normpdf(x, u_bg, var_bg); 
%     plot(x, y_bg);
%     hold off
%     title(feature_idx);
% end  
% saveas(gcf, 'FG_marginal_4.jpg');  

% automatically select best 
class_diff = zeros(64, 1);
for feature_idx = 1:64 
    u_fg = sample_mean_FG(feature_idx); 
    var_fg = sample_cov_FG(feature_idx, feature_idx); 
    x = linspace(u_fg-5*var_fg, u_fg+5*var_fg,100);

    u_bg = sample_mean_BG(feature_idx); 
    var_bg = sample_cov_BG(feature_idx, feature_idx); 
    y_bg = normpdf(x, u_bg, var_bg); 
    
    diff = abs(u_fg - u_bg) / (var_fg + var_bg);
    class_diff(feature_idx) = diff;
end
transpose(class_diff); 
% compare the class_diff results with the feature marginal plots,
%   this simple class_diff could indeed reflects the difference between
%   the two classe somehow. 

% choose best 8 features and worst 8 features by class_diff ranking
[class_diff_sorted,r]=sort(class_diff,'descend'); 
transpose(class_diff_sorted); % the sorted data
transpose(r); % the corresponding indices
% best_feature_idxes = r(19:26); failures 
best_feature_idxes = r(13:20); % good results
worst_feature_idxes = r(57:64); 
transpose(best_feature_idxes)

% the extracted pdf:  
sample_mean_FG_extracted = sample_mean_FG(best_feature_idxes); 
sample_mean_FG_extracted;
sample_cov_FG_extracted = sample_cov_FG(best_feature_idxes, best_feature_idxes);

sample_mean_BG_extracted = sample_mean_BG(best_feature_idxes); 
sample_mean_BG_extracted;
sample_cov_BG_extracted = sample_cov_BG(best_feature_idxes, best_feature_idxes);


% % plot the marginal pdf for best 8 features 
% for i = 1:8 
%     feature_idx = best_feature_idxes(i);
%     subplot(3, 3, i);
%     u_fg = sample_mean_FG(feature_idx); 
%     var_fg = sample_cov_FG(feature_idx, feature_idx); 
%     x = linspace(u_fg-5*var_fg, u_fg+5*var_fg,100);
%     y_fg = normpdf(x, u_fg, var_fg); 
%     plot(x, y_fg);
%     hold on 
% 
%     u_bg = sample_mean_BG(feature_idx); 
%     var_bg = sample_cov_BG(feature_idx, feature_idx); 
%     y_bg = normpdf(x, u_bg, var_bg); 
%     plot(x, y_bg);
%     hold off
%     title(feature_idx);
% end   
% saveas(gcf, 'FG_marginal_best8features.jpg');

% % plot the marginal pdf for worst 8 features 
for i = 1:8 
    feature_idx = worst_feature_idxes(i);
    subplot(3, 3, i);
    u_fg = sample_mean_FG(feature_idx); 
    var_fg = sample_cov_FG(feature_idx, feature_idx); 
    x = linspace(u_fg-5*var_fg, u_fg+5*var_fg,100);
    y_fg = normpdf(x, u_fg, var_fg); 
    plot(x, y_fg);
    hold on 

    u_bg = sample_mean_BG(feature_idx); 
    var_bg = sample_cov_BG(feature_idx, feature_idx); 
    y_bg = normpdf(x, u_bg, var_bg); 
    plot(x, y_bg);
    hold off
    title(feature_idx);
end   
% saveas(gcf, 'FG_marginal_worst8features.jpg');

%% Task(c) Part I: solve the cheetah problem using 64 features 
% load data 
mask = imread('cheetah_mask.bmp');
size(mask); % check size: 255   270 
cheetah = imread('cheetah.bmp'); 
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

%         % extract feature for a 64D data point
%         unbiased_fg = dct_output_vec - sample_mean_FG;
%         posterior_fg = -0.5*unbiased_fg*inv(sample_cov_FG)*transpose(unbiased_fg)...
%             -0.5*log((2*pi)^64*det(sample_cov_FG) + log(prior_prob(1)));
%         posterior_fg
% 
%         unbiased_bg = dct_output_vec - sample_mean_BG;
%         posterior_bg = -0.5*unbiased_bg*inv(sample_cov_BG)*transpose(unbiased_bg)...
%             -0.5*log((2*pi)^64*det(sample_cov_BG) + log(prior_prob(2)));
%         posterior_bg

        % extract feature for a 8D data point  
        d = size(best_feature_idxes, 1);
        dct_output_vec_extracted = dct_output_vec(best_feature_idxes);
        unbiased_fg = dct_output_vec_extracted - sample_mean_FG_extracted;
        posterior_fg = -0.5*unbiased_fg*inv(sample_cov_FG_extracted)*transpose(unbiased_fg)...
            -0.5*log((2*pi)^d*det(sample_cov_FG_extracted) + log(prior_prob(1))); 
        %posterior_fg;

        unbiased_bg = dct_output_vec_extracted - sample_mean_BG_extracted;
        posterior_bg = -0.5*unbiased_bg*inv(sample_cov_BG_extracted)*transpose(unbiased_bg)...
            -0.5*log((2*pi)^d*det(sample_cov_BG_extracted) + log(prior_prob(2))); 
        %posterior_fg - posterior_bg

        if posterior_fg > posterior_bg
            test_results(i, j) = 255;
        else
            test_results(i, j) = 0; 
        end
    end
end
  
% plot the test result 
colormap(gray(255));
imagesc(test_results);  

% estimate the probability of errors 
% error = sum(test_results ~= mask, "all") / (255*270);  
% 0.4491 for 64 features; 
% 0.1031 for best 8 features 





%% Task(c) Part II: solve the cheetah problem using extracted 8 features 

