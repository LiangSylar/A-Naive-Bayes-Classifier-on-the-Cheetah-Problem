% step 1. load data 
% by setting the working directory, already include the train data!
load('TrainingSamplesDCT_8.mat') 

% step 2. estimate prior probability 
size(TrainsampleDCT_FG); % 250    64
size(TrainsampleDCT_BG); % 1053   64
prior_prob = [0, 0];
% 250 / (250 + 1053) % 0.1919
% 1053 / (250 + 1053) % 0.8081
prior_prob(1) = 0.1919;  % fg prior
prior_prob(2) = 0.8081;  % bg prior 
% prior_prob 

% step 3. feature extracting x_train; 
% goal: get train data {X_BG, X_FG}; both are vectors. 
X_FG = 1:250;
for i = 1:250
    x = abs(TrainsampleDCT_FG(i,:));
    [m, p] = max(x);
    x(p) = min(x); % turn the max into min
    [m, p] = max(x); % now we get the pos for second largest num as p
    X_FG(i) = p;
end

X_BG = 1:1053;
for i = 1:1053
    x = abs(TrainsampleDCT_BG(i,:));
    [m, p] = max(x);
    x(p) = min(x); % turn the max into min
    [m, p] = max(x); % now we get the pos for second largest num as p
    X_BG(i) = p;
end 
 
% step 4. compute index histgram for X_BG, X_FG 
% nBins = 64;
h_fg = histogram(X_FG);
nBins_fg = h_fg.NumBins;
likelihood_fg = h_fg.Values / 250 ;
fg_binEdges = h_fg.BinEdges; 
saveas(gcf, 'fg_likelihood.jpg'); 

h_bg = histogram(X_BG);
likelihood_bg = h_bg.Values / 1053 ;
bg_binEdges = h_bg.BinEdges;
nBins_bg = h_bg.NumBins; 
saveas(gcf, 'bg_likelihood.jpg'); 

%% No need to compute marginal prob of x. 
% marginal_prob_x = 1:nBins;
% for i = 1:nBins
%     marginal_prob_x = likelihood_fg(i)*prior_prob(1) + ...
%     likelihood_bg(i)*prior_prob(2); 
% end 

% step 5. extract features from test img 
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

        % extract feature for a single data point x
        x = abs(dct_output_vec); 
        [m, p] = max(x);
        x(p) = min(x); % turn the max into min
        [m, p] = max(x); % now we get the pos for second largest num as p

        % compute its posterior prob and give its prediction
        d = p;
        idx_fg = max(1, min(nBins_fg+1 - sum(d < fg_binEdges), nBins_fg));
        posterior_fg = likelihood_fg(idx_fg)*prior_prob(1);
        idx_bg = max(1 ,min(nBins_bg+1 - sum(d < bg_binEdges), nBins_bg)); 
        posterior_bg = likelihood_bg(idx_bg)*prior_prob(2);
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
saveas(gcf, 'test result.jpg'); 

% estimate the probability of errors 
error = sum(test_results ~= mask, "all") / (255*270);  % 0.1753
