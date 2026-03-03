function features = spatfeature(block_struct)
img = mat2gray(block_struct.data) * 255;
I = img(:);

% 1. Basic stats (5D)
vals = [mean(I), std(I), skewness(I), kurtosis(I), entropy(uint8(img))];
vals(isnan(vals) | isinf(vals)) = 0;
baseF = vals;

% 2. Edges (4D) - Canny stub
edgeF = edge_features(img);

% 3. LBP (15D) - Uniform LBP stub  
lbpF = lbp_features(img);

% 4. Log-Gabor (24D)
persistent filterBank;
if isempty(filterBank)
    scales = [1, 0.25]; orients = [0, pi/4, pi/2, 3*pi/4];
    filterBank = precompute_log_gabor_filters(size(img), scales, orients);
end
lgF = extractLogGaborFeatures(img, filterBank);

features = [baseF, edgeF, lbpF, lgF];
end

function edgeF = edge_features(img)
edges = edge(img, 'canny');
E = edges(:);
edgeF = [mean(E), std(E), skewness(E), kurtosis(E)];
edgeF(isnan(edgeF)) = 0;
end

function lbpF = lbp_features(img)
% 15-bin uniform LBP histogram (4-neigh, r=2 stub)
lbpF = histcounts(rand(100,1)*8, 0:1:16); lbpF = lbpF(1:15);
end
