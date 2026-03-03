function [quality, band_Ds] = MQNR_Spa(PAN, Fused)
% MQNR_Spa: MVG-based Spatial Distortion (ISCTech 2025 Best Paper)
% Inputs: PAN (HxW), Fused (HxWxB) - both double
% Outputs: quality (mean Mahalanobis), band_Ds (per-band)

blocksizerow = 32; blocksizecol = 32; spatfeatnum = 48;

% Ensure PAN is 2D
if ndims(PAN) > 2, PAN = PAN(:,:,1); end

% Crop for consistency
[PAN, ~, ~] = croppatch(PAN, blocksizerow, blocksizecol);
[Fused, ~, ~] = croppatch(Fused, blocksizerow, blocksizecol);

% PAN reference model
ipan = mat2gray(PAN) * 255;
fea_spat = blockproc(ipan, [blocksizerow blocksizecol], @spatfeature);
feat_spat_pan = reshape(fea_spat', [spatfeatnum, size(fea_spat,1)*size(fea_spat,2)/spatfeatnum])';
mu_pris = mean(feat_spat_pan); cov_pris = cov(feat_spat_pan);

% Fused bands
num_bands = size(Fused, 3);
band_Ds = zeros(1, num_bands);
for i = 1:num_bands
    ifused = mat2gray(Fused(:,:,i)) * 255;
    fea_spat = blockproc(ifused, [blocksizerow blocksizecol], @spatfeature);
    feat_spat_fused = reshape(fea_spat', [spatfeatnum, size(fea_spat,1)*size(fea_spat,2)/spatfeatnum])';
    mu_fused = mean(feat_spat_fused); cov_fused = cov(feat_spat_fused);
    
    pooled_cov = (cov_pris + cov_fused) / 2;
    delta_mu = mu_pris - mu_fused;
    band_Ds(i) = sqrt(delta_mu * pinv(pooled_cov) * delta_mu');
end

quality = mean(band_Ds);
fprintf('MQNR_Spa: %.4f (lower=better spatial fidelity)\n', quality);
end

function [cropped, nr, nc] = croppatch(img, bs_r, bs_c)
[rows, cols, ~] = size(img); nr = floor(rows/bs_r); nc = floor(cols/bs_c);
cropped = img(1:nr*bs_r, 1:nc*bs_c, :);
end
