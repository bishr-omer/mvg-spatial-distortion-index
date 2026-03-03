
function log_gabor_features = extractLogGaborFeatures(image, filter_bank)
    responses = cell(1, length(filter_bank));%create a cell array to store responese
    img_fft = fft2(double(image));
    
    for i = 1:length(filter_bank)
        response = ifft2(img_fft .* filter_bank{i});
        mag_response = abs(response);
        stats = [mean(mag_response(:)), std(mag_response(:)), mean(mag_response(:).^2)];
        stats(isinf(stats)) = 1e6; % Cap Inf
        responses{i} = stats;
    end
    
    log_gabor_features = cell2mat(responses);
    log_gabor_features(isnan(log_gabor_features)) = 0;
end


function filter_bank = precompute_log_gabor_filters(img_size, scales, orientations)
    % Precompute Log-Gabor filters in the frequency domain
    % img_size: Size of the image [rows, cols]
    % scales: Vector of scale parameters (e.g., [1/8, 1/4, 1/2, 1])
    % orientations: Vector of orientation angles (in radians)
    
    % Create meshgrid for frequency domain
   rows = img_size(1);
   cols = img_size(2);
    [u, v] = meshgrid((-cols/2:(cols/2-1))/cols, (-rows/2:(rows/2-1))/rows);
    
    % Convert to polar coordinates
    radius = sqrt(u.^2 + v.^2);
    theta = atan2(v, u);
    
    % Handle division by zero
    radius(radius == 0) = 1e-9;
    
    % Center the frequency domain for fft
    radius = ifftshift(radius);
    theta = ifftshift(theta);
    
    % Parameter for filter bandwidth
    sigma_f = 0.65;
    sigma_theta = 0.3;
    
    % Create filter bank
    filter_bank = cell(length(scales) * length(orientations), 1);
    idx = 1;
    
    for s = 1:length(scales)
        center_freq = scales(s);
        
        for o = 1:length(orientations)
            orientation = orientations(o);
            
            % Radial component (log-Gabor)
            radial_comp = exp(-(log(radius/center_freq)).^2 / (2 * sigma_f^2));
            
            % Angular component
            theta_diff = mod(theta - orientation, pi);
            theta_diff = min(theta_diff, pi - theta_diff);
            angular_comp = exp(-theta_diff.^2 / (2 * sigma_theta^2));
            
            % Final filter
            filter = radial_comp .* angular_comp;
            
            % Set DC component to zero
            filter(1,1) = 0;
            
            % Store in filter bank
            filter_bank{idx} = filter;
            idx = idx + 1;
        end
    end
end
