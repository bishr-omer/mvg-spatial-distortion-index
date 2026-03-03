function filter_bank = precompute_log_gabor_filters(img_size, scales, orientations)
    % Precompute log-Gabor filters in frequency domain
    rows = img_size(1);
    cols = img_size(2);
    
    % Create frequency grid
    [u, v] = meshgrid(([1:cols]-(fix(cols/2)+1))/(cols/2), ...
                      ([1:rows]-(fix(rows/2)+1))/(rows/2));
    
    % Calculate radius and angle
    radius = sqrt(u.^2 + v.^2);
    radius(radius == 0) = 1;  % Avoid division by zero
    theta = atan2(v, u);
    
    % Parameters for log-Gabor
    sigma_f = 0.7;  % Bandwidth parameter
    
    % Create filter bank
    filter_bank = cell(1, length(scales) * length(orientations));
    
    % Generate filters for each scale and orientation
    filter_idx = 1;
    for s = 1:length(scales)
        wavelength = scales(s);
        f0 = 1 / wavelength;  % Center frequency
        
        % Log-Gabor radial component (frequency scale)
        radial_component = exp(-((log(radius/f0)).^2) / (2 * sigma_f^2));
        
        % Set DC component to zero
        radial_component(1,1) = 0;
        
        for o = 1:length(orientations)
            angle = orientations(o);
            
            % Angular component (orientation)
            sigma_theta = 0.5;  % Angular bandwidth
            angular_component = exp(-((theta - angle).^2) / (2 * sigma_theta^2));
            
            % Create filter
            filter_bank{filter_idx} = radial_component .* angular_component;
            filter_idx = filter_idx + 1;
        end
    end
    
    % Shift to correct position in frequency domain
    for i = 1:length(filter_bank)
        filter_bank{i} = ifftshift(filter_bank{i});
    end
end
