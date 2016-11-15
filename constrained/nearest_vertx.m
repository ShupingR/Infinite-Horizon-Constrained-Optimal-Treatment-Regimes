% generate grids for discretization
% obs of state variable will be pointed to the nearest grids (nearby grids 
% based on triangularization)
% policy evaluaiton based on the states linked to
function [ closest_index_current, closest_index_next ] = nearest_vertx(grid, sample)
    % generate uniform grids for now
    point_current = [sample.wellness, sample.tumorsize];
    closest_index_current = dsearchn(grid, point_current);
    % nearest_current = grid(closest_index_current, :);
    
    point_next = [sample.next_wellness, sample.next_tumorsize];
    closest_index_next = dsearchn(grid, point_next);
    % nearest_next = grid(closest_index_next, :);
end

