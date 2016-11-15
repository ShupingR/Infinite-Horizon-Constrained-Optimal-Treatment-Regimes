% generate grids for discretization
% obs of state variable will be pointed to the nearest grids (nearby grids 
% based on triangularization)

function grid = gen_grid(min_x, max_x, min_y, max_y, M_x, M_y)
    % generate uniform grids for now
    m = 1;
    grid = nan(M_x*M_y, 2);
    for m_x = 1: M_x
        for m_y = 1: M_y
            grid(m, 1)= min_x + m_x*(max_x - min_x)/M_x;
            grid(m, 2)= min_y + m_y*(max_y - min_y)/M_y;
            m = m+1;
        end
    end
end

