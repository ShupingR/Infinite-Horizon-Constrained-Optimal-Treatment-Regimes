function qval = qfun(m, w, d, f, med_m, med_w, weights, k)
     qval = -1 * ( ( d >= 0 && d <= 1) * ...
            feature(m, w, d, f, med_m, med_w, k)' * weights );
                % ...- ( d < 0 || d > 1) * 100000 );
     %if(qval >= 0)
        
     %else
     %    qval = 0;
     %end
end