%% CALCULATE AREA OF A TRIANGLE
%  Simple script to test running MATLAB from Python

% Calculate area:
function [a] = triangle_area(b,h)
a = 0.5*(b.* h);   %area
save('a.mat','a')
end
