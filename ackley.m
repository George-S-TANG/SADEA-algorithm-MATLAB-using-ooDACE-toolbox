function [y] = ackley(xx, a, b, c)
% Inputs
% 
%     xx: A row vector of the input variables [x1, x2, ..., xd]
%     a: A constant parameter of the function (optional). The default value is 20.
%     b: A constant parameter of the function (optional). The default value is 0.2.
%     c: A constant parameter of the function (optional). The default value is 2*pi.
% 
% Outputs
% 
%     y: The value of the Ackley function at the input point xx.
% 
% Details
% 
% The Ackley function is a commonly used optimization benchmark function with a global minimum at (0,0). It is defined as follows:
% f(x) = -a * exp(-b*sqrt(1/d * sum(x.^2))) - exp(1/d * sum(cos(c*x))) + a + exp(1)

d = length(xx);

if (nargin < 4)
    c = 2*pi;
end
if (nargin < 3)
    b = 0.2;
end
if (nargin < 2)
    a = 20;
end

sum1 = 0;
sum2 = 0;
for ii = 1:d
	xi = xx(ii);
	sum1 = sum1 + xi^2;
	sum2 = sum2 + cos(c*xi);
end

term1 = -a * exp(-b*sqrt(sum1/d));
term2 = -exp(sum2/d);

y = term1 + term2 + a + exp(1);

end