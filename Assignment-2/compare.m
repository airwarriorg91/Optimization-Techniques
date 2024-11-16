%%% Testing fminunc function

%%% Unimodal Function
x0 = [-5 5];
tic;
[x, fval] = fminunc(@unimodalBenchmark, x0);
uniTime = toc;
disp('Unimodal Benchmark:');
disp(['x = ', mat2str(x)]);
disp(['fval = ', num2str(fval)]);

%%% Multimodal Function

%%% Local Minima
x0 = [-5 5];
tic;
[x, fval] = fminunc(@multimodalBenchmark, x0);
multiTime1 = toc;
disp('Multimodal Benchmark (Local Minima):');
disp(['x = ', mat2str(x)]);
disp(['fval = ', num2str(fval)]);

%%% Global Minima
x0 = [-0.1 0.1];
tic;
[x, fval] = fminunc(@multimodalBenchmark, x0);
multiTime2 = toc;
disp('Multimodal Benchmark (Global Minima):');
disp(['x = ', mat2str(x)]);
disp(['fval = ', num2str(fval)]);

%%% Testing GA Function

%%% Unimodal Function
tic;
xGA1 = ga(@unimodalBenchmark,2);
gatime1 = toc;

%%% Unimodal Function
tic;
xGA2 = ga(@multimodalBenchmark,2);
gatime2 = toc;

%%% Unimodal Benchmark Function
function out = unimodalBenchmark(x)
    % Bohachevsky Unimodal function
    % f(x,y) = x^2 + 2*y^2 - 0.3*cos(3*pi*x) - 0.4*cos(4*pi*y) + 0.7
    % Minimum at (0,0)
    out = x(1)^2 + 2*x(2)^2 - 0.3*cos(3*pi*x(1)) - 0.4*cos(4*pi*x(2)) + 0.7;
end

%%% Multimodal Benchmark Function
function out = multimodalBenchmark(x)
    % Ackley Function
    % f(x,y) = -20exp(-0.2*sqrt(0.5*(x^2 + y^2))) - exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + 20 + exp(1)
    % Global Minimum at (0,0)
    out = -20*exp(-0.2*sqrt(0.5*(x(1)^2 + x(2)^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + 20 + exp(1);
end


   