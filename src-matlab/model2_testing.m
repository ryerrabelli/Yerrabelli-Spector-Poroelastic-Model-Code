%%
syms s;
vs=0;
tg=7e3;
Es=7e6;
eps0=0.001;
a=0.003;
epss = - eps0/s;
alpha = (1-2*vs)/(2*(1+vs));
F = epss*(3*besseli(0,sqrt(s))-8*alpha*besseli(1,sqrt(s))/sqrt(s)) ...
    /(besseli(0,sqrt(s))-2*alpha*besseli(1,sqrt(s))/sqrt(s));
%%
% inputting a value of time=0 doesn't error (just returns None/NaN), but takes longer (about 2x as much) on python; not really MATLAB though
times = [0.05:.05:5]';
times = [0.01:0.01:1]';
ss=[0:0.02:1 1.1:0.1:5];
func = matlabFunction(F);
inv_tal = talbot_inversion(func, times);  % Talbot doesn't perform well for small times (has a lot of NaNs)
tic
inv_eul = euler_inversion(func, times);  
[times inv_tal inv_eul]
toc

laplace_vals = func(ss);

%%
figure; 
subplot(2,1,1)
hold on; grid on;
plot(times, inv_eul, 'rx','DisplayName','Inversion (Euler)')
plot(times, inv_tal, 'bo','DisplayName','Inversion (Talbot)')
xlabel('t');
legend('show')
subplot(2,1,2)
plot(ss, laplace_vals, 'bo','DisplayName','Laplace Space Equation')
xlabel('s');
legend('show')

%%
ilaplace(F)
