clear all;

eps0 = 0.1; % 10 percent
strain_rate = 0.1; % 1 percent per s (normally 1%/s)

ln = @(x) log(x);  % natural log

% Parameters to be determined by exprimental fitting to the unknown
% material
tg=40.62; %in units of s   % for porosity_sp == 0.5
Err = 1;
% Not actually a v, but a greek nu used to represent Poisson's ratio
Vrtheta = 1;
c = 1;
%tau1 = 1;
%tau2 = 1;
%tau = [tau1 tau2]
tau = [1 1]

Vrz = 1;
Ezz = 1;


syms s t
% Test inversion
F = 1/s^2+1/s;t
ilaplace(F, s, t)  % Specificying s and t is not actually needed as those are the default


%  1
%eps0 = strain_rate * t0
t0 = eps0/strain_rate;
epszz = 1 - exp(-s*t0)/s^2;  %#  Laplace transform of the axial strain



%  2
Srr     = 1/Err;
Srtheta = -Vrtheta/Err;
Srz     = -Vrz/Err;
Szz     = 1/Ezz;


%  3
C13     = @(Sij)  Srz/(alpha(Sij));
C33     = @(Sij) -(Srr+Srtheta)/(alpha(Sij));
alpha   = @(Sij) 2*Srz^2-Szz*Srtheta-Srr*Szz;


%  4
g       = @(Sij) -(2*Srz+Szz)*(Srr-Srtheta)/(alpha(Sij));

%  5  
% !!Confirm- can't it be simplified bc Srz divided and multiplied by 2
f1      = @(Sij) -(2*Srz/2+Szz)*(Srr*Szz-Srz^2)/(alpha(Sij));

%  6
% Viscoelastic parameters: c, tau 1, tau 2
f2      = @(c,tau) 1 + c*ln( (1+s*tau(2))/(1+s*tau(1)) );

%  7
% !!Confirm should be a function of Sij
Ecap    = @(Sij) -2*(Srr*Szz-Srz^2)/(alpha(Sij));


%  8
%f       = r0^2*s / (Ecap*k*f2(c,tau))
% Simplified using tg=r0^2/(Ecap*k)
% !!Confirm should be a function of c, tau
f       = tg * s/f2(c,tau);


% Laplace transform of the average axial stress
% !!Confirm should be a function of Sij
sigcap  = @(Sij) ...
    2*epszz*(C13(Sij)*(g(Sij)) * besseli(1,sqrt(f)/sqrt(f))/(Ecap(Sij)*besseli(0,sqrt(f))-2*besseli(1,sqrt(f))/sqrt(f)) -1/2) ...
    + C33(Sij)/2 ...
    + f1(Sij)*f2(c,tau)*...
        (besseli(0,sqrt(f))-2*besseli(1,sqrt(f))/sqrt(f))...
        /(2*Ecap(Sij)*besseli(0,sqrt(f)) - besseli(1,sqrt(f)/sqrt(f)) );