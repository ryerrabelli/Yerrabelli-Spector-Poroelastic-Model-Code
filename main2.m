%% CREDITS
% Made by Rahul Yerrabelli for Dr. Alexander Spector's lab at Johns Hopkins
% University, Department of Biomedical Engineering. 2021-.
% To contact authors, reach out to aspector@jhu.edu for Dr. Spector and
% ryerrab1@alumni.jh.edu or rsy2@illinois.edu for Rahul.

% DESCRIPTION
% TODO:

% ARGUMENTS
% TODO:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% NOTES FROM EMAIL CHAIN
% Email chain on Mar 28, 2021. 
% Questions from Rahul, answers from Dr. Spector.
% *** means not addressed
%
% ------------------------------------------------------------------------
% QUESTION 1:
% The list of fitted parameters (tg, Err, Vr?, c, tau1, tau2) should also
% Vrz and Ezz. Otherwise, what is the equation to get Vrz and Ezz?
%
% ANSWER: 
% Ezz and Vrz are not the subjects of stress relaxation fitting
% because they are directly determined by the mesh deformation part of the
% experiment (see our paper with Daniel).
%
%
% ------------------------------------------------------------------------
% QUESTION 2:
% In equation 5, the first Srz is both multiplied by and divided by 2.
% Please confirm that these cancel and the equation can be simplified.
%
% ANSWER: 
% Yes two 2s cancel each  other, but it was convenient to keep for
% the equation check.
%
%
% ------------------------------------------------------------------------
% QUESTION 3:
% Confirm equation 7 is a function of Sij (like equation 4 and 5 are) even
% though the arguments are not explicitly stated
%
% ANSWER:
% Confirm
%
%
% ------------------------------------------------------------------------
% ***QUESTION 4:
% Confirm equation 8 is a function of c and tau (like equation 6 is) even
% though the arguments are not explicitly stated
%
% ANSWER:
% It not true because Ehat is a function of Sij.
%
%
% ------------------------------------------------------------------------
% QUESTION 5:
% Confirm tau is a two-element vector of tau1 and tau2  (thus, equation 6
% has a total of 3 arguments, not 2)
%
% ANSWER:
% There were places where I wrote just tau  meaning (tau1,tau2).
%
%
% ------------------------------------------------------------------------
% ***QUESTION 6:
% In the final equation for the average axial stress, confirm that this is
% still a function of Sij. In other words, Sij's values are not defined (or
% iterated through) to get the final result
%
% ANSWER: 
% The sigma bar function is a function of 6 parameters:  Err,
% Vrtheta , tg, c, tau1, tau2.
%
%
% ------------------------------------------------------------------------
% 

%% SETUP

clear all;

ln = @(x) log(x);  % ln (x) = natural log = log(x) in matlab/coding
I0 = @(x) besseli(0,x);
I1 = @(x) besseli(1,x);

% Symbolics for Laplace
syms s t

% Add the subfolder with the library
% Note- quotes cannot be used around the library name
% How to add libraries: https://www.mathworks.com/help/matlab/matlab_oop/organizing-classes-in-folders.html
% This library: https://www.mathworks.com/matlabcentral/fileexchange/39035-numerical-inverse-laplace-transform?s_tid=mwa_osa_a
% We need this library bc the built-in ilaplace function of matlab only
% gives closed form solutions, which is not possible here.
addpath Numerical_Inverse_Laplace_Transform


%% PARAMETERS

% Predefined constants
eps0 = 0.1; % 10 percent
strain_rate = 0.1; % 1 percent per s (normally 1%/s)
% Below are directly determined by the mesh deformation part of the 
% experiment (see our paper with Daniel).  -Dr. Spector
Vrz = 0.5; % Not actually v, but greek nu (represents Poisson's ratio)
Ezz = 10;  % Note- don't mix up Ezz with epszz

 

% Fitted parameters (to be determined by experimental fitting to 
% the unknown material)
c = 1;
tau1 = 1;
tau2 = 1;
%tau = [tau1 tau2];
%tau = [1 1];
tg=40.62; %in units of s   % for porosity_sp == 0.5
Vrtheta = 1; % Not actually v, but greek nu (represents Poisson's ratio)
Err = 1;






%% BASE EQUATIONS
%  1
%eps0 = strain_rate * t0
t0 = eps0/strain_rate;
epszz = 1 - exp(-s*t0)/(s*s);  %  Laplace transform of the axial strain



%  2
Srr     = 1/Err;
Srtheta = -Vrtheta/Err;
Srz     = -Vrz/Err;
Szz     = 1/Ezz;
%Sij     = [Srr, Srtheta, Srz;   Srtheta, Srr, Srz;   Srz, Srz, Szz];

%  3
alpha   =  2*Srz*Srz-Szz*Srtheta-Srr*Szz;
C13     =   Srz/(alpha);
C33     =  -(Srr+Srtheta)/(alpha);


%  4
g       =  -(2*Srz+Szz)*(Srr-Srtheta)/(alpha);

%  5  
% Note- below could be simplified bc both divided and multiplied by 2
f1      =  -(2*Srz+Szz)/2 * 2*(Srr*Szz-Srz*Srz)/(alpha);

%  6
% Viscoelastic parameters: c, tau 1, tau 2
f2      = 1 + c*ln( (1+s*tau2)/(1+s*tau1) );

%  7
% Note- Ehat is a function of Sij although wasn't stated in Spector's notes
Ehat    =  -2*(Srr*Szz-Srz*Srz)/(alpha);


%  8
%f      =  r0^2*s / (Ehat*k*f2(c,tau1,tau2))
% Simplified using tg=r0^2/(Ehat*k)
% !!Confirm should be a function of c, tau also maybe Sij or tg
f       = tg * s/f2;


%% FINAL EQUATION
% Laplace transform of the average axial stress
% !!Confirm should be a function of Sij
sigbar  =  ... 
    2*epszz*(...
        C13...
            *(...
                g ...
                    * I1(sqrt(f))/sqrt(f) ...
                    /(Ehat*I0(sqrt(f))-2*I1(sqrt(f))/sqrt(f)) ...
                -1/2 ...
            ) ...
        + C33/2 ...
        + f1...
            *f2*...
            (I0(sqrt(f))-2*I1(sqrt(f))/sqrt(f))...
            /(2 * ( Ehat*I0(sqrt(f)) - I1(sqrt(f))/sqrt(f) ) ) ...
    );


%% TEST MATLAB'S LAPLACE INVERSION 
F = 1/(s*s)+1/s;
% Specifying s and t in ilaplace isn't actually needed as those 
% are the default
f = ilaplace(F, s, t);  % Output of ilaplace is actually of class/type char
ezplot(f)
%ezplot(f, [-10 10])


%% LAPLACE INVERSION - CLOSED BOUND SOLUTION <-- not possible for sigbar
F = epszz;
F = sigbar;
% Specifying s and t in ilaplace isn't actually needed as those 
% are the default
f = ilaplace(F, s, t);  % Output of ilaplace is actually of class/type char
% ezplot(f, [-10 10])
f = matlabFunction(f);  % convert char to actual function
ts=[-2:0.01:2];
%plot(ts,f(ts))

%% LAPLACE INVERSION - NUMERICAL  <-- uses external library
F = sigbar;
time = [0:.05:5]';
inv_tal = talbot_inversion(matlabFunction(F), time);  % Talbot doesn't perform well for small times (has a lot of NaNs)
inv_eul = euler_inversion(matlabFunction(F), time);  
[time inv_tal inv_eul]
