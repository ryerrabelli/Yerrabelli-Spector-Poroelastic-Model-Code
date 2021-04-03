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
% The sigma bar function is a function of 6 parametrov:  Err,
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


%% PARAMETERS

%# Predefined constants
eps0 = 0.1; % 10 percent
strain_rate = 0.1; % 1 percent per s (normally 1%/s)
%# Below are directly determined by the mesh deformation part of the 
%# experiment (see our paper with Daniel).  -Dr. Spector
Vrz = 1; % Not actually v, but greek nu (represents Poisson's ratio)
Ezz = 1;  % Note- don't mix up Ezz with epszz


%# Fitted parameters (to be determined by experimental fitting to 
% the unknown material)
c = 1;
%tau1 = 1;
%tau2 = 1;
%tau = [tau1 tau2];
tau = [1 1];
tg=40.62; %in units of s   % for porosity_sp == 0.5
Vrtheta = 1; % Not actually v, but greek nu (represents Poisson's ratio)
Err = 1;


%# Symbolics for Laplace
syms s t



%% BASE EQUATIONS
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
% Note- below could be simplified bc both divided and multiplied by 2
f1      = @(Sij) -(2*Srz+Szz)/2 * 2*(Srr*Szz-Srz^2)/(alpha(Sij));

%  6
% Viscoelastic parameters: c, tau 1, tau 2
f2      = @(c,tau) 1 + c*ln( (1+s*tau(2))/(1+s*tau(1)) );

%  7
% Note- Ehat is a function of Sij although wasn't stated in Spector's notes
Ehat    = @(Sij) -2*(Srr*Szz-Srz^2)/(alpha(Sij));


%  8
%f      = @(Sij) r0^2*s / (Ehat(Sij)*k*f2(c,tau))
% Simplified using tg=r0^2/(Ehat*k)
% !!Confirm should be a function of c, tau also maybe Sij or tg
f       = tg * s/f2(c,tau);


%% FINAL EQUATION
% Laplace transform of the average axial stress
% !!Confirm should be a function of Sij
sigbar  = @(Sij) ...
    2*epszz*(...
        C13(Sij)...
            *(...
                g(Sij) ...
                    * I1(sqrt(f))/sqrt(f) ...
                    /(Ehat(Sij)*I0(sqrt(f))-2*I1(sqrt(f))/sqrt(f)) ...
                -1/2 ...
            ) ...
        + C33(Sij)/2 ...
        + f1(Sij)...
            *f2(c,tau)*...
            (I0(sqrt(f))-2*I1(sqrt(f))/sqrt(f))...
            /(2 * ( Ehat(Sij)*I0(sqrt(f)) - I1(sqrt(f))/sqrt(f) ) ) ...
    );


%% TEST MATLAB'S LAPLACE INVERSION 
F = 1/s^2+1/s;
% Specifying s and t in ilaplace isn't actually needed as those 
% are the default
f = ilaplace(F, s, t);  % Output of ilaplace is actually of class/type char
ezplot(f)
%ezplot(f, [-10 10])


%% LAPLACE INVERSION
F = epszz;
F = sigbar;
% Specifying s and t in ilaplace isn't actually needed as those 
% are the default
f = ilaplace(F, s, t);  % Output of ilaplace is actually of class/type char
ezplot(f, [-10 10])
