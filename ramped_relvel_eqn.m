% CREDITS
% Made by Rahul Yerrabelli for Dr. Alexander Spector's lab at Johns Hopkins
% University, Department of Biomedical Engineering. 2017-2020.
% To contact authors, reach out to aspector@jhu.edu for Dr. Spector and
% ryerrab1@alumni.jh.edu or ryerrabelli@gmail.com for Rahul.

% DESCRIPTION
% 1) Under the condition of ramped applied strain (strain that increases
% linearly until time t0 and then is constant afterwards)
% 2) Returns a dimensionalized value for fluid velocity divided by e0 and
% divided by r, i.e. v^f_r/(e0 * r)
% NOTE: The equation was not reconfirmed while adding this extra comment
% text (which was originally done Jun 22nd, 2019)

% ARGUMENTS
% time:     Vector containing the time points to be evaluated at (unit of time)
% v:        Matrix containing the two Poisson's ratios, v21 and v31 (unitless)
% E:        Vector containing the two Young's moduli at the first and third
%           position of the vector (unit of pressure, i.e. force/area)
% r:        Vector containing the radial coordinate values to be evaluated
%           at (unit of distance)
% a:        Scalar representing fiber radius. It is used to make r
%           nondimensional (unit of distance) and take values on [0,1]
% porosity: Scalar (unitless)
% t0_tg:    Scalar representing t0/tg (t0 divided by tg). t0 represents the
%           time when the ramped strain ends and becomes flat (unitless 
%           since t0 and tg are in units of time)
% tg:       Scalar time constant called the gel diffusion time (unit of time)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%For stepwise applied strain 
%Returns a dimensionalized value for relative velocity divided 
%by e0 divided by r
%Relative velocity defined as = v^f-v^s=-(1+?)[?u/?t+0.5*r*??/?t]

function [ relvel_r ] = ramped_relvel_eqn( time, v, E, r, a, porosity, t0_tg,tg )
    
    v21=v(2,1);
    v31=v(3,1);
    E1=E(1);
    E3=E(3);
    E1_E3=E1/E3;
    %tg is 1/(HA1*k1/a^2)
    
    A1_r= -v31/(t0_tg*tg);
    C11 = E1/(1+v21) * (1   - v31^2*E1_E3) / (1-v21-2*v31^2*E1_E3);
    C12 = E1/(1+v21) * (v21 + v31^2*E1_E3) / (1-v21-2*v31^2*E1_E3);
    C13 = E1         * v31                 / (1-v21-2*v31^2*E1_E3);
    C33 = E3*(1 +            2*v31^2*E1_E3 / (1-v21-2*v31^2*E1_E3));
    
    C0 = (C11-C12)/C11;
    C1 = (2*C33+C11+C12-4*C13)/(C11-C12); %not sure if I should remove this *E3
    C2 = 2*[C33*(C11-C12)+C11*(C11+C12-4*C13)+2*C13^2]/(C11-C12)^2; %not sure if I should remove this *E3

    C11p = C11/(C11-C12);
    C12p = C12/(C11-C12);
    C13p = C13/(C11-C12);
    kp = C11p+C12p-2*C13p; %p means prime
    
    bessel_len = 20;
    list = zeros(1,bessel_len);
    C = (1 - v31^2 * E1_E3)/(1 - v21 - 2 * v31^2 * E1_E3); %is equal to 1/C0
    eqn = @(alpha) besselj(1, alpha) - C*alpha*besselj(0, alpha);

    for n = 1:bessel_len
        list(n) = fzero(eqn, n*pi);
    end
    
    relvel_r = zeros([length(time) length(r)]);
    
    for it_t = 1:length(time)
        t = time(it_t); %to get unitless time, divide by tg
        for it_r = 1:length(r)
            r_i = r(it_r);
            r_ip = r_i/a;
            if t <= t0_tg*tg
                summation = 0;
                for n=1:bessel_len
                    alphaN = list(n);
                    s=-alphaN^2;
                    expon = exp(-alphaN^2*t/tg);
                    
                    numer = -kp*C0*besseli(1,sqrt(s)*r_ip)/(sqrt(s)*r_ip);
                    denom = 2*besseli(0,sqrt(s))*s*(s/(2*C0)-C0/2+1);
                    
                    summation = summation + alphaN^2*numer/denom*(expon);
                end
                du_dt = A1_r + 1/2*summation; %divided by r and eps0dot
                deps_dt = 1; %divided by r and eps0dot
                relvel_r(it_t,it_r) = -(porosity+1)*(du_dt+0.5*deps_dt);
            else
                summation = 0;
                for n = 1:bessel_len
                    alphaN = list(n);
                    s=-alphaN^2;
                    %tg is 1/(HA1*k1/a^2)
                    expon1 = exp(-alphaN^2*t/tg);
                    expon2 = exp(-alphaN^2*(t/tg-t0_tg));
                    
                    numer = -kp*C0*besseli(1,sqrt(s)*r_ip)/(sqrt(s)*r_ip);
                    denom = 2*besseli(0,sqrt(s))*s*(s/(2*C0)-C0/2+1);
                    
                    summation = summation + alphaN^2*numer/denom*(expon1-expon2);
                end
                
                du_dt = 0 + 1/2*summation; %divided by r and eps0dot
                deps_dt = 0; %divided by r and eps0dot
                
                relvel_r(it_t,it_r) = -(porosity+1)*(du_dt+0.5*deps_dt);
            end
        end
    end


end

