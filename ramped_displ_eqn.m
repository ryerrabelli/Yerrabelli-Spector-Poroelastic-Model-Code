% CREDITS
% Made by Rahul Yerrabelli for Dr. Alexander Spector's lab at Johns Hopkins
% University, Department of Biomedical Engineering. 2017-2020.
% To contact authors, reach out to aspector@jhu.edu for Dr. Spector and
% ryerrab1@alumni.jh.edu or ryerrabelli@gmail.com for Rahul.

% DESCRIPTION
% 1) Under the condition of ramped applied strain (strain that increases
% linearly until time t0 and then is constant afterwards)
% 2) Returns a dimensionalized value for displacement divided by e0dot (but not
% divided by r or divided by a), i.e. u/e0dot
% 3) If you divide the result by a, then you get a nondimensionalized 
% (unitless) value for displacement
% Note: r represents the dimensionalized radial position (not the
% nondimensionalized radial position, r' = r/a)
% Note #2: If the input r is actually r/a and a is just given as 1, then
% the output would in fact be u/e0dot/a

% ARGUMENTS
% time:     Vector containing the time points to be evaluated at (unit of time)
% v:        Matrix containing the two Poisson's ratios, v21 and v31 (unitless)
% E:        Vector containing the two Young's moduli (unit of pressure,
%           i.e. force/area) at the first and third position of the vector
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

function displ = ramped_displ_eqn(time, v, E, r, a, t0_tg, tg)
    
    v21=v(2,1);
    v31=v(3,1);
    E1=E(1);
    E3=E(3);
    E1_E3=E1/E3;
    %tg is 1/(HA1*k1/a^2)
    
    A1_r= -v31/(t0_tg*tg); %Note: this is A1/r not A1/r' (i.e. this is A1/(r'*a)
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
    
    displ = zeros([length(time) length(r)]);
    
    for it_t = 1:length(time)
        t = time(it_t); %to get unitless time, divide by tg
        for it_r = 1:length(r)
            r_i = r(it_r);
            % Up until Jun 25, 2019, there was a r_i/a in the denom
            % expression. I now believe that that should not be there
            % according to the equations on the papers. I put it out here
            % to make it easier to change to compare to old results.
            r_iax =  1; %r_i/a; 
            if t <= t0_tg*tg
                summation = 0;
                for n=1:bessel_len
                    alphaN = list(n);
                    s=-alphaN^2;
                    expon = exp(-alphaN^2*t/tg);
                    numer = -kp*besseli(1,sqrt(s)*r_i/a);
                    denom = sqrt(s)*r_iax*besseli(0,sqrt(s))*s*(s/C0^2+2/C0-1);
                    
                    addsum = numer/denom*(expon-1);
                    summation = summation + addsum;
                end
                displ(it_t,it_r) = A1_r*r_i*t - 0.5*tg*r_i*[+summation];
            else
                summation = 0;
                for n = 1:bessel_len
                    alphaN = list(n);
                    s=-alphaN^2;
                    %tg is 1/(HA1*k1/a^2)
                    expon1 = exp(-alphaN^2*t/tg);
                    expon2 = exp(-alphaN^2*(t/tg-t0_tg));
                    numer = -kp*besseli(1,sqrt(s)*r_i/a);
                    denom = sqrt(s)*r_iax*besseli(0,sqrt(s))*s*(s/C0^2+2/C0-1);
                    
                    addsum = numer/denom*(expon1-expon2);
                    summation = summation + addsum;
                end
                displ(it_t,it_r) = A1_r*r_i*t0_tg*tg - 0.5*tg*r_i*[+summation];
            end
        end
    end
    
end

% END OF CODE
