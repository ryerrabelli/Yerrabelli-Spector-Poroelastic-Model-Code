% CREDITS
% Made by Rahul Yerrabelli for Dr. Alexander Spector's lab at Johns Hopkins
% University, Department of Biomedical Engineering. 2017-2020.
% To contact authors, reach out to aspector@jhu.edu for Dr. Spector and
% ryerrab1@alumni.jh.edu or ryerrabelli@gmail.com for Rahul.

% DESCRIPTION
% 1) Under the condition of ramped applied strain (strain that increases
% linearly until time t0 and then is constant afterwards)
% 2) Returns a dimensionalized (can be nondimensional if input shouldDimensionalize is false) 
% value for pressure divided by e0 i.e. p/e0
% Note: r represents the dimensionalized radial position (not the
% nondimensionalized radial position, r' = r/a)

% ARGUMENTS
% time:     Vector containing the time points to be evaluated at (unit of time)
% v:        Matrix containing the two Poisson's ratios, v21 and v31 (unitless)
% E:        Vector containing the two Young's moduli (unit of pressure,
%           i.e. force/area) at the first and third position of the vector
% r:        Vector containing the radial coordinate values to be evaluated
%           at (unit of distance)
% a:        Scalar representing fiber radius. It is used to make r
%           nondimensional (unit of distance) and take values on [0,1]
% t0_tg:    Scalar representing t0/tg (t0 divided by tg). t0 represents the
%           time when the ramped strain ends and becomes flat (unitless 
%           since t0 and tg are in units of time)
% tg:       Scalar time constant called the gel diffusion time (unit of time)
% shouldDimensionalize:
%           Boolean indicating whether the result should be dimensionalized or (is dimensionalized by default)
% Note: porosity is not an argument because this pressure equation does not
% depend on porosity

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function press = ramped_press_eqn( time, v, E, r, a, t0_tg, tg, shouldDimensionalize )
    if nargin < 8 %if number of inputs < 8 (i.e. shouldDimensionalize is not entered)
        shouldDimensionalize = true; % then make the value true as default
    end

    v21 = v(2,1);
    v31 = v(3,1);
   
    v21=v(2,1);
    v31=v(3,1);
    E1=E(1);
    E3=E(3);
    E1_E3=E1/E3;
    %tg is 1/(HA1*k1/a^2)
    
    A1=0;
    C11 = E1/(1+v21) * (1   - v31^2*E1_E3) / (1-v21-2*v31^2*E1_E3);
    C12 = E1/(1+v21) * (v21 + v31^2*E1_E3) / (1-v21-2*v31^2*E1_E3);
    C13 = E1         * v31                 / (1-v21-2*v31^2*E1_E3);
    C33 = E3*(1 +            2*v31^2*E1_E3 / (1-v21-2*v31^2*E1_E3));
    
    C0 = (C11-C12)/C11; %=2/C11p
    C1 = (2*C33+C11+C12-4*C13)/(C11-C12); %not sure if I should remove this *E3
    C2 = 2*[C33*(C11-C12)+C11*(C11+C12-4*C13)+2*C13^2]/(C11-C12)^2; %not sure if I should remove this *E3
    C11p = 2*C11/(C11-C12);
    C12p = 2*C12/(C11-C12);
    C13p = 2*C13/(C11-C12);
    k = C11+C12-2*C13;
    kp = C11p+C12p-2*C13p;
    
    bessel_len = 20;
    list = zeros(1,bessel_len);
    C = (1 - v31^2 * E1_E3)/(1 - v21 - 2 * v31^2 * E1_E3); %is equal to 1/C0
    eqn = @(alpha) besselj(1, alpha) - C*alpha*besselj(0, alpha);

    for n = 1:bessel_len
        list(n) = fzero(eqn, n*pi);
    end
    
    press = zeros([length(time) length(r)]);
    for i = 1:length(time)
        t = time(i); %to get unitless time, divide by tg
        for it_r = 1:length(r)
            r_i = r(it_r);
            if t <= t0_tg*tg
                summation = 0;
                for n=1:bessel_len
                    alphaN = list(n);
                    s=-alphaN^2;
                    %tg is 1/(HA1*k1/a^2)
                    expon = exp(-alphaN^2*t/tg);
                    besselrat=besseli(0,sqrt(s)*r_i/a)/besseli(0,sqrt(s));
                    %p/gprime
                    p = (besselrat-1)*C0/2;
                    gprime = (alphaN^2*(alphaN^2/(2*C0) + C0/2 - 1));
                    summation = summation + p/gprime*(expon-1);
                end
                
                if (shouldDimensionalize)
                    summation = summation *(C11-C12)/2; %Dimensionalize
                end
                
                %Note: A1 is 0, but is left here for completeness
                press(i,it_r) = C11p*kp/2*tg* (A1*t + summation);
                
            else
                summation = 0;
                for n = 1:bessel_len
                    alphaN = list(n);
                    s=-alphaN^2;
                    %tg is 1/(HA1*k1/a^2)
                    expon1 = exp(-alphaN^2*t/tg);
                    expon2 = exp(-alphaN^2*(t/tg-t0_tg));
                    besselrat=besseli(0,sqrt(s)*r_i/a)/besseli(0,sqrt(s));
                    %p/gprime
                    p = (besselrat-1)*C0/2;
                    gprime = (alphaN^2*(alphaN^2/(2*C0) + C0/2 - 1));                
                    summation = summation + p/gprime*(expon1-expon2);
                end
                
                if (shouldDimensionalize)
                    summation = summation *(C11-C12)/2; %Dimensionalize
                end
                
                %Note: A1 is 0, but is left here for completeness
                press(i,it_r) = C11p*kp/2*tg*(A1*t0_tg*tg+summation);
                
            end
        end
    end
end

% END OF CODE
