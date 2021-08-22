% Made by Rahul Yerrabelli for Dr. Alexander Spector's lab at Johns Hopkins
% University, Department of Biomedical Engineering. 2017-2020.
% To contact authors, reach out to aspector@jhu.edu for Dr. Spector and
% ryerrab1@alumni.jh.edu or rsy2@illinois.edu for Rahul.

% For force divided by e0 from ramp applied strain
% Returns a dimensionalized value

function f = force_eqn( time, v, E, t0_tg, tg )
    v21 = v(2,1);
    v31 = v(3,1);
   
    v21=v(2,1);
    v31=v(3,1);
    E1=E(1);
    E3=E(3);
    E1_E3=E1/E3;
    %tg is 1/(HA1*k1/a^2)
    
    A1=E3;
    C11 = E1/(1+v21) * (1   - v31^2*E1_E3) / (1-v21-2*v31^2*E1_E3);
    C12 = E1/(1+v21) * (v21 + v31^2*E1_E3) / (1-v21-2*v31^2*E1_E3);
    C13 = E1         * v31                 / (1-v21-2*v31^2*E1_E3);
    C33 = E3*(1 +            2*v31^2*E1_E3 / (1-v21-2*v31^2*E1_E3));
    
    C0 = (C11-C12)/C11;
    C1 = (2*C33+C11+C12-4*C13)/(C11-C12); %not sure if I should remove this *E3
    C2 = 2*[C33*(C11-C12)+C11*(C11+C12-4*C13)+2*C13^2]/(C11-C12)^2; %not sure if I should remove this *E3

    bessel_len = 20;
    list = zeros(1,bessel_len);
    C = (1 - v31^2 * E1_E3)/(1 - v21 - 2 * v31^2 * E1_E3); %is equal to 1/C0
    eqn = @(alpha) besselj(1, alpha) - C*alpha*besselj(0, alpha);

    for n = 1:bessel_len
        list(n) = fzero(eqn, n*pi);
    end
    
    f = zeros(size(time));
    for i = 1:length(time)
        t = time(i); %to get unitless time, divide by tg
        if t <= t0_tg*tg
            summation = 0;
            for n=1:bessel_len
                alphaN = list(n);
                expon = exp(-alphaN^2*t/tg);
                %1 represents the bessel which cancels out when dividing
                %p/gprime
                p = (C1 - C2) * 1;
                gprime = (alphaN^2*(alphaN^2/(2*C0) + C0/2 - 1)) * 1;
                summation = summation + 3.6116*tg*p/gprime*(expon-1);
            end
            f(i) = A1*t+summation*(C11-C12)/2;
        else
            summation = 0;
            for n = 1:bessel_len
                alphaN = list(n);
                %tg is 1/(HA1*k1/a^2)
                expon1 = exp(-alphaN^2*t/tg);
                expon2 = exp(-alphaN^2*(t/tg-t0_tg));
                %1 represents the bessel which cancels out when dividing
                %p/gprime
                p = (C1 - C2) * 1;
                gprime = (alphaN^2*(alphaN^2/(2*C0) + C0/2 - 1)) * 1;                
                summation = summation + tg*p/gprime*(expon1-expon2);
            end
            f(i) = A1*t0_tg*tg+3.6116*summation*(C11-C12)/2;
        end
    end
    
end