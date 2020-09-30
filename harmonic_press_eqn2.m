% CREDITS
% Made by Rahul Yerrabelli for Dr. Alexander Spector's lab at Johns Hopkins
% University, Department of Biomedical Engineering. 2017-2020.
% To contact authors, reach out to aspector@jhu.edu for Dr. Spector and
% ryerrab1@alumni.jh.edu or rsy2@illinois.edu for Rahul.

% Result is divided by epsilon0

function [ press ] = harmonic_press_eqn2( times, v, E, r, a, alph, omega, tg, shouldDimensionalize)
%HARMONIC_PRESS_EQN Summary of this function goes here
%   Detailed explanation goes here
    
    clear i; %necessary so that i indicates the imaginary constant

    v21=v(2,1);
    v31=v(3,1);
    E1=E(1);
    E3=E(3);
    E1_E3=E1/E3;
    %tg is 1/(HA1*k1/a^2)
    
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
    R = C11+C12-2*C13;
    
    rp = r./a;
    rtia= sqrt(i)*alph;
    %{
    A= 0.5*R/(rtia*(C11*besseli(0,rtia)-(C11-C12)*besseli(1,rtia)/rtia )); %Note: this is A1/r not A1/r' (i.e. this is A1/(r'*a)
    B= 0.5*R/(      C11*besseli(0,rtia)-(C11-C12)*besseli(1,rtia)/rtia );
    

    %below is the equation for pressure divided by 0.5*(C11-C12)
    press = real( 2*C11/(C11-C12)*B*( besseli(0,rtia*rp)-besseli(0,rtia) ).*exp(i*omega*times) );
    if (shouldDimensionalize) %if the result should be dimensionalized, then multiply by (C11-C12)/2
        press = press*(C11-C12)/2;
    end
    %}
    
    A= R/( 2*rtia*( C11*besseli(0,rtia)-(C11-C12)*besseli(1,rtia)/rtia ) ); %divided by eps0
    B= R/( 2     *( C11*besseli(0,rtia)-(C11-C12)*besseli(1,rtia)/rtia ) ); %divided by eps0
    fnc= C11/(0.5*(C11-C12)) * B * ( besseli(0, rtia*rp) - besseli(0, rtia) ); %divided by eps0
    press= abs(fnc) * cos(omega*times + angle(fnc) );
    if (shouldDimensionalize) %if the result should be dimensionalized, then multiply by (C11-C12)/2
        press = press*(C11-C12)/2;
    end
end
