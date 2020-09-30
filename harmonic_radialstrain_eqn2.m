function radialstrain = harmonic_radialstrain_eqn2( times, v, E, r, a, alph, omega, tg, shouldDimensionalize)

    %BELOW IS A NUMERICAL WAY. IT ASSUMES THAT THE DISPLACEMENT EQUATION IS
    %CORRECT.
    delr = 0.0001;
    displ1 = harmonic_displ_eqn2( times, v, E, r-delr/2, a, alph, omega, tg, shouldDimensionalize);
    displ2 = harmonic_displ_eqn2( times, v, E, r+delr/2, a, alph, omega, tg, shouldDimensionalize);
    
    %Derivative of displacement with respect to r
    radialstrain = (displ2-displ1)/delr;
    return
     
end