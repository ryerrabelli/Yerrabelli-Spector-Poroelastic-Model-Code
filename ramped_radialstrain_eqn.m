% Made by Rahul Yerrabelli for Dr. Alexander Spector's lab at Johns Hopkins
% University, Department of Biomedical Engineering. 2017-2020.
% To contact authors, reach out to aspector@jhu.edu for Dr. Spector and
% ryerrab1@alumni.jh.edu or ryerrabelli@gmail.com for Rahul.

function radialstrain = ramped_radialstrain_eqn(time, v, E, r, a, t0_tg, tg, analytically)

    %BELOW IS A NUMERICAL WAY. IT ASSUMES THAT THE DISPLACEMENT EQUATION IS
    %CORRECT.
    
    delr = 0.0001;
    displ1 = ramped_displ_eqn(time, v, E, r-delr/2, a, t0_tg, tg);
    displ2 = ramped_displ_eqn(time, v, E, r+delr/2, a, t0_tg, tg);
    
    %Derivative of displacement with respect to r
    radialstrain = (displ2-displ1)/delr;
    return;
    

end

