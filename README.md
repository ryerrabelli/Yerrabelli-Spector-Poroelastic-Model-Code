# Yerrabelli-Spector-Poroelastic-Model-Code
 
**NOTE: For the most updated version of this document and all associated code and files, see: https://github.com/ryerrabelli/Yerrabelli-Spector-Poroelastic-Model-Code**

## Contact  
Code was created by [Rahul S. Yerrabelli](https://orcid.org/0000-0002-7670-9601)<sup>1,2</sup> in the lab of [Alexander A. Spector](https://orcid.org/0000-0003-0701-8185
)<sup>1</sup>.  
 1. [Johns Hopkins University, Department of Biomedical Engineering, Baltimore, MD, USA](https://www.bme.jhu.edu/)  
 1. [Carle Illinois College of Medicine, University of Illinois at Urbana-Champaign, Urbana, IL, USA](https://medicine.illinois.edu/).  

![Email addresses as an image to prevent spam](email-address-image.png "Email Addresses as Image")



## Background  
* The code here is what I used to create the analyses in our manuscript, which was accepted for publication on Nov 8, 2020 to [Medical & Biological Engineering & Computing (MBEC)](https://www.springer.com/journal/11517). The manuscript title is ["**IModeling the Mechanics of Fibrous-Porous Scaffolds for Skeletal Muscle Regeneration**" by Rahul S. Yerrabelli, Sarah M. Somers, Warren L. Grayson, and Alexander A. Spector.
* The outputs of the mathematical derivations are represented in these files and subsequently plotted.
* The modeling project spanned Sep 2017 - Nov 2020.  
* All code is in MATLAB®.




## Instructions for Understanding the Model's MATLAB® Code  
* Each of the major mechanical functions (specifically displacement, fluid velocity, pressure, and radial strain) are represented as separate .m files. Different files are also used for the parameters under ramped (increasing at a constant rate, then staying at a max) and harmonic (sinusoidal) strain conditions.
  * Ramped strain functions (Used in **Figure 3**, **Figure 4**, and **Figure 5** of the manuscript)
    * ___Displacement:___ **[ramped_displ_eqn.m](ramped_displ_eqn.m)**
    * ___Fluid velocity:___ **[ramped_fluidvel_eqn.m](ramped_fluidvel_eqn.m)**
    * ___Pressure:___ **[ramped_press_eqn.m](ramped_press_eqn.m)**
    * ___Radial strain:___ **[ramped_radialstrain_eqn.m](ramped_radialstrain_eqn.m)**
    * ___Solid velocity:___ A separate solid velocity (velocity) function was not created as it can just be calculated as the numerical derivative of displacement.
    * ___Relative velocity:___ **[ramped_relvel_eqn.m](ramped_relvel_eqn.m)** <- This file is not truly necessary as it can be calculated from fluid velocity and solid velocity (itself calculable from displacement).
    * ___Force:___ **[ramped_force_eqn.m](ramped_force_eqn.m)** <- The force function under ramped strain has already been well described for decades in prior literature, and thus was not a focus of our studies. It is included only for completion.
  * Harmonic functions (Used in **Figure 6** of the manuscript)
    * ___Displacement:___ **[harmonic_displ_eqn2.m](harmonic_displ_eqn2.m)**
    * ___Fluid velocity:___ **[harmonic_fluidvel_eqn2.m](harmonic_fluidvel_eqn2.m)**
    * ___Pressure:___ **[harmonic_press_eqn2.m](harmonic_press_eqn2.m)**
    * ___Radial strain:___ **[harmonic_radialstrain_eqn2.m](harmonic_radialstrain_eqn2.m)**
    * ___Solid velocity:___ A separate solid velocity (velocity) function was not created as it can just be calculated as the numerical derivative of displacement.
    * ___Relative velocity:___ A separate relative velocity function was not created as it can simply be calculated from fluid velocity - d/dt(displacement) as shown in main.mlx
    * ___Force:___ A force function under harmonic strain was not created.
* The **[main.mlx](main.mlx)** file runs the above equations under various conditions (time, radial position, etc) and plots them. Many of the analyses runs and followup plotting take several minutes to run.
* The **[main.mlx](main.mlx)** file also has the representation of the various strains we tested (represented in **Figure 2**).
* In our manuscript, we studied cyclic strains in both the non-harmonic and harmonic cases. Ramped strain of course is not cyclic in it of itself; it represents strain increases from 0 at a constant rate until it reaches a  max, at which point the strain stays at that high level. Thus, the ramped functions only represent one of the components of a non-harmonic cycle of strain. Each one can be run multiple times and added or subtracted (with a time offset) to reach the various non-harmonic cyclic strains represented in **Figure 2A-C** and the consequent non-harmonic cyclic functions represented in **Figure 3, Figure 4, and Figure 5** of the manuscript.
* In the **[main.mlx](main.mlx)** file, only 3 panels of can be cleanly produced at a time for Figures 3-6. For example, we did A, B, C as one output and D, E, and F as another output. We then combined the images in an image editor. The legends are also produced separately.


## Notes, Warnings, and Potential Sources of Confusion  
* The "2" at the end of the harmonic function file names (i.e. **[harmonic_displ_eqn2.m](harmonic_displ_eqn2.m)**) does not mean the output is squared. It was only named this way because it was the second version of the code we made. 
* In general, the outputs are made to be non-dimensional (i.e. by dividing by a reference unit such as radius length for the displacement output). Internally in the code, we tried to stick to SI units.  


## Tested OS and History Details  
* Code created and tested on MATLAB® 2017a for Mac (macOS Mojave, 2018 MacBook Pro 15in).


