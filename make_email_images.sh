#!/bin/sh

echo Saving email addresses

echo "Rahul Yerrabelli: \n ryerrab1@alumni.jh.edu \n rsy2@illinois.edu \n\nAlexander Spector: \n aspector@jhu.edu" | convert -background none -density 196 -resample 72 -unsharp 0x.5 -font "Courier" text:- -trim +repage -bordercolor white -border 0 email-address-image.gif

echo Done
