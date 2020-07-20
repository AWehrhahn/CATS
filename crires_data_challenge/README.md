CRIRES+ Data Challenge
======================

The data is from the internal CRIRES+ data challenge (2020, version 4), which is based on simulated data, as real data is not available yet.

Principle
---------

The concept is the same as for all of CATS, create a model of the observation (stellar, tellurics, etc.) and than use an inverse problem approach to find the planet spectrum. Special care is taken, that the initial observations are altered as little as possible.

New in this iteration is the final step of the solution. Instead of just using the model, we actually solve the problem twice. But the second time around, the spectra are shifted the other way around. I.e. as if the planet was running backwards, with its negative radial velocity. We then use the difference between these two solutions as the extracted planet spectrum. The idea being that we remove residual signal this way, are only left with the planet transmission we were looking for.