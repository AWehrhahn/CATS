CRIRES+ Data Challenge
======================

The data is from the internal CRIRES+ data challenge (2020, version 4), which is based on simulated data, as real data is not available yet.

Principle
---------

The concept is the same as for all of CATS, create a model of the observation (stellar, tellurics, etc.) and than use an inverse problem approach to find the planet spectrum. Special care is taken, that the initial observations are altered as little as possible.

New in this iteration is the final step of the solution. Instead of just using the model, we actually solve the problem twice. But the second time around, the spectra are shifted the other way around. I.e. as if the planet was running backwards, with its negative radial velocity. We then use the difference between these two solutions as the extracted planet spectrum. The idea being that we remove residual signal this way, are only left with the planet transmission we were looking for.

Workflow
--------
Step 1:
    Collect the observations from the input data files and store them in our custom SpectrumArray objects

Step 2: Prepare tellurics
    Use the calculated airmass and a telluric model to get a guess of the telluric spectrum

Step 3: Determine stellar parameters
    Stellar parameters are determined using PySME. For this purpose all observations are first aligned (shifted) to the barycentric restframe and coadded into a single spectrum. We then manually have to add a mask to this first guess. Finally PySME can determine the stellar parameters. Unfortunately the observations are heavily impacted by the telluric spectrum, which means there are only a few lines, that are actually usuable.

Step 4: Prepare stellar models
    Based on the stellar paramerers we can calculate the best fit stellar model. This spectrum will be used in conjunction with the specific intensities from SME, to get the intensities used in the model. And also for the normalization of the observations.

Step 5: Create coadded (combined) stellar spectrum
    Coadd the observations to get a combined stellar spectrum. These will be used for the model.

Step 6: Normalize observations
    The observations are normalized, by comparing it to the "ideal" SME stellar * model telluric. We use the best fit polynomial so that observation * polynomial = stellar * telluric. Additional we allow additional broadening on the stellar and telluric spectrum. We repeat this process 3 times, to make sure we get the optimal values.

Step 7: Determine planet transit
    For the planet transit, we take the median of the unnormalized observations along the spectral axis, creating the lightcurve. To account for the tellurics, we divide by the same median of the telluric model.
    The lightcurve is then fitted using the batman model with initial parameters from the literature (exoplanets.org).

Step 8: Refine the telluric model
    The telluric model is improved by performing a linear fit of the spectrum versus the airmass at each spectral bin. However we fall back to the telluric model if the fit is too bad, since not all telluric features will be picked up.
    TODO: Use some other code (What was it called they use it in cross corellation)

Step 9: Prepare stellar intensities
    Calculate the specific intensities for each observation, using the determined planet orbit parameters and stellar parameters. We use several mu values around each point and then average, since the planet is more than just a point. This is especially important when the planet is entering or exiting the stellar disk.

Step 10: Extract Planet spectrum
    This is what we are here to do, get the planet spectrum out. First we have to do some more normalization.
    We again use the lightcurve (the median along the spectral axis) of both the observations and the model. We fit them with a 5th order polynomial each and use that as an additional factor that goes into the model. Second we have to fit the area of the planet, based on the specific intensities. For this we allow the specific intensity to vary with each observation. We smooth the solution with a small gaussian of width 1.
    This will all go into the model "g" and the stellar intensity blocked by the planet "f", the two components of the inverse problem f * x - g = 0, where x is the planet transmission.
    The wavelength is involved in the regularization. For this we determine the wavelength in the planet restframe, for each point in each observation, and put them all in one big array. We sort this, together with f and g, so that we effectively get (number of spectral points) x (number of observations) individual points. The regularization is the difference (similar to the first derivative) of the 8 neighbouring points, at each point. I.e. strong regularization punished variation with wavelength. The actual problem is solved by solving the explicit solution of the Tikhonov equation.
