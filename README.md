# GibbsNet
Implemented an image denoising algorithm using Gibbs sampling and Bayesian Networks to iteratively restore noisy grayscale images. Modeled each pixel’s value as a posterior distribution based on neighboring pixels, simulating over 50,000 iterations. Visualized the denoising effect using Matplotlib. 



    The program is suppose to denoise a noisy image using gibbs sampling. 
    I would like to inform the grader that most of the code has been adapted from the file 
    named: "image_denoising_gibbs_sampling.ipynb" with the Professor's permission. 
    
    The task is to create a noisy image where each pixel value is in the range of 0, 255 with mean 0 and standard 
    of 25
    The program estimates true value of a particular pixel at a given position using both its neighbouring pixels and a 
    prior belief. The prior belief follows a Gaussian distribution, assuming that pixel value is close to 0
    The likelihood is derived from the average of its neighbors with some noise. The noise has a std = 25
    The mean is the average of surrounding pixels

    To calculate the posterior mean and variance, the program uses bayes theorem.
    
Took some idea from here: https://web.stanford.edu/class/stats200/Lecture20.pdf - Section 20.1 Prior and Posterior Distributions
    
        p(x∣y)=  ( p(y∣x)⋅p(x) )/ p(y)
        p(x) is prior distribution based on the neighbors
        p(y) is observation so the actual pixel value
        p(y|x) is the likelihood
        and 
        p(x|y) is the posterior distribution.

    Since both prior and likelihood follow gaussian distrubution:
        
    1 / posterior variance = 1 / prior variance + 1 / noise varince
    This gives : 
    posterior variance = 1 / (1/prior variance + 1/ noise_variance) 

    posterior mean = ( prior mean (neighrbor mean) * prior variance + observed pixel * likelihood variance) / 
    (prior_variance + likelihood_variance)
    
    Using the posterior mean the new pixel value is calculated and updated in the image. 
