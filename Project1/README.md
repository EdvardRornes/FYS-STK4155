# TODO Project 1:

## Programming:

### Part a:

- [x] Perform OLS reg using polynomials in $x$ and $y$ up to fifth order. 
- [x] Evaluate MSE and $R^2$. Plot the MSE and $R^2$ as a function of the polynomial degree. 
- [x] Plot the parameters $\beta$ as you increase the order of the polynomials. Note: Code must include scaling/centering of the data and a split in training and test data. **ER: I think we need to reconsider how we do this point. Currently it is very hard to get any information from those plots, even at low degrees. The first few degrees are also completely impossible to view since higher polynomials are written above it. Perhaps reversing the plotting order will help?**

### Part b:

- [x] Write own code for Ridge regression.
- [x] Perform the same analysis as above but now for different values of $\lambda$.
- [x] Study the dependence on $\lambda$.

### Part c:

- [x] Write own code for LASSO regression. Use Scikit-learn. Scikit-learn excludes the intercept by default. **Not really sure what including/excluding intercept really means...**
- [x] Perform the same analysis as above but now for different values of $\lambda$.
- [x] Study the dependence on $\lambda$.

### Part d:

- [x] Show that $\mathbb{E}(y_i)=\boldsymbol X_{i,*}\boldsymbol\beta$, $\text{Var}(y_i)=\sigma^2$, $\mathbb{E}(\hat{\boldsymbol\beta})=\boldsymbol\beta$ and $\text{Var}(\hat{\boldsymbol\beta})=\sigma^2(\boldsymbol X^T\boldsymbol X)^{-1}$.

### Part e:

- [x] (Week 38) Show that $\mathbb{E}[(\boldsymbol y-\tilde{\boldsymbol y})^2]=\text{Bias}[\tilde y]+\text{Var}[\tilde y]+\sigma^2$ where $\text{Bias}[\tilde y]=\mathbb{E}\left[(\boldsymbol y-\mathbb E[\tilde{\boldsymbol y}])^2\right]$ and $\text{Var}[\tilde y]=\mathbb E\left[(\tilde{\boldsymbol y}-\mathbb E[\tilde{\boldsymbol y}])^2\right]=\frac 1n\sum_i (\tilde y_i-\mathbb E[\tilde{\boldsymbol y}])^2$
- [ ] Perform a bias-variance analysis of the Franke function by studying the MSE value as a function of the complexity of the model. Here we need to calculate the Bias and the Variance and plot these to perform an analysis.

### Part f:

- [x] Implement the $k$-fold cross-validation algorithm (using Scikit-Learn) and evaluate again the MSE function resulting from the test folds. 
- [ ] Make it so that we dont have a billion files for the above. Probably just pick the particular values of $\lambda$ which are reasonable from the above discussions.

### Part g:

- [x] Download data from https://earthexplorer.usgs.gov/. Register a user and decide where to fetch the data from. The format should be **SRTM Arc-Second Global** and download the data as a **GeoTIF** file. *tif* files can be imported into Python using **ER: Should probably add a simpler place as well, can exclude either Everest or Grand Canyon. Everest is the least interesting imo**
> scipy.misc.imread
- [x] Use OLS on the terrain data
- [x] Use Ridge on the terrain data
- [x] Use LASSO on the terrain data **ER: All of the above still need refining**


## Discussion

### Part a:

- [ ] Regarding β as a function of the polynomial degree: Comment on the results. 
- [ ] Present a critical discussion why and why not you have scaled the data. If we have then explain why. **ER: I have scaled it but other than it just being convenient I am not sure. The previous point also states that we have to do it??**

### Part b:

- [ ] Compare and analyze the results with those obtained in the previous part with OLS.

### Part c:

- [ ] Compare and analyze the results with those obtained in the previous part with OLS and Ridge.
- [ ] Give a critical discussion of the three methods and a judgement of which model fits the data best.

### Part d:

### Part e:
- [x] Discuss the bias and variance trade-off as a function of your model complexity (the degree of the polynomial), the number of data points, and possibly also the training and test data using the **bootstrap** resampling method. **ER: Discussion missing, needs refining.**

### Part f:

- [ ] Compare the MSE you get from your cross-validation code with the one you got from your bootstrap code. 
- [ ] Comment on your results. Try 5-10 folds. Do this with OLS, Ridge and LASSO regression.

### Part g:

- [ ] Evaluate which model fits the data best
- [ ] Present a critical evaluation of the results and discuss the applicability of these regression methods to the type of data presented here.


## Actual report TODO's

- [ ] Subsection "Connection to Statistics" has mistakes, look through it and fix things.
- [ ] Subsection "Bias-Variance" is currently written on the form of an exercise instead of a report. Fix language and the way things are presented.
- [ ] Section "Implementation" needs a lot of work. I have pretty much just written down the basics. We need to choose (and reason) the train-test split and keep it throughout. We also need to comment on the intercept part regarding LASSO.
- [ ] All figures need titles which describe them in more detail. Captions are also necessary.
- [ ] Bias-Variance proof in Appendix A is very scuffed. I completely ignored vector/index notation and just brute forced everything as if they were commuting scalars. Do this proof properly.
- [ ] Write results.
- [ ] Write conclusion.
- [ ] Finalize introduction.
- [ ] Finalize abstract.