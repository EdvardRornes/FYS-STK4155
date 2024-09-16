## TODO Project 1:


### Part a:

- [ ] Perform OLS reg using polynomials in $x$ and $y$ up to fifth order. 
- [ ] Evaluate MSE and $R^2$. Plot the MSE and $R^2$ as a function of the polynomial degree. 
- [ ] Plot the parameters $\beta$ as you increase the order of the polynomials. Comment on the results. Note: Code must include scaling/centering of the data and a split in training and test data. 
- [ ] **Present a critical discussion why and why not you have scaled the data. If we have then explain why.**

### Part b:

- [ ] Write own code for Ridge regression.
- [ ] Perform the same analysis as above but now for different values of $\lambda$.
- [ ] Compare and analyze the results with those obtained in the previous part with OLS.
- [ ] Study the dependence on $\lambda$.

### Part c:

- [ ] Write own code for Ridge regression. Use Scikit-learn. 
- [ ] Perform the same analysis as above but now for different values of $\lambda$.
- [ ] Compare and analyze the results with those obtained in the previous part with OLS and Ridge.
- [ ] Study the dependence on $\lambda$.
- [ ] Scikit-learn excludes the intercept by default. Give a critical deiscussion of the three methods and a judgement of which model fits the data best.

### Part d:

- [x] Show that $\mathbb{E}(y_i)=\boldsymbol X_{i,*}\boldsymbol\beta$, $\text{Var}(y_i)=\sigma^2$, $\mathbb{E}(\hat{\boldsymbol\beta})=\boldsymbol\beta$ and $\text{Var}(\hat{\boldsymbol\beta})=\sigma^2(\boldsymbol X^T\boldsymbol X)^{-1}$.

### Part e:

- [ ] (Week 38) Show that $\mathbb{E}[(\boldsymbol y-\tilde{\boldsymbol y})^2]=\text{Bias}[\tilde y]+\text{Var}[\tilde y]+\sigma^2$, where $\text{Bias}[\tilde y]=\mathbb{E}\left[(\boldsymbol y-\mathbb E[\tilde{\boldsymbol y}])^2\right]$ and $\text{Var}[\tilde y]=\mathbb E\left[(\tilde{\boldsymbol y}-\mathbb E[\tilde{\boldsymbol y}])^2\right]=\frac1n\sum_i (\tilde y_i-\mathbb E[\tilde{\boldsymbol y}])^2$. 
- [ ] Perform a bias-variance analysis of the Franke function by studying the MSE value as a function of the complexity of the model. 
- [ ] Discuss the bias and variance trade-off as a function of your model complexity (the degree of the polynomial), the number of data points, and possibly also the training and test data using the **bootstrap** resampling method.

### Part f:

- [ ] Implement the $k$-fold cross-validation algorithm (using Scikit-Learn) and evaluate again the MSE function resulting from the test folds. 
- [ ] Compare the MSE you get from your cross-validation code with the one you got from your bootstrap code. 
- [ ]Comment on your results. Try 5-10 folds. Do this with OLS, Ridge and LASSO regression.

### Part g:

- [ ] Download data from https://earthexplorer.usgs.gov/. Register a user and decide where to fetch the data from. The format should be **SRTM Arc-Second Global** and download the data as a **GeoTIF** file. *tif* files can be imported into Python using
> scipy.misc.imread
- [ ] Use OLS on the terrain data
- [ ] Use Ridge on the terrain data
- [ ] Use LASSO on the terrain data
- [ ] Evaluate which model fits the data best
- [ ] Present a critical evaluation of the results and discuss the applicability of these regression methods to the type of data presented here.