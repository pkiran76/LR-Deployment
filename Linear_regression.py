"""
1-Linear Regression-y=mx+c
2-m-slope/tan@/tangent/(dy/dx)/rate of change of y w.r.t. x-coefficient/proportionality constant
2.1-y=0.5x=>for one unit change in x, there will be 0.5 units change in y
3-c-intercept-value of y when x=0=>even when the value of x=0,y is getting affected by c units
4-How to pick the line of best fit among infinite number of lines?-the best value of m and c which can generalise the eqn.
4.1-The line giving the least difference/average error b/n the EV and the PV
4.2-Automated way of determining the optimal values of the parameters(where error=0)-Learning
4.3-Aim: to achieve->de/dm=0,de/dc=0->foundation of gradient-descent
5-When do i need to a LR??
5.1-Assumptions-
    1-Linear r/nship b/n y and X
    Check: Scatter plot
    2-Mean of residuals=0-why?-It implies that the average deviation of each error from the regression line is zero,
      implying that the average of a the residuals for all the points above the regression line and for that for all
      the points below the regression line would be the same...hence summation would be zero.
      Check-The plot of IV vs residuals or the fitted values vs residuals must be symmetric about zero
    3-Error terms are not supposed to be correlated with each other-no auto-correlation
      else it means the error are holding some of the info about the data within them
      In other words when the value of y(x+1) is not independent from the value of y(x).
      if you are attempting to model a simple linear relationship but the observed relationship is non-linear
      (i.e., it follows a curved or U-shaped function), then the residuals will be autocorrelated.
      Check: Durbin-Watson test<4->0 implies highly +ve correlation and 4 implies highly -ve correlation
    4-IVs and residuals must not be correlated-exogenity-Exogeneity means that each X variable
      does not depend on the dependent variable Y; rather Y depends on the Xs and on ɛ, the model error.
      Since Y depends on ɛ,this means that the Xs are assumed to be independent of Y hence error.
      All OLS estimators will be biased and inconsistent in the presence of endogenous regressors.
      Endogeneity can arise as a result of measurement error, reverse casualty/simultaneity, omitted variable
      or unobserved variables, omitted selection,lagged dependent variables.
      Check:Housman‟s test (also known as Housman Specification test or Durbin, Hausman and Wu Test)
      you are effectively comparing the OLS estimate of your parameter on VariableOne to the 2SLS estimate
      of the same. If these two estimates differ, then you should believe that the variable is endogenous.
      If they are the same (in a statistical sense), then you should usually use OLS.
      Use from linearmodels import IV2SLS, IVLIML, IVGMM, IVGMMCUE
      :How to repair:If endogenous, use IV/2SLS approach;IV-Instrumental Variable/2 stage Least Sqyares Regression
      IV/2SLS-X(hat)=lambda*X+error-we find another varaible z which acts as an instrument and is correlated with X but not with the error term
       Estimate X using Z and then use this estimate of X to do another regression-y=beta*X(hat)+U

    5-Error term must show a constant variance-homoscedasticity-constant variance in errors
      Heteroscedasticity implies presence of outliers/extreme values and non-identical PD
      No pattern in error distribution or the error distribution must be random..
      If the pattern is random, then it points to NL of data-do transformation of the IV
      If the pattern is of a funnel shape,do transformation of the DV
      Check:Scatter plot b/n residuals and PV,
      Goldfeld-Quandt test-It compares variances of two subgroups; one set of high values and one set of low values.
      If the variances differ, the test rejects the null hypothesis that the variances of the errors are not
       constant-statsmodels.stats.diagnostic.het_goldfeldquandt-The Null hypothesis is that the variance in the two sub-samples are the same.
    6-No multi-collinearity-r/ship b/n IVs
      A key goal of regression analysis is to isolate the relationship between each independent variable and the
      dependent variable. The interpretation of a regression coefficient is that it represents the mean change
      in the dependent variable for each 1 unit change in an independent variable when you hold all of the
      other independent variables constant.That last portion is crucial for our discussion about multicollinearity.
      The idea is that you can change the value of one independent variable and not the others.
      The stronger the correlation, the more difficult it is to change one variable without changing another.
      Multicollinearity affects the coefficients and p-values, but it does not influence the predictions,
      precision of the predictions, and the goodness-of-fit statistics.
      If your primary goal is to make predictions, and you don’t need to understand the role of each independent
      variable,you don’t need to reduce severe multicollinearity.
      Check:Correlation coeff M'.,VIF (A value of 1 indicates that there is no correlation between this
      independent variable and any others. VIFs between 1 and 5 suggest that there is a moderate correlation,
       but it is not severe enough to warrant corrective measures.VIFs greater than 5 represent critical levels of multicollinearity)
       How to remove??-Center the Independent Variables-This process involves calculating the mean for each
       continuous independent variable and then subtracting the mean from all observed values of that variable or
       remove the highly correlated variables,Lasso and Ridge Regression,linearly combine them
    7-Error distribution must be normal-Usually, there are 2 reasons why this issue(error does not follow a normal
      distribution) would occur:
      Dependent or independent variables are too non-normal(can see from skewness or kurtosis of the variable)
      Existence of a few outliers/extreme values which disrupt the model prediction, remove them or do a box-cox
      transformation of the non-normal variables.
      It means that a majority of the errors are close to zero or very close to zero.
      Check:Shapiro-Wilk test,Q-Q plot,kolmogonov-smirnov test, Jarque-Bera test,Anderson-Darling test

6-Residual??-r=y-yhat=y-(mX+c)
7-TO make r independent of the direction, use r2
8-For all the data points, the residual is aggregate as sigma(r2)
9-Goal-dr/dm=dr/dc=0
9.1->r2=R==sigma(y-(mX+c))^2=y2+m2X2+c2+2mXc-2ymX-2yc
9.2->dR/dm->dr/dm=0+2*mX2+0+2Xc-2yX-0=>dr/dm=2mX2+2Xc-2yX=>2X(m2+c-y)=0->this is for a single sample
9.3->dR/dc=Ec->2(c+mX-y)=0->this is for a single sample,for the entire dataset, sigma dR/dc and sigma dR/dm=0
10-m(new)=m(old)-alpha* 1/m * (sigma(y-yhat))=>m(new)=m(old)-alpha * Em (Em=>dR/dm=0-value of e or r=0)
11-c(new)=c(old)-alpha * Ec (Em=>dR/dc=0)
12-alpha=learning rate-will control the rate of change-hyperparameter->10 to 0.00001->ideal one->0.001
13-Gradient descent is an iterative optimization algorithm for finding the local minimum of a function.
14-To find the local minimum of a function using gradient descent, we must take steps proportional to the
   negative of the gradient (move away from the gradient) of the function at the current point.
15-Gradient descent was originally proposed by CAUCHY in 1847. It is also known as steepest descent.
16-It's based on a convex function and tweaks its parameters iteratively to minimize a given function
   to its local minimum.
17-Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function.
   Gradient descent is simply used in machine learning to find the values of a function's parameters (coefficients)
   that minimize a cost function as far as possible.
18-A gradient measures how much the output of a function changes if you change the inputs a little bit.
19-You can also think of a gradient as the slope of a function. The higher the gradient, the steeper the slope
   and the faster a model can learn. But if the slope is zero, the model stops learning.
   In mathematical terms, a gradient is a partial derivative with respect to its inputs.
20-the gradient simply measures the change in all weights with regard to the change in error.
21-A Cost Function/Loss Function evaluates the performance of our Machine Learning Algorithm.
22-The Loss function computes the error for a single training example, while the Cost function is the
   average of the loss functions for all the training examples.
23-The gradient (or derivative) tells us the incline or slope of the cost function.
   Hence, to minimize the cost function, we move in the direction opposite to the gradient.
23.1-Essentially, there are two things that you should know to reach the minima, i.e. which way to go and how
   big a step to take.
23.2-A derivative is a term that comes from calculus and is calculated as the slope of the graph at a particular
     point. The slope is described by drawing a tangent line to the graph at the point.
     So, if we can compute this tangent line, we might compute the desired direction to reach the minima.

24.1-Decide on the Loss function to use(ex:MSE) to evaluate how well a line fits a data
24.2-Then take the derivative of the loss function w.r.t each of the parameters by using the chain rule or take the
     gradient of the Loss Function.
24.3-Compute the sigma of the derivative for a particular line by taking
     the IV and the DV value for each point and computing the individual derivative
24.4-Take one random value each initially of m & c and compute the respective gradients/slopes.
24.5-Compute the step sizes for each of the parameters.
     -step size=LR(alpha) * slope(gradient)->as the slope approaches zero, the step size approaches zero(optimal soln)
24.6- Then use these slopes(gradients) and the LR to compute the new parameter values.
      -new parameter=old parameter-step size
      -The max. no. of steps>=1000
      -Gradient-When we have 2 or more derivatives of the same function
      -This gradient is used to descend to the lowest point in the Loss Function-hence Gradient Descent
24.7-Repeat the above two steps till the max. no. of steps specified are reached or the step size becomes very small.
24.8-If there are more parameters, then we would just take more derivatives and everything else would remain the same..


Accuracy=R2-Coef. of determination=0 to 1=SSR/SST=1-SSE/SST=(S(mean)-SS(fit))/SS(mean)
SSR-measure of explained variation;SSE-measure of un-explained variation;SST-measure of total variation
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg', force=True)
import pickle
from pandas_profiling import ProfileReport
from pandas.plotting import scatter_matrix
import seaborn as sns
plt.switch_backend('TkAgg')

"""
df=pd.read_csv("D:\Data_Science\General_Books\Interview\Linear_regression\Advertising.csv")
df.drop(columns=["Unnamed: 0"],axis=1,inplace=True)
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())

# prof = ProfileReport(df)
# prof.to_file(output_file='output.html')

X=df[["TV"]]
y=df[["Sales"]]

from sklearn.linear_model import LinearRegression
linear=LinearRegression() #set object of the class LinearRegression
linear.fit(X,y) #object of the class (LinearRegression) is fit on the data and model is done=> m and c are computed
print(linear.intercept_) #[7.03259355]-c
print(linear.coef_) #[[0.04753664]]-m  #model is nothing but the line-y=mx+c with optimised values of m and c
y_pred=linear.predict(X)
#Accuracy
print(linear.score(X,y)) #0.611875050850071-same as R2
from sklearn.metrics import r2_score
print(r2_score(y,y_pred)) #0.611875050850071
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y,y_pred)) #10.512652915656757
# Save the model as a pickle file
# pd.to_pickle(linear,"adv.pkl")
# pd.read_pickle("adv.pkl")
"""


# import statsmodels.api as sm
# print(sm.__version__) #0.13.2
# y=sm.add_constant(y,prepend=False) #why do we need to add a constant??
"""
#By default, statsmodels fits a line passing through the origin, i.e. it doesn't fit an intercept. 
#Hence, you need to use the command 'add_constant' so that it also fits an intercept.
#It doesn't add a constant to your values, it adds a constant term to the linear equation it is fitting. 
# In the single-predictor case, it's the difference between fitting an a line y = mx to your data vs fitting y = mx + b.
"""
# mod=sm.OLS(X,y)
# res=mod.fit()
# print(res.summary())

"""
# MLR
df1=pd.read_csv("D:\Data_Science\General_Books\Interview\Linear_regression\Ai4i2020.csv")
# df1.drop(columns=["Unnamed: 0"],axis=1,inplace=True)

print(df1.head())
print(df1.shape)
print(df1.describe())
print(df1.info())
print(df1.isnull().sum())

# prof = ProfileReport(df1,infer_dtypes=False)
# prof.to_file(output_file='output_ML.html')
y=df1[["Air temperature [K]"]]
X=df1[["Process temperature [K]"]]
"""

"""
from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(X,y)
print(linear.intercept_)
print(linear.coef_)
y_pred=linear.predict(X)
#Accuracy
print(linear.score(X,y)) #0.767563752503258-same as R2
from sklearn.metrics import r2_score
print(r2_score(y,y_pred)) #0.767563752503258
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y,y_pred)) #0.9298925161343161

# Save the model as a pickle file
# pd.to_pickle(linear,"adv.pkl")
# pd.read_pickle("adv.pkl")
X1=df1.drop(columns=["UDI","Product ID"],axis=1)
print(X1.info())
print(type(X1))
ohe=pd.get_dummies(X1,columns=["Type"])
# ord(ohe["Type_H"].decode("utf-8"))
print(ohe.info())
print(type(ohe))
print(ohe.head())
X2=ohe.iloc[:,1:]
print(X2)
y2=ohe.iloc[:,0]
print(y2)
# X2=df1[["Air temperature [K]","Rotational speed [rpm]",""]]
import statsmodels.api as sm
X2=sm.add_constant(X2,prepend=False)
model=sm.OLS(y2,X2)
result=model.fit()
print(result.summary())

#Assumptions Validation
#1-Linear r/nship b/n y and X
# pd=sm.graphics.plot_partregress_grid(result)
# pd.tight_layout()
# plt.show()
# scatter_matrix(df1)
# plt.show()
# see pandas_profiling-best answer

#2-Mean of residuals=0
print(result.resid.mean()) #-7.281005309778265e-12

#3-Error terms are not supposed to be correlated with each other-no auto-correlation
# 3.1-Durbin-Watson: 0.074
#3.2-ACF plot
# import statsmodels.tsa.api as smt
# acf = smt.graphics.plot_acf(result.resid, lags=40 , alpha=0.05)
# plt.show()

#4-Exogenity

# from linearmodels import IV2SLS, IVLIML, IVGMM, IVGMMCUE
# from linearmodels.iv import IV2SLS
# ivmod=IV2SLS(y2,X2,None,None)
# res_2sls = ivmod.fit()
# print(res_2sls.summary)
# exog-IVs
# endog-DV
#Independent
"""
"""
Hausman-Test: In simple termns, the Hausman-Test is a test of endogeneity. By running the Hausman-Test, 
the null hypothesis is that the covariance between IV(s) and alpha is zero. If this is the case, 
then RE is preferred over FE. If the null hypothesis is not true, we must go with the FE-model.
RE-Random-Effects
FE-Fixed-Effects
"""

"""
# y=df1["Air temperature [K]"]
# N=len(y)
# exogenousX=df1[[]]
# Predicted values
y_pred=result.predict(X2)
print(y_pred.head())


#Step1:IVs vs Residuals plots
# X3=df1.drop(columns=["UDI","Product ID","Type"],axis=1)
# print(X3.columns)
# for i in X3.columns:
#     sns.residplot(result.resid, X3[i], lowess=True, line_kws={'color': 'r', 'lw': 1})
#     plt.title('Residual plot')
#     plt.xlabel(i)
#     plt.ylabel('Residuals')
#     plt.show()

X3=X1.drop(columns=["Type"],axis=1)
print(X3.columns)
X3.columns=['Airtemperature', 'Processtemperature',
       'Rotationalspeed', 'Torque', 'Toolwear',
       'Machinefailure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
print(X3.info())

from linearmodels.iv import IV2SLS

# Step2:To identify the exogenous, endogenous and instrument variables
# y=X3["Airtemperature"]
# N=len(y)
# exogenousX=X3[["Rotationalspeed",'Torque',"Toolwear","TWF","HDF","PWF","OSF","RNF"]].values
# endogenoeus1=X3["Processtemperature"].values.reshape(-1,1)
# instrument1=X3[["Machinefailure"]].values.reshape(-1,1)

#Step3:Compute the OLS using the linear models module IV2SLS by giving only the DV and the IV
formula1='Airtemperature~1+Torque+Toolwear+TWF+HDF+PWF+OSF+RNF+Processtemperature+Machinefailure+Rotationalspeed'
mod1=IV2SLS.from_formula(formula1,X3).fit(cov_type="unadjusted")
print(mod1.summary)

# Step4:Compute the IV/2SLS using the linear models module IV2SLS by giving the DV and the exo,endo and Instrumental variables
# Method1:Using IV2SLS
formula='Airtemperature~1+Torque+Toolwear+TWF+HDF+PWF+OSF+RNF+[Processtemperature~Machinefailure+Rotationalspeed]'
mod=IV2SLS.from_formula(formula,X3).fit(cov_type="unadjusted")
print(mod.summary)
print(mod.wooldridge_regression)
print(mod.wu_hausman())

#Method2: Using statsmodels
# from statsmodels.api import add_constant
# exogenousX=add_constant(exogenousX,has_constant="add")
# mod=IV2SLS(dependent=y,exog=exogenousX,endog=endogenoeus1,instruments=instrument1).fit(cov_type="unadjusted")
# print(mod.summary)

#Step5: Check for the Wooldridge and WU-Harman coeff. hypothesis by looking at the p-value

Wooldridge's regression test of exogeneity
H0: Endogenous variables are exogenous
Statistic: 1.8434
P-value: 0.1745
Distributed: chi2(1)
Wu-Hausman test of exogeneity
H0: All endogenous variables are exogenous
Statistic: 2.6069
P-value: 0.1064
Distributed: F(1,9990)

"""

"""
Procedure for Exogenity:
Step1: plot the residuals vs IV and check for any correlation/pattern in the behaviour.
Step2:To identify the exogenous, endogenous and instrument variables.
Step2.1:Identify the suspected IVs having such correlations and label them as exogenous variables.
Step2.2:Identify the exogenous variables, i.e.,IVs having little or no correlation with the residuals
Step2.3:Identify the Instrumental variables,i.e., variables having strong correlation with th endogenous variable
    but very little correlation with the DV
Step3:Compute the OLS using the linear models module IV2SLS by giving only the DV and the IV.
Step4:Compute the IV/2SLS using the linear models module IV2SLS by giving the DV and the exo,endo and Instrumental variables 
Step5:Check for the WU-Harman coeff. hypothesis by looking at the p-value 
"""

"""
# Regression plot
# Plot predicted values

fix, ax = plt.subplots()
ax.scatter(X2['Process temperature [K]'],y_pred, alpha=0.5,
        label='predicted')

# Plot observed values

ax.scatter(X2['Process temperature [K]'], y2, alpha=0.5,
        label='observed')

ax.legend()
ax.set_title('OLS predicted values')
ax.set_xlabel('Process temperature [K]')
ax.set_ylabel('Air temperature [K]')
plt.show()

"""

"""
#5-Error term must show a constant variance-homoscedasticity
fig, ax = plt.subplots()
ax.scatter(y_pred, result.resid, color = 'blue')
ax.axhline(0, color = 'r', ls = '--')
ax.set_xlabel('Predicted Values', fontsize = 15)
ax.set_ylabel('Residuals', fontsize = 15)
ax.set_title('Homoscedasticity Test', fontsize = 30)
plt.show()

#White and the Breusch-Pagan-Test
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
#White-Test
pooled_OLS_dataset = pd.concat([X2, result.resid], axis=1)
pooled_OLS_dataset.rename(columns={0:"residual"},inplace=True)
# print(pooled_OLS_dataset.head())
exog = sm.tools.tools.add_constant(X2['Rotational speed [rpm]']).fillna(0)
white_test_results = het_white(pooled_OLS_dataset['residual'], exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val']
print(dict(zip(labels, white_test_results)))
# if p < 0.05, then heteroskedasticity is indicated
#{'LM-Stat': 39.073558002308275, 'LM p-val': 3.2755534127119096e-09, 'F-Stat': 19.607531569660566, 'F p-val': 3.17113972778943e-09}

#Breusch-Pagan-Test
breusch_pagan_test_results = het_breuschpagan(pooled_OLS_dataset['residual'], exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val']
print(dict(zip(labels, breusch_pagan_test_results)))
# if p < 0.05, then heteroskedasticity is indicated
# {'LM-Stat': 19.294443498639737, 'LM p-val': 1.1203190578308733e-05, 'F-Stat': 19.32787667237993, 'F p-val': 1.1122454846798788e-05}


# 6-No multi-collinearity-r/ship b/n IVs
#Method 1:VIF-to check for correlation b/n IVs
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif["features"] = X2.columns
print(vif)
"""
# VIF=1/(1-R2)

#Method 2:Cond.No. in sm.OLS output
"""
The condition number (abbreviated “Cond. No.” in the summary) is a measure of “how close to singular” 
a matrix is; the higher, the “more singular” (and infinite means singular — i.e. noninvertible),
 and the more “error” a best fit approximation is.
Higher cond.no. implies multi-collinearity
"""

"""
# 7-Error distribution must be normal
#Method 1-Histogram
plt.hist(result.resid_pearson)
plt.show()
#Method 2-QQ plot
import pylab
import scipy.stats as st
st.probplot(result.resid_pearson,dist="norm",plot=pylab)
plt.show()

#Method 3-Jarque-Bera Test
# From sm.OLS result or
print(st.jarque_bera(result.resid))
# Jarque_beraResult(statistic=240.5289761976904, pvalue=0.0)
#H0=the distribution is normal

#Method 4-Shapiro-Wilcox Test
print(st.shapiro(result.resid))
# ShapiroResult(statistic=0.9805911779403687, pvalue=9.887019847862652e-35)
# Ho(Accepted): Sample is from the normal distributions.(pvalue>0.05)

# Check for outliers in the data using Cook's Distance
from statsmodels.stats.outliers_influence import OLSInfluence as influence
inf=influence(result)
(c, p) = inf.cooks_distance
plt.figure(figsize=(8,5))
plt.title("Cook's distance plot for the residuals",fontsize=16)
plt.stem(np.arange(len(c)), c, markerfmt=",")
plt.grid(True)
plt.show()
#SUMMARY OF OLS_INFUENCE TO GIVE INFO ON INFLUENCERS
# infl=result.get_influence()
# print(infl.summary_frame())

"""
"""
1-LR is a high bias low variance kind of a model
2-Bias-training error, variance-dispersion of the regression line
3-A high bias as the estimate is based on a single line
4-Low variance, as even further addition of data points hardly alters the regression line slope/intercept
5-On test data in LR, the results will have a not so great accuracy, but tends to be more reliable
6-A linear algorithm often has high bias, which makes them learn fast. In linear regression analysis,
  bias refers to the error that is introduced by approximating a real-life problem, which may be complicated,
  by a much simpler model. Though the linear algorithm can introduce bias, it also makes their output easier
  to understand. The simpler the algorithm, the more bias it has likely introduced. 
7-In contrast, nonlinear algorithms often have low bias.
8-Variance indicates how much the estimate of the target function will alter if different training data 
  were used. Variance is based on a single training set. Variance measures the inconsistency of different 
  predictions using different training sets — it’s not a measure of overall accuracy.
  It measures how scattered (inconsistent) are the predicted values from the correct value due to different
  training data sets.
9-Bias can lead to underfitting, while variance can lead to overfitting.
10-Machine learning algorithms with low variance include linear regression, logistics regression, and linear discriminant analysis.
   Those with high variance include decision trees, support vector machines and k-nearest neighbors.
11-Error = Reducible Error + Irreducible Error
12-Reducible Error = Bias² + Variance
13-Overfitting: It is a Low Bias and High Variance model. Generally, Decision trees are prone to Overfitting.
14-Underfitting: It is a High Bias and Low Variance model. Generally, Linear and Logistic regressions are prone to Underfitting.

15-
A-High Bias Low Variance-The data points are far from the target but are bunched together
B-Low Bias High Variance-The data points are close to the target but are spread apart (or all over the place)
C-High Bias High Variance-The data points are far from the target and are spread apart (or all over the place)
D-Low Bias Low Variance-The data points are close to the target and are bunched together

16-problems associated with different Bias - Variance combinations?
A-High Bias - Low Variance (Underfitting): Predictions are consistent, but inaccurate on average. 
  This can happen when the model uses very few parameters.
B-Low Bias - High Variance (Overfitting): Predictions are inconsistent and accurate on average. 
  This can happen when the model uses a large number of parameters.
C-High Bias - High Variance: Predictions are inconsistent and inaccurate on average.
D-Low Bias - Low Variance: It is an ideal model. But, we cannot achieve this.

17-How to identify High Variance or High Bias?
17.1-High Variance can be identified when we have:
-Low training error (lower than acceptable test error)
-High test error (higher than acceptable test error)
17.2-High Bias can be identified when we have:
-High training error (higher than acceptable test error)
-Test error is almost same as training error

18-How to address High Variance or High Bias?
18.1-High Variance is due to a model that tries to fit most of the training dataset points making it complex. Consider the following to reduce High Variance:
-Reduce input features(because you are overfitting)
-Use less complex model
-Include more training data
-Increase Regularization term
18.2-High Bias is due to a simple model. Consider the following to reduce High Bias:
-Use more complex model (Ex: add polynomial features)
-Increase input features
-Decrease Regularization term

19-What is Bias-Variance Trade-Off?
"""

# Adjusted R2:
"""
statsmodels can only be used for getting the statistics and understanding the data
whereas the scikit-learn models can be productionised as they can be saved as pickle files...
p-value-out of 100 experiments, how many have failed
ex: if the p value is 0.86, it means out of 100 trials,86 have failed and hence the variable will not be significant

Adjusted R2=1-((1-R2)*(N-1)/(N-P-1))
N=No. of data points/obsvns.,P-No.of IVs
The adjusted R-squared compensates for the addition of variables and only increases if the new predictor 
enhances the model above what would be obtained by probability. 
Conversely, it will decrease when a predictor improves the model less than what is predicted by chance.
In multiple linear regression, the R-squared can not tell us which regression variable is more important than the other.
"""

# REGULARIZATION
"""
It is method for controlling the error terms so that one can generalise a model in the best possible way-
controlling the d.o.f. of m and c by controlling the error term
3 types-LASSO,RIDGE,ElasticNet
1-LASSO/L1-Least Absolute Shrinkage and Selection Operator
       L1 Regularization=SSE+lambda*sigma(|beta j|);lambda-Shrinkage factor;beta=m
       -The above equation ensures that there is a factor which will control the m(new) and not allow it to be 
       changed drastically
       -Consider the previous changes and change the new values as per the previous ratio-why?
       -As otherwise, if there are large errors, large changes are made and if there are small errors, small
        changes are made and this behaviour needs to be avoided
       -Too wild a fluctuations in the changes to the values of m and c need to be avoided
       -L1 can shrink the parameter all the way to zero
       -L1 will perform better if there a lot of useless/statistically insignificant variables
       -L1 picks only one of the correlated variables and eliminates the other

2-RIDGE/L2-
       L2 Regularization=SSE+lambda*sigma(beta j)^2;lambda-Shrinkage factor;beta=m
       -L2 can shrink the parameter asymptotically to zero
       -L2 will perform better if there a lot of useful variables
       -L2 shrinks all of the parameters of the correlated variables together
       -L2 introduces a small amount of bias to the model which would in-turn reduce the variance
       -L2 tries to penalise  an overfitting model by introducing bias thereby reducing the slope and variance

       
-When to use what?L1 or L2??
-When m<1, use L1 as this gives greater impact/a better step as compared to L2 where the step size would be very small
 ex:m=0.1->L1=0.1*lambda,L2=0.01*lambda
-When m>1, use L2 as this gives greater control/impact/a better step as compared to L2 where the step size 
  would be very small
 ex:m=10->L1=10*lambda,L2=100*lambda

3-ElasticNet-Brought in to penalise irrespective of the magnitude of m
  -Takes the best of both worlds
  -(SSE/2n)+lambda*((1-alpha))/2 *sigma(Beta)^2+alpha*sigma(|beta j|); alpha-mixing parameter
  -If there are a lot of variables  and very little info is known about them
  -Preferred if ther is correlation b/n parameters
  -ENR groups and shrinks all of the parameters of the correlated variables together and then either drops
   or retains them all at once
"""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,Ridge,ElasticNet,LassoCV,RidgeCV,ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas_profiling import ProfileReport

print(os.getcwd())
df=pd.read_csv("D:\Data_Science\General_Books\Interview\Linear_regression\Admission_Prediction.csv")
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())

# prof = ProfileReport(df,infer_dtypes=False)
# prof.to_file(output_file='output_Admission.html')
# print(df.corr())

# Handling Missing values
df["GRE Score"]=df["GRE Score"].fillna(df["GRE Score"].mean())
df["TOEFL Score"]=df["TOEFL Score"].fillna(df["TOEFL Score"].mean())
df["University Rating"]=df["University Rating"].fillna(df["University Rating"].mean())
print(df.describe())
print(df.info())
print(df.isnull().sum())

# Drop unwanted column
df=df.drop(columns=["Serial No."],axis=1)
print(df.describe())
print(df.info())
print(df.isnull().sum())

# Create X and y
X=df.iloc[:,:-1]
print(X.head())
y=df.iloc[:,-1]
print(y.head())

# Normalisation/standardisation
# Why?-
# If the data dispersion has a lot of variability, the model will not be understand/interpret the relation
# b/n IV and the DVto lower the level/range/scale of the attribute values without modifying the meaning/content thereby enabling
# the model to understand and interpret the data and build the model in a better way
# Also, features with large order of variances will tend to dominate over other lower variance range features
# and model will tend to be biased towards them while ignoring the other features
# SVM and L1,L2 regularisers also assume that the distribution is normal, centered around zero and having a standard variance.
# Standardscaler-Z-score

scaler=StandardScaler()
arr=scaler.fit_transform(X) #fit_transform-function of the StandardScaler class
print(arr)
df1=pd.DataFrame(arr)
print(df1)
df1.columns=X.columns
print(df1.columns)


# prof = ProfileReport(df1,infer_dtypes=False)
# prof.to_file(output_file='output_Admission_scaler.html')
print(df1.describe())

# Check Multi-Collinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
# variance_inflation_factor(())
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df1.values, i) for i in range(df1.shape[1])]
vif["features"] = df1.columns
print(vif)
# As VIF<10, multi-collinearity is moderate and all columns can be used
"""
   VIF Factor           features
0    4.153268          GRE Score
1    3.792866        TOEFL Score
2    2.508768  University Rating
3    2.775750                SOP
4    2.037308                LOR
5    4.651670               CGPA
6    1.459311           Research

"""

# Split the data into train and test sets
#Iter 1- X_train,X_test,y_train,y_test=train_test_split(df1,y,test_size=0.25,random_state=42)
# X_train,X_test,y_train,y_test=train_test_split(df1,y,test_size=0.25,random_state=0) #Iter 2
# X_train,X_test,y_train,y_test=train_test_split(df1,y,test_size=0.15,random_state=42) #Iter 3
X_train,X_test,y_train,y_test=train_test_split(df1,y,test_size=0.30,random_state=42) #Iter 4

# random_state-to prevent random assignment of data and maintain the sam sets for any no. of iterations
print(X_train.shape) #(375, 7)
print(X_test.shape) #(125, 7)
print(y_train.shape) #(375,)
print(y_test.shape) #(125,)

# Model building
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

# Save the model as a pickle file
# pd.to_pickle(lr,"admission.pkl")
# pd.read_pickle("admission.pkl")

pd.to_pickle(scaler,"scaling.pkl")
print(lr.predict([[337,118,4,4.5,4.5,9.65,1]])) #[11.32582343]

# Similar transformations on the unseen data as performed on the training data must be done before passing the
# data to the model
test1=scaler.transform([[337,118,4,4.5,4.5,9.65,1]])
print(test1) #[[1.84274116 1.78854223 0.77890565 1.13735981 1.09894429 1.77680627  0.88640526]]

print(lr.predict([[1.84274116,1.78854223,0.77890565,1.13735981,1.09894429,1.77680627, 0.88640526]]))#[0.95976005]

# Metrics
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred)) #0.8175497115836483


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred)) #0.0035617495747129516

#To check for overfitting or underfitting
print(lr.score(X_train,y_train)) #0.8205773138971959,0.828,0.819,0.816
print(lr.score(X_test,y_test)) #0.8175497115836483,0.792,0.823,0.825

"""
# Checking for statistics
import statsmodels.api as sm
X_train=sm.add_constant(X_train,prepend=False)
model=sm.OLS(y_train,X_train)
result=model.fit()
print(result.summary())
"""
# How to improve the accuracy?-change the random_state, change the test size

#Regularization
#1-LASSO/L1
lassocv=LassoCV(cv=10,max_iter=200000,normalize=True)
lassocv.fit(X_train,y_train)
print(lassocv.alpha_) #3.258324690865486e-05
lasso=Lasso(alpha=lassocv.alpha_)
lasso.fit(X_train,y_train)

y_pred1=lasso.predict(X_test)

print(r2_score(y_test,y_pred1)) #0.8256463202475404
print(lasso.score(X_test,y_test)) #0.8256463202475404

"""
CV-A technique to extract the best possible parameter thru a random/series of random experiments.
ex:cv=5
It means divide the entire dataset into 5 subsets and at every iteration, keep one subset as the test set 
and treat the 4 others as train sets and compute the metrics at every iteration. Repeat this process for 
the specified no. of iterations and extract the parameters from the iteration for which the highest 
metrics are obtained.

"""
#2-RIDGE
ridgecv=RidgeCV(alphas=np.random.uniform(0,10,50),cv=10,normalize=True) #does not have max_iter,0 to 10, 50 numbers in b/n them
ridgecv.fit(X_train,y_train)
print(ridgecv.alpha_) #0.08568816585790895

ridge=Ridge(alpha=ridgecv.alpha_)
ridge.fit(X_train,y_train)

y_pred2=ridge.predict(X_test)

print(r2_score(y_test,y_pred2)) #0.8256511503372924
print(ridge.score(X_test,y_test)) #0.8256511503372924

#ElasticNet
elastnetcv=ElasticNetCV(alphas=None,cv=10)
elastnetcv.fit(X_train,y_train)
print(elastnetcv.alpha_) #0.0011178753566638416

enet=ElasticNet(alpha=elastnetcv.alpha_)
enet.fit(X_train,y_train)

y_pred3=enet.predict(X_test)

print(r2_score(y_test,y_pred3)) #0.8255772806476936
print(enet.score(X_test,y_test)) #0.8255772806476936

# alpha for all the above regularisation techniques is actually lambda in theory and
# alpha in ENet is actually the l1_ratio=0.5
"""
All the three techniques are giving almost the same accuracy implying the model is a stable model and 
no overfitted results are observed.
"""

# Task-Deployment in Heroku,AWS
