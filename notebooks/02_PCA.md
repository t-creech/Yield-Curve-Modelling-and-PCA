# PCA

## Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

## Read in the Data


```python
# Define maturities
TENORS = ['1MO','3MO','6MO','1Y','2Y','3Y','5Y','7Y','10Y','20Y','30Y']

# Load yield changes (stationary)
df = pd.read_csv("../data/processed/cleaned_with_changes.csv", parse_dates=['date'])
df = df.set_index('date')[TENORS].dropna()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1MO</th>
      <th>3MO</th>
      <th>6MO</th>
      <th>1Y</th>
      <th>2Y</th>
      <th>3Y</th>
      <th>5Y</th>
      <th>7Y</th>
      <th>10Y</th>
      <th>20Y</th>
      <th>30Y</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-08-01</th>
      <td>-0.02</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>2001-08-02</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>0.06</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>2001-08-03</th>
      <td>-0.02</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>2001-08-06</th>
      <td>-0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>-0.03</td>
      <td>-0.05</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2001-08-07</th>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>



For PCA, we apply the analysis to daily changes in yields, not the raw yield levels. As shown in the EDA section, the levels of yields are non-stationary—they drift over time due to monetary policy regimes, inflation trends, and macroeconomic cycles. In contrast, yield changes are stationary, meaning they fluctuate around a stable mean with roughly constant variance.

This matters because PCA relies on the covariance matrix, and that covariance is assumed to be stable through time. When a series is non-stationary, its covariance structure drifts, and PCA ends up capturing long-term trends rather than the true co-movement of the yield curve.

By differencing the data, we remove these slow-moving trends and ensure that PCA reflects only the dynamic movements of the curve—the level, slope, and curvature changes—rather than artifacts of drift. This allows the principal components to be economically meaningful and stable across time.

## Fit PCA


```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio (first 5 PCs):")
for i, v in enumerate(pca.explained_variance_ratio_[:5]):
    print(f"PC{i+1}: {v:.3%}")
```

    Explained variance ratio (first 5 PCs):
    PC1: 64.380%
    PC2: 18.840%
    PC3: 7.513%
    PC4: 4.476%
    PC5: 2.173%


Now that we have the data read in, we should be able to apply PCA. However, before we do that, we will standardize the data. To standardize the data, we will utilize the StandardScalar object in sklearn. Using the fit_transform method, we can standardize our data to have mean 0 and variance 1. This is particularly important when dealing with yields as short rate yields are often times much more volatile than those at the long end of the curve, causing them to have greater variance. If not standardized, this greater variance would cause the short term rates to have much more influence on our principal components than our long term rates, which would make it much more difficult to identify the co-movements of the curve.  

Once the data is standardized, we are ready to apply PCA. To start, we will apply PCA using sklearn's decomposition library. The specific thing we need from this library is the object PCA. Once we initiate the object, we can use the fit_transform method to apply PCA. Once we have the principalk components, we can get the explained_variance_ratio attributes to see the explained variance of each prinicpal component.

## Scree Plot


```python
expl_var = np.cumsum(pca.explained_variance_ratio_) * 100
plt.figure(figsize=(8,5))
plt.plot(range(1, len(TENORS)+1), expl_var, marker='o')
plt.title("Cumulative Variance Explained by PCA Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained (%)")
plt.grid(True)
plt.show()
```


    
![png](02_PCA_files/02_PCA_10_0.png)
    


Now, we have created a skree plot for the PCA components. This plot shows us the amount of variance that is explained by each of the components. From this plot, we can see that over 90% of the variance is explained by the first three components.


```python
loadings = pd.DataFrame(pca.components_.T, index=TENORS, 
                        columns=[f'PC{i+1}' for i in range(len(TENORS))])

loadings.iloc[:, :3].plot(marker='o', figsize=(8,5))
plt.title("PCA Loadings for the First Three Components")
plt.xlabel("Maturity")
plt.ylabel("Loading Value")
plt.grid(True)
plt.show()

loadings.iloc[:, :3]
```


    
![png](02_PCA_files/02_PCA_12_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1MO</th>
      <td>0.095646</td>
      <td>0.453230</td>
      <td>0.630024</td>
    </tr>
    <tr>
      <th>3MO</th>
      <td>0.160418</td>
      <td>0.535036</td>
      <td>0.196127</td>
    </tr>
    <tr>
      <th>6MO</th>
      <td>0.234660</td>
      <td>0.454178</td>
      <td>-0.172919</td>
    </tr>
    <tr>
      <th>1Y</th>
      <td>0.296911</td>
      <td>0.292377</td>
      <td>-0.343598</td>
    </tr>
    <tr>
      <th>2Y</th>
      <td>0.338151</td>
      <td>0.052058</td>
      <td>-0.337391</td>
    </tr>
    <tr>
      <th>3Y</th>
      <td>0.352918</td>
      <td>-0.016395</td>
      <td>-0.239619</td>
    </tr>
    <tr>
      <th>5Y</th>
      <td>0.361513</td>
      <td>-0.110275</td>
      <td>-0.073166</td>
    </tr>
    <tr>
      <th>7Y</th>
      <td>0.358294</td>
      <td>-0.167586</td>
      <td>0.046016</td>
    </tr>
    <tr>
      <th>10Y</th>
      <td>0.350523</td>
      <td>-0.207937</td>
      <td>0.152499</td>
    </tr>
    <tr>
      <th>20Y</th>
      <td>0.326180</td>
      <td>-0.251577</td>
      <td>0.293444</td>
    </tr>
    <tr>
      <th>30Y</th>
      <td>0.307689</td>
      <td>-0.258464</td>
      <td>0.358549</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's take a look at the components themselves which are also stored as an attribute of our PCA object. For the first principal component, we see that it is positive across all tenors. This shows us that the principal components is explaining the movement of the tenors as they all move together (all move up or all move down). Therefore, we can take this to mean that the principal component is explaining the "level" of the yield curve. The second principal component is positive for the short maturities and negative for the long maturities. Therefore, this component is describing variance of the yield curve where the short maturities move up while the long maturities move down. This would be the idea of the slope of the yield curve. The final component is positive for the short maturities, negative for medium maturities, and positive for long maturities. Therefore, this component is describing how the "hump" of "U-shape" of the yield curve where short and long maturities move together and medium maturities move opposite of the other two. This would be the curvature of the yield curve.


```python
factors = pd.DataFrame(X_pca[:, :3], index=df.index, 
                       columns=['Level','Slope','Curvature'])

factors.plot(subplots=True, figsize=(10,8), legend=False,
             title=["PC1 - Level","PC2 - Slope","PC3 - Curvature"])
plt.suptitle("Principal Components of Daily Yield Changes", y=0.94)
plt.show()

factors.head()
```


    
![png](02_PCA_files/02_PCA_14_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Level</th>
      <th>Slope</th>
      <th>Curvature</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-08-01</th>
      <td>1.583647</td>
      <td>-0.557097</td>
      <td>-0.550486</td>
    </tr>
    <tr>
      <th>2001-08-02</th>
      <td>2.574513</td>
      <td>-0.971528</td>
      <td>-0.079391</td>
    </tr>
    <tr>
      <th>2001-08-03</th>
      <td>1.134251</td>
      <td>-0.544871</td>
      <td>-0.324036</td>
    </tr>
    <tr>
      <th>2001-08-06</th>
      <td>-0.690933</td>
      <td>-0.108093</td>
      <td>0.360891</td>
    </tr>
    <tr>
      <th>2001-08-07</th>
      <td>0.558202</td>
      <td>-0.093182</td>
      <td>0.036467</td>
    </tr>
  </tbody>
</table>
</div>


