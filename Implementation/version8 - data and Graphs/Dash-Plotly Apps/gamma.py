import scipy.stats as stats

# Generate some sample data
import numpy as np
y_values = np.random.gamma(2, 1, 1000)

# Fit a gamma distribution to the data
params = stats.gamma.fit(y_values)

# Create an x-axis for plotting the distribution
x = np.linspace(y_values.min(), y_values.max(), 100)

# Calculate the probability density function (PDF) for the fitted distribution
pdf = stats.gamma.pdf(x, *params)

# Plot the PDF
import matplotlib.pyplot as plt
plt.plot(x, pdf)
plt.show()
