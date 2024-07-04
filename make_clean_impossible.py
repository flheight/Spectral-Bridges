import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from spectralbridges import SpectralBridges

# Set random seed for reproducibility
np.random.seed(0)

# Load the dataset
data = np.genfromtxt('datasets/impossible.csv', delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Define the number of clusters
k = 7

# Initialize and fit the SpectralBridges model
net = SpectralBridges(n_clusters=k, n_nodes=250)
net.fit(X)

# Predict cluster labels
guess = net.predict(X)

# Create a plotly figure
fig = go.Figure()

# Add scatter plots for each cluster
for i in range(k):
    fig.add_trace(go.Scatter(
        x=X[(y == i), 0], 
        y=X[(y == i), 1], 
        mode='markers', 
        marker=dict(color='black', size=5), 
    ))
    
for i in range(k):
    fig.add_trace(go.Scatter(
        x=net.clusters_[i][:, 0], 
        y=net.clusters_[i][:, 1], 
        mode='markers', 
        marker=dict(size=15), 
    ))

# Update layout
fig.update_layout(
    xaxis_title="X",
    yaxis_title="Y",
    width=800,
    height=600,
    showlegend=False
)

# Save the figure as an pdf file
pio.kaleido.scope.mathjax = None
pio.write_image(fig, 'clean_impossible.pdf')
