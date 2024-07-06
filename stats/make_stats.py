import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Define the dimensions and scores
dim = ['h=8', 'h=16', 'h=32', 'h=64', 'h=784 (full)']

km_ari = [0.3625, 0.3661, 0.3726, 0.3821, 0.3665]
em_ari = [0.4587, 0.4974, 0.4799, 0.4440, 0.1850]
wc_ari = [0.4240, 0.4716, 0.5068, 0.4904, 0.4297]
sb_ari = [0.5985, 0.6876, 0.7213, 0.7069, 0.6619]

km_nmi = [0.4750, 0.4890, 0.4957, 0.5036, 0.4915]
em_nmi = [0.5921, 0.6424, 0.6318, 0.5958, 0.3252]
wc_nmi = [0.5467, 0.6220, 0.6577, 0.6536, 0.6129]
sb_nmi = [0.6709, 0.7627, 0.7960, 0.7854, 0.7628]

# Define colors for each method and their lighter shades
colors = {
    'KM': ('blue', 'lightblue'),
    'EM': ('green', 'lightgreen'),
    'WC': ('red', 'lightcoral'),
    'SB': ('purple', 'plum')
}

# Helper function to determine the color of each bar
def get_bar_color(score, max_score, method):
    return colors[method][0] if score == max_score else colors[method][1]

# Create the figure with subplots for ARI and NMI
fig = make_subplots(rows=1, cols=2)

# Methods and their corresponding scores
methods = ['KM', 'EM', 'WC', 'SB']
ari_scores = [km_ari, em_ari, wc_ari, sb_ari]
nmi_scores = [km_nmi, em_nmi, wc_nmi, sb_nmi]

# Plot bars for each method for ARI scores
for i, (method, score) in enumerate(zip(methods, ari_scores)):
    max_scores = [max(km_ari[i], em_ari[i], wc_ari[i], sb_ari[i]) for i in range(len(dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(dim))],
        width=0.2,
        showlegend=True  # Only show legend once
    ), row=1, col=1)

# Plot bars for each method for NMI scores
for i, (method, score) in enumerate(zip(methods, nmi_scores)):
    max_scores = [max(km_nmi[i], em_nmi[i], wc_nmi[i], sb_nmi[i]) for i in range(len(dim))]
    fig.add_trace(go.Bar(
        name=method,
        x=dim,
        y=score,
        marker_color=[get_bar_color(score[i], max_scores[i], method) for i in range(len(dim))],
        width=0.2,
        showlegend=False  # Only show legend once
    ), row=1, col=2)

# Update layout with larger and bold axis titles and tick labels
fig.update_layout(
    plot_bgcolor='whitesmoke',
    width=1200,
    height=600,
    xaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    ),
    yaxis=dict(
        title_font=dict(size=28),
        tickfont=dict(size=24)
    ),
    barmode='group',
    legend=dict(
        font=dict(
            size=20,
            weight='bold'
        ),
        title_font_size=24,
        itemsizing='constant'
    )
)

fig.update_xaxes(title_font=dict(size=24), tickfont=dict(size=22))
fig.update_yaxes(title_text='ARI Score', row=1, col=1, title_font=dict(size=28), tickfont=dict(size=24))
fig.update_yaxes(title_text='NMI Score', row=1, col=2, title_font=dict(size=28), tickfont=dict(size=24))

# Save the figure as a pdf file
pio.kaleido.scope.mathjax = None
pio.write_image(fig, "mnist_summary.pdf")
