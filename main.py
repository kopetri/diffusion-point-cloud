import numpy as np
import plotly.graph_objects as go
def vis_pc(point_cloud):
    # Assuming the point cloud has shape (N, 3) where N is the number of points (x, y, z)
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Create the 3D scatter plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=z,  # Optional: Color by z value for variation
            colorscale='Viridis',  # Color scale for variation
            opacity=0.8
        )
    )])

    # Set the layout of the plot
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        title='3D Point Cloud Visualization'
    )
    # Show the plot interactively
    fig.show()
if __name__ == '__main__':
    #ref = np.load('results/AE_Ours_all_1726125068/ref.npy')
    #out = np.load('results/AE_Ours_all_1726125068/out.npy')
    
    ref = np.load('Point Diffusion/z284kers/predictions/ref.npy')
    out = np.load('Point Diffusion/z284kers/predictions/out.npy')
    
    vis_pc(out[0])
    vis_pc(ref[0])
    
    
