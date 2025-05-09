    if plot:
        block__ = ids_block[grid.dense_grid_slice]
        
        # Define specific levels for the four categories: <1, 1-2, 3-4, >4
        levels = [float('-inf'), 1.0, 2.0, 3.0, 4.0, float('inf')]
        
        # Use a colormap with distinct colors for the categories
        # You can choose a different colormap like 'viridis', 'tab10', 'Set1', etc.
        cmap = plt.cm.get_cmap('autumn', 4)  # 4 distinct colors
        
        contour = plt.contourf(block__.reshape(50, 5, 50)[:, 0, :].T, 
                              levels=levels, 
                              cmap=cmap,
                              extent=(.25, .75, .25, .75))

        xyz = interpolation_input.surface_points.sp_coords
        plt.plot(xyz[:, 0], xyz[:, 2], "o")
        
        # Create a colorbar with custom ticks and labels
        cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['< 1', '1-2', '3-4', '> 4'])  # Custom labels

        plt.show()
