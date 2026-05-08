import random

m='G05'
sess=TFC_cond[m]
#cell_ = 20

num_random_cells = 10
plt.figure(); 
#plt.plot(sess.S[cell_,:], 'r', alpha=0.5); 
random_cells = random.sample(range(sess.S.shape[0]), num_random_cells)
for cell in random_cells:
    plt.plot(sess.S[cell, :], 'r', alpha=0.5)
plt.plot(sess.velocities_miniscope, 'b'); 
plt.plot(sess.velocities_miniscope_smooth, 'k')

def plot_random_cells(mouse, session, num_random_cells=10, multiplier=1):
    """
    Plots random cells' activity for a given mouse and session.
    Parameters:
    mouse (str): Identifier for the mouse.
    session (dict): Dictionary containing session data for each mouse.
    num_random_cells (int, optional): Number of random cells to plot. Default is 10.
    The function selects a random subset of cells from the session data and plots their activity.
    It also overlays velocity data and event markers (tone and shock onsets/offsets) on the plots.
    Example usage:
    --------------
    plot_random_cells('mouse1', session_data, num_random_cells=10)
    """

    sess = session[mouse]
    random_cells = random.sample(range(sess.S.shape[0]), num_random_cells)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
    fig.suptitle(f'Random Cells for Mouse {mouse}', fontsize=16)
    for i, cell in enumerate(random_cells[:9]):
        ax = axes[i // 3, i % 3]
        ax.plot(sess.velocities_miniscope, 'b', alpha=0.5)
        ax.plot(sess.velocities_miniscope_smooth, 'k', alpha=0.5)
        for x in sess.tone_onsets * multiplier:
            ax.axvline(x, c='b', ls='--')
        for x in sess.tone_offsets * multiplier:
            ax.axvline(x, c='b', ls='--')
        for x in sess.shock_onsets * multiplier:
            ax.axvline(x, c='r', ls='--')
        for x in sess.shock_offsets * multiplier:
            ax.axvline(x, c='r', ls='--')
        norm_value = sess.S[cell,:].max()
        ax.plot(sess.S[cell, :], 'r') 
        ax.plot((sess.C[cell, :]/sess.C[cell,:].max())*norm_value, 'orange') 
        ax.text(0.95, 0.95, f'Cell {cell}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=1))
    plt.tight_layout()
    print(random_cells[:9])

# Example usage
m='G05'
sess=TFC_cond
plot_random_cells(m, TFC_cond)

from scipy.stats import pearsonr
# plot_random_cells('G05',TFC_cond)
# [96, 193, 558, 536, 153, 44, 566, 414, 14]
cell = 193
S_ = TFC_cond['G05'].S[cell,:]
v_ = TFC_cond['G05'].velocities_miniscope_smooth
correlation, p_value = pearsonr(S_, v_)
print(f'correlation {correlation} p_value {p_value}')
sess = TFC_cond['G05']
shock_binary_vector = np.zeros(sess.S.shape[1], dtype=int)
for onset, offset in zip(sess.shock_onsets, sess.shock_offsets):
    shock_binary_vector[onset:offset] = 1
shock_binary_vector
s_ = shock_binary_vector

X = np.column_stack((v_, s_))
X = sm.add_constant(X)  # Add intercept (baseline activity)
model_ = sm.OLS(S_, X).fit()
print(model_.summary())
residual_activity = model_.resid  # Residuals (activity after accounting for velocity and shock)


from statsmodels.tsa.tsatools import lagmat

# Create lagged versions of velocity (up to 3 lags)
v_lags = lagmat(v_, maxlag=3, trim='both')  # Creates lagged velocity matrix
s_lags = lagmat(s_, maxlag=3, trim='both')  # Lagged shock indicators
S_trimmed = S_[3:]  # Trim the first 3 rows of y to align with lags

# Combine all predictors
X_lags = np.column_stack((v_lags, s_lags))
X_lags = sm.add_constant(X_lags)

# Fit the model with lags
model_lags = sm.OLS(S_trimmed, X_lags).fit()
print(model_lags.summary())



import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLSAR

# Prepare the design matrix X (independent variables)
# Add a constant term for the intercept
X = np.column_stack([v_, s_])  # Combine velocity and shock indicator
X = sm.add_constant(X)       # Add intercept column

# Fit GLSAR with AR(1) structure for residuals
model = GLSAR(S_, X, rho=1)  # rho=1 specifies AR(1) errors
results = model.iterative_fit(maxiter=10)  # Iteratively estimate rho and fit the model

# Print the summary of results
print(results.summary())

# Add lagged predictors
lagged_v = np.roll(v_, 1)
lagged_s = np.roll(s_, 1)
X = np.column_stack([v_, s_, lagged_v, lagged_s])
X = sm.add_constant(X)

model = GLSAR(S_, X, rho=1)
results = model.iterative_fit(maxiter=10)
print(results.summary())

# Interaction between shock and velocity
interaction = v_ * s_
X = np.column_stack([v_, s_, interaction])
X = sm.add_constant(X)

model = GLSAR(S_, X, rho=1)
results = model.iterative_fit(maxiter=10)
print(results.summary())
# x1 is velocity
# x2 is shock
# x3 is interaction between velocity and shock

def calculate_and_plot_shock_period_activities_only_S(session, mice_per_group):
    """
    Calculate the average S activities for all shock periods per mouse and per cell,
    and plot a boxplot of these values for each group.
    Parameters:
    session (dict): Dictionary containing session data for each mouse.
    mice_per_group (dict): Dictionary where keys are group names and values are lists of mouse identifiers.
    """
    avg_activities = {'S': {}}
    
    for group, mice in mice_per_group.items():
        avg_activities['S'][group] = []
        
        for mouse in mice:
            sess = session[mouse]
            num_cells = sess.S.shape[0]
            
            for cell in range(num_cells):
                avg_S = []
                for onset, offset in zip(sess.shock_onsets, sess.shock_offsets):
                    S_values = sess.S[cell, onset:offset]
                    avg_S.append(np.nanmean(S_values))
                avg_activities['S'][group].append(np.mean(np.nan_to_num(avg_S)))
    
    # Plotting
    plt.figure(figsize=(10, 5))
    data = [avg_activities['S'][group] for group in mice_per_group.keys()]
    plt.boxplot(data, labels=mice_per_group.keys())
    plt.xlabel('Group')
    plt.ylabel('Average S Activity')
    plt.title('Boxplot of Average S Activities During Shock Periods')
    plt.show()


def calculate_and_plot_shock_period_activities_both_C_S(session, mice_per_group):
    """
    Calculate the average C and S activities for all shock periods per mouse and per cell,
    normalize per number of cells per mouse, then per group and plot across groups.
    Parameters:
    session (dict): Dictionary containing session data for each mouse.
    mice_per_group (dict): Dictionary where keys are group names and values are lists of mouse identifiers.
    """
    avg_activities = {'C': {}, 'S': {}}
    
    for group, mice in mice_per_group.items():
        avg_activities['C'][group] = []
        avg_activities['S'][group] = []
        
        for mouse in mice:
            sess = session[mouse]
            num_cells = sess.S.shape[0]
            shock_periods = zip(sess.shock_onsets, sess.shock_offsets)
            
            avg_C_per_cell = []
            avg_S_per_cell = []
            
            for cell in range(num_cells):
                avg_C = []
                avg_S = []
                for onset, offset in shock_periods:
                    C_values = sess.C[cell, onset:offset]
                    S_values = sess.S[cell, onset:offset]
                    avg_C.append(np.nanmean(np.nan_to_num(C_values)))
                    avg_S.append(np.nanmean(np.nan_to_num(S_values)))
                avg_C_per_cell.append(np.nanmean(avg_C))
                avg_S_per_cell.append(np.nanmean(avg_S))
            
            avg_activities['C'][group].append(np.nanmean(avg_C_per_cell))
            avg_activities['S'][group].append(np.nanmean(avg_S_per_cell))
        
        avg_activities['C'][group] = np.array(avg_activities['C'][group])
        avg_activities['S'][group] = np.array(avg_activities['S'][group])
        
        # Normalize per number of cells per mouse
        avg_activities['C'][group] /= [session[mouse].S.shape[0] for mouse in mice]
        avg_activities['S'][group] /= [session[mouse].S.shape[0] for mouse in mice]
        
        # Normalize per group
        avg_activities['C'][group] /= len(mice)
        avg_activities['S'][group] /= len(mice)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    for group in mice_per_group.keys():
        plt.bar(group, avg_activities['C'][group], alpha=0.5, label=f'{group} C Activity')
        plt.bar(group, avg_activities['S'][group], alpha=0.5, label=f'{group} S Activity')
    plt.xlabel('Group')
    plt.ylabel('Normalized Activity')
    plt.title('Average C and S Activities During Shock Periods')
    plt.legend()
    plt.show()
    

def calculate_and_plot_shock_period_activities_only_S(session, mice_per_group):
    """
    Calculate the average S activities for all shocbk periods per mouse and per cell,
    normalize per number of cells per mouse, then per group and plot across groups.
    Parameters:
    session (dict): Dictionary containing session data for each mouse.
    mice_per_group (dict): Dictionary where keys are group names and values are lists of mouse identifiers.
    """
    avg_activities = {'S': {}}
    
    for group, mice in mice_per_group.items():
        avg_activities['S'][group] = []
        
        for mouse in mice:
            sess = session[mouse]
            num_cells = sess.S.shape[0]
            shock_periods = list(zip(sess.shock_onsets, sess.shock_offsets))
            
            avg_S_per_cell = []
            
            for cell in range(num_cells):
                avg_S = []
                for onset, offset in shock_periods:
                    S_values = sess.S[cell, onset:offset]
                    avg_S.append(np.nanmean(np.nan_to_num(S_values)))
                avg_S_per_cell.append(np.nanmean(avg_S))
            
            avg_activities['S'][group].append(np.nanmean(avg_S_per_cell))
        
        avg_activities['S'][group] = np.array(avg_activities['S'][group])
        
        # Normalize per number of cells per mouse
        avg_activities['S'][group] /= [session[mouse].S.shape[0] for mouse in mice]
        
        # Normalize per group
        avg_activities['S'][group] /= len(mice)
        #avg_activities['S'][group] = np.mean(avg_activities['S'][group])
    
    # Plotting
    plt.figure(figsize=(10, 5))
    #for group in mice_per_group.keys():
    #    plt.bar(group, avg_activities['S'][group], alpha=0.5, label=f'{group} S Activity')
    data = [avg_activities['S'][group] for group in mice_per_group.keys()]
    for group in mice_per_group.keys():
        y = avg_activities['S'][group]
        x = np.random.normal(loc=list(mice_per_group.keys()).index(group) + 1, scale=0.04, size=len(y))
        plt.scatter(x, y, alpha=0.6)
    plt.boxplot(data, labels=mice_per_group.keys())        
    plt.xlabel('Group')
    plt.ylabel('Normalized S Activity')
    plt.title('Average S Activities During Shock Periods')
    plt.legend()
    plt.show()

def calculate_and_plot_shock_period_activities_only_S(session, mice_per_group):
    """
    Calculate the average S activities for all shock periods per mouse and per cell,
    normalize per number of responsive cells per mouse, then per group and plot across groups.
    Parameters:
    session (dict): Dictionary containing session data for each mouse.
    mice_per_group (dict): Dictionary where keys are group names and values are lists of mouse identifiers.
    """
    avg_activities = {'S': {}}
    
    for group, mice in mice_per_group.items():
        avg_activities['S'][group] = []
        
        for mouse in mice:
            sess = session[mouse]
            num_cells = sess.S.shape[0]
            shock_periods = list(zip(sess.shock_onsets, sess.shock_offsets))
            
            avg_S_per_cell = []
            responsive_cells = 0
            
            for cell in range(num_cells):
                avg_S = []
                for onset, offset in shock_periods:
                    S_values = sess.S[cell, onset:offset]
                    #avg_S.append(np.nanmean(np.nan_to_num(S_values)))
                    avg_S.append(np.mean(S_values))
                if len(avg_S) > 0:
                    avg_S_per_cell.append(np.mean(avg_S))
                    responsive_cells += 1
            
            if responsive_cells > 0:
                avg_activities['S'][group].append(np.nanmean(avg_S_per_cell))
        
        avg_activities['S'][group] = np.array(avg_activities['S'][group])
        
        # Normalize per number of responsive cells per mouse
        #avg_activities['S'][group] /= responsive_cells
        avg_activities['S'][group] /= [session[mouse].S.shape[0] for mouse in mice]
        
        # Normalize per group
        avg_activities['S'][group] /= len(mice)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    data = [avg_activities['S'][group] for group in mice_per_group.keys()]
    for group in mice_per_group.keys():
        y = avg_activities['S'][group]
        x = np.random.normal(loc=list(mice_per_group.keys()).index(group) + 1, scale=0.04, size=len(y))
        plt.scatter(x, y, alpha=0.6)
    plt.boxplot(data, labels=mice_per_group.keys())        
    plt.xlabel('Group')
    plt.ylabel('Normalized S Activity')
    plt.title('Average S Activities During Shock Periods')
    plt.legend()
    plt.show()


    def plot_percentage_responsive_cells_per_mouse(session, mice_per_group):
        """
        Plot the percentage of responsive cells per mouse for each group, and then averaged per group.
        Parameters:
        session (dict): Dictionary containing session data for each mouse.
        mice_per_group (dict): Dictionary where keys are group names and values are lists of mouse identifiers.
        """
        percentage_responsive_cells_per_mouse = {group: [] for group in mice_per_group.keys()}
        
        for group, mice in mice_per_group.items():
            for mouse in mice:
                sess = session[mouse]
                num_cells = sess.S.shape[0]
                shock_periods = list(zip(sess.shock_onsets, sess.shock_offsets))
                
                responsive_cells = 0

                for cell in range(num_cells):
                    avg_S = []
                    for onset, offset in shock_periods:
                        S_values = sess.S[cell, onset:offset]
                        avg_S.append(np.mean(S_values))
                    if np.count_nonzero(avg_S) ==5:
                        avg_S_per_cell.append(np.mean(avg_S))
                        responsive_cells += 1

                percentage_responsive_cells_per_mouse[group].append((responsive_cells / num_cells) * 100)
        
        # Plotting
        plt.figure(figsize=(10, 5))
        data = [percentage_responsive_cells_per_mouse[group] for group in mice_per_group.keys()]
        for group in mice_per_group.keys():
            y = percentage_responsive_cells_per_mouse[group]
            x = np.random.normal(loc=list(mice_per_group.keys()).index(group) + 1, scale=0.04, size=len(y))
            plt.scatter(x, y, alpha=0.6)
        plt.boxplot(data, labels=mice_per_group.keys())
        plt.xlabel('Group')
        plt.ylabel('Percentage of Responsive Cells')
        plt.title('Percentage of Responsive Cells Per Mouse')
        plt.show()
        
        # Plotting averaged per group
        avg_percentage_responsive_cells_per_group = {group: np.mean(percentage_responsive_cells_per_mouse[group]) for group in mice_per_group.keys()}
        plt.figure(figsize=(10, 5))
        plt.bar(avg_percentage_responsive_cells_per_group.keys(), avg_percentage_responsive_cells_per_group.values(), alpha=0.6)
        plt.xlabel('Group')
        plt.ylabel('Average Percentage of Responsive Cells')
        plt.title('Average Percentage of Responsive Cells Per Group')
        plt.show()

def get_responsive_cells(session, mice_per_group):
    """
    Get a dictionary of mice with values being lists of the cells that were deemed responsive.
    Parameters:
    session (dict): Dictionary containing session data for each mouse.
    mice_per_group (dict): Dictionary where keys are group names and values are lists of mouse identifiers.
    Returns:
    dict: Dictionary where keys are mouse identifiers and values are lists of responsive cell indices.
    """
    responsive_cells_dict = {mouse: [] for group in mice_per_group.values() for mouse in group}
    
    for group, mice in mice_per_group.items():
        for mouse in mice:
            sess = session[mouse]
            num_cells = sess.S.shape[0]
            shock_periods = list(zip(sess.shock_onsets, sess.shock_offsets))
            
            for cell in range(num_cells):
                avg_S = []
                for onset, offset in shock_periods:
                    S_values = sess.S[cell, onset:offset]
                    avg_S.append(np.mean(S_values))
                if np.count_nonzero(avg_S) >= 3:
                    responsive_cells_dict[mouse].append(cell)
    
    return responsive_cells_dict



def plot_avg_S_activities_per_shock_period(session, mice_per_group):
    """
    Plot average S activities for each shock period, averaged over all cells and mice.
    Parameters:
    session (dict): Dictionary containing session data for each mouse.
    mice_per_group (dict): Dictionary where keys are group names and values are lists of mouse identifiers.
    """
    avg_activities_per_period = {'S': {}}
    
    for group, mice in mice_per_group.items():
        avg_activities_per_period['S'][group] = []
        
        for mouse in mice:
            sess = session[mouse]
            num_cells = sess.S.shape[0]
            shock_periods = list(zip(sess.shock_onsets, sess.shock_offsets))
            
            avg_S_per_period = []
            
            for onset, offset in shock_periods:
                avg_S = []
                for cell in range(num_cells):
                    S_values = sess.S[cell, onset:offset]
                    avg_S.append(np.nanmean(np.nan_to_num(S_values)))
                avg_S_per_period.append(np.nanmean(avg_S))
            while len(avg_S_per_period) < 5:
                avg_S_per_period.append(0)
            
            avg_activities_per_period['S'][group].append(avg_S_per_period)
        
        avg_activities_per_period['S'][group] = np.mean(np.array(avg_activities_per_period['S'][group]), axis=0)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    for group in mice_per_group.keys():
        plt.plot(range(1, len(avg_activities_per_period['S'][group]) + 1), avg_activities_per_period['S'][group], label=f'{group} S Activity')
    plt.xticks(range(1, len(avg_activities_per_period['S'][group]) + 1))
    plt.xlabel('Shock Period')
    plt.ylabel('Average S Activity')
    plt.title('Average S Activities Per Shock Period')
    plt.legend()
    plt.show()


#
# PCA
#

S = TFC_cond['G05'].S
S_normalized = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
plt.figure(); plt.imshow(S); plt.title('S');
plt.figure(); plt.imshow(S_normalized); plt.title('S_normalized');
cell_=10; plt.figure(); plt.plot(S[cell_,:],'b'); plt.plot(S_normalized[cell_,:],'r')

from sklearn.decomposition import PCA
pca = PCA(n_components=3)  # Reduce to 3 components for 3D visualization
PCs = pca.fit_transform(S_normalized.T)  # Transpose to get frames × PCs

# 2D trajectory (e.g., PC1 vs PC2)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(PCs[:, 0], PCs[:, 1], marker='o', color='b', alpha=0.6, label='Trajectory')
plt.scatter(PCs[:, 0], PCs[:, 1], c=range(len(PCs)), cmap='viridis', label='Time')
plt.colorbar(label='Time (frames)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Population Activity Trajectory (PC1 vs PC2)')
plt.legend()
plt.show()

# 3D trajectory (e.g., PC1, PC2, PC3)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(PCs[:, 0], PCs[:, 1], PCs[:, 2], c=range(len(PCs)), cmap='viridis')
ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color='gray', alpha=0.5)
plt.colorbar(scatter, label='Time (frames)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Population Activity Trajectory in PC Space')
for onset, offset in zip(sess.shock_onsets, sess.shock_offsets):
    ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='red', label='Shock Period', zorder=10)
plt.show()

#
# Trial-averaged PCA
# 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA

def calculate_overlap(ax, PCs):
    """
    Calculate the amount of overlap of points in the 3D plot.
    This is a heuristic function and may need to be adjusted based on specific requirements.
    """
    # Project the 3D points to 2D
    proj_matrix = ax.get_proj()
    PCs_homogeneous = np.hstack([PCs, np.ones((PCs.shape[0], 1))])  # Convert to homogeneous coordinates
    projected_points = PCs_homogeneous @ proj_matrix.T  # Apply the projection matrix
    projected_points /= projected_points[:, 3].reshape(-1, 1)  # Normalize by the fourth coordinate

    # Calculate the density of points in the 2D projection
    hist, xedges, yedges = np.histogram2d(projected_points[:, 0], projected_points[:, 1], bins=50)
    overlap = np.sum(hist > 1)  # Count the number of bins with more than one point
    
    return overlap

def find_best_view(PCs):
    """
    Find the best viewing angles to minimize the amount of overlap of points.
    """
    best_elev = 30
    best_azim = 45
    min_overlap = float('inf')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for elev in range(0, 90, 10):
        for azim in range(0, 360, 10):
            ax.view_init(elev=elev, azim=azim)
            overlap = calculate_overlap(ax, PCs)
            if overlap < min_overlap:
                min_overlap = overlap
                best_elev = elev
                best_azim = azim
    
    plt.close(fig)
    return best_elev, best_azim  # Ensure the function returns the best angles

for m in mouse_groups.keys():
    print(f'*** PCA_avg: processing {m} {mouse_groups[m]} ...', end='')
    #m = 'G08'
    sess = TFC_cond[m]
    S = sess.S
    #S_normalized = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    S_normalized = np.nan_to_num((S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True))
    S = S_normalized

    # Define the number of frames for each period
    pre_tone_frames = 20
    tone_frames = 20
    post_tone_frames = 20
    shock_frames = 2
    post_shock_frames = 20

    # Initialize the new matrix
    S_new = np.zeros((S.shape[0], pre_tone_frames + tone_frames + post_tone_frames + shock_frames + post_shock_frames))

    # Iterate over each cell
    for cell in range(S.shape[0]):
        pre_tone_avg = []
        tone_avg = []
        post_tone_avg = []
        shock_avg = []
        post_shock_avg = []
        
        # Iterate over each tone and shock period
        for tone_onset, tone_offset, shock_onset, shock_offset in zip(sess.tone_onsets, sess.tone_offsets, sess.shock_onsets, sess.shock_offsets):
            pre_tone_avg.append(S[cell, tone_onset - pre_tone_frames:tone_onset])
            tone_avg.append(S[cell, tone_onset:tone_onset + tone_frames])
            post_tone_avg.append(S[cell, tone_onset + tone_frames:tone_onset + tone_frames + post_tone_frames])
            shock_avg.append(S[cell, shock_onset:shock_onset + shock_frames])
            post_shock_avg.append(S[cell, shock_onset + shock_frames:shock_onset + shock_frames + post_shock_frames])
        
        # Average the periods
        S_new[cell, :pre_tone_frames] = np.mean(pre_tone_avg, axis=0)
        S_new[cell, pre_tone_frames:pre_tone_frames + tone_frames] = np.mean(tone_avg, axis=0)
        S_new[cell, pre_tone_frames + tone_frames:pre_tone_frames + tone_frames + post_tone_frames] = np.mean(post_tone_avg, axis=0)
        S_new[cell, pre_tone_frames + tone_frames + post_tone_frames:pre_tone_frames + tone_frames + post_tone_frames + shock_frames] = np.mean(shock_avg, axis=0)
        S_new[cell, pre_tone_frames + tone_frames + post_tone_frames + shock_frames:] = np.mean(post_shock_avg, axis=0)

    pca = PCA(n_components=3)  # Reduce to 3 components for 3D visualization
    PCs = pca.fit_transform(S_new.T) # Transpose to get frames × PCs

    best_elev, best_azim = find_best_view(PCs)

    # 3D trajectory (e.g., PC1, PC2, PC3)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    pre_tone = (0, pre_tone_frames)
    tone = (pre_tone[1], pre_tone[1] + tone_frames)
    post_tone = (tone[1], tone[1] + post_tone_frames)
    shock = (post_tone[1], post_tone[1] + shock_frames)
    post_shock = (shock[1], shock[1] + post_shock_frames)

    # Plot each period with different colors
    ax.scatter(PCs[pre_tone[0]:pre_tone[1], 0], PCs[pre_tone[0]:pre_tone[1], 1], PCs[pre_tone[0]:pre_tone[1], 2], s=50, color='black', label='Pre-tone Period')
    ax.scatter(PCs[tone[0]:tone[1], 0], PCs[tone[0]:tone[1], 1], PCs[tone[0]:tone[1], 2], color='blue', label='Tone Period')
    ax.scatter(PCs[post_tone[0]:post_tone[1], 0], PCs[post_tone[0]:post_tone[1], 1], PCs[post_tone[0]:post_tone[1], 2], color='grey', label='Post-tone Period')
    ax.scatter(PCs[shock[0]:shock[1], 0], PCs[shock[0]:shock[1], 1], PCs[shock[0]:shock[1], 2], color='red', s=100, label='Shock Period')
    ax.scatter(PCs[post_shock[0]:post_shock[1], 0], PCs[post_shock[0]:post_shock[1], 1], PCs[post_shock[0]:post_shock[1], 2], color='green', label='Post-shock Period')
    
    ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color='gray', alpha=0.5, label='Trajectory')
    # Designate start and stop points
    ax.scatter(PCs[0, 0], PCs[0, 1], PCs[0, 2], color='black', s=100, marker='o', label='Start', zorder=10)
    ax.scatter(PCs[-1, 0], PCs[-1, 1], PCs[-1, 2], color='black', s=100, facecolors='none', edgecolors='black', label='End', zorder=10)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=best_elev, azim=best_azim)

    ax.set_title(f'{m} {mouse_groups[m]} trial-averaged PCA')
    ax.legend()
    plt.show()

    save_path = os.path.join(PLOTS_DIR, 'PCA_avg')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'PCA_avg-{mouse_groups[m]}-{m}.png'), format='png', dpi=600)
    if want_svg:
        svg_save_path = os.path.join(save_path, 'svg')
        os.makedirs(svg_save_path, exist_ok=True)
        plt.savefig(os.path.join(svg_save_path, f'PCA_avg-{mouse_groups[m]}-{m}.svg'), format='svg')
    #plt.close()
    print('done.')

#
# Previous PCA ploltting
#
# 3D trajectory (e.g., PC1, PC2, PC3)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(PCs[:, 0], PCs[:, 1], PCs[:, 2], c=range(len(PCs)), cmap='viridis')
ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color='gray', alpha=0.5)
plt.colorbar(scatter, label='Time (frames)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Population Activity Trajectory in PC Space')
for onset, offset in zip(sess.shock_onsets, sess.shock_offsets):
    ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='red', label='Shock Period', zorder=10)
plt.show()


#
# Crossreg PCA - AVG time course
# 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA

method = 1 # Method 1: Calculate PCA on Encoding and Apply to Others
           # Method 2: Concatenate and Calculate PCA
n_components = 3
engram_thresh = 0
auto_close = True
auto_angle_adjust = False

def calculate_overlap(ax, PCs):
    """
    Calculate the amount of overlap of points in the 3D plot.
    This is a heuristic function and may need to be adjusted based on specific requirements.
    """
    # Project the 3D points to 2D
    proj_matrix = ax.get_proj()
    PCs_homogeneous = np.hstack([PCs, np.ones((PCs.shape[0], 1))])  # Convert to homogeneous coordinates
    projected_points = PCs_homogeneous @ proj_matrix.T  # Apply the projection matrix
    projected_points /= projected_points[:, 3].reshape(-1, 1)  # Normalize by the fourth coordinate

    # Calculate the density of points in the 2D projection
    hist, xedges, yedges = np.histogram2d(projected_points[:, 0], projected_points[:, 1], bins=50)
    overlap = np.sum(hist > 1)  # Count the number of bins with more than one point
    
    return overlap

def find_best_view(PCs):
    """
    Find the best viewing angles to minimize the amount of overlap of points.
    """
    best_elev = 30
    best_azim = 45
    min_overlap = float('inf')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for elev in range(0, 90, 10):
        for azim in range(0, 360, 10):
            ax.view_init(elev=elev, azim=azim)
            overlap = calculate_overlap(ax, PCs)
            if overlap < min_overlap:
                min_overlap = overlap
                best_elev = elev
                best_azim = azim
    
    plt.close(fig)
    return best_elev, best_azim  # Ensure the function returns the best angles

def on_move(event):
    for ax in axes:
        ax.view_init(elev=axes[0].elev, azim=axes[0].azim)
    fig.canvas.draw_idle()

for m in mouse_groups.keys():
    if m in ['G07', 'G15']:
        continue
    print(f'*** PCA_crossreg: processing {m} {mouse_groups[m]} ', end='')

    S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)

    sess_TFC_cond = TFC_cond[m]
    sess_Test_B = Test_B[m]
    sess_Test_B_1wk = Test_B_1wk[m]

    S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
    S_Test_B = sess_Test_B.S[S_i_Test_B, :]
    S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]

    shock_idx = [0, len(sess_TFC_cond.shock_onsets)-1]
    tone_idx_TFC_cond = [0 , len(sess_TFC_cond.tone_onsets)-1]
    tone_idx_Test_B = [0 , len(sess_Test_B.tone_onsets)-1]
    tone_idx_Test_B_1wk = [0 , len(sess_Test_B_1wk.tone_onsets)-1]

    S_TFC_cond_n = np.nan_to_num((S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True)) / S_TFC_cond.std(axis=1, keepdims=True))
    S_Test_B_n = np.nan_to_num((S_Test_B - S_Test_B.mean(axis=1, keepdims=True)) / S_Test_B.std(axis=1, keepdims=True))
    S_Test_B_1wk_n = np.nan_to_num((S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True)) / S_Test_B_1wk.std(axis=1, keepdims=True))

    # Define the number of frames for each period
    pre_tone_frames = 20
    tone_frames = 20
    post_tone_frames = 20
    shock_frames = 2
    post_shock_frames = 20

    pre_tone = (0, pre_tone_frames)
    tone = (pre_tone[1], pre_tone[1] + tone_frames)
    post_tone = (tone[1], tone[1] + post_tone_frames)
    shock = (post_tone[1], post_tone[1] + shock_frames)
    post_shock = (shock[1], shock[1] + post_shock_frames)

    for want_engram in [True]:#[True, False]:
        if want_engram:
            score = sp.stats.zscore(np.sum(S_TFC_cond_n,axis=1))
            engram_cells = np.where(score > engram_thresh)[0]

            S_TFC_cond_n = S_TFC_cond_n[engram_cells, :]
            S_Test_B_n = S_Test_B_n[engram_cells, :]
            S_Test_B_1wk_n = S_Test_B_1wk_n[engram_cells, :]

        # Initialize the new matrix
        S_TFC_cond_avg = np.zeros((S_TFC_cond_n.shape[0], pre_tone_frames + tone_frames + post_tone_frames + shock_frames + post_shock_frames))
        S_Test_B_avg = np.zeros((S_Test_B_n.shape[0], pre_tone_frames + tone_frames + post_tone_frames))
        S_Test_B_1wk_avg = np.zeros((S_Test_B_1wk_n.shape[0], pre_tone_frames + tone_frames + post_tone_frames))
                
        # Iterate over each cell
        for [S, S_avg, sess] in [[S_TFC_cond_n, S_TFC_cond_avg, sess_TFC_cond], \
                                [S_Test_B_n, S_Test_B_avg, sess_Test_B], \
                                [S_Test_B_1wk_n, S_Test_B_1wk_avg, sess_Test_B_1wk]]:
            
            for cell in range(S.shape[0]):
                pre_tone_avg = []
                tone_avg = []
                post_tone_avg = []
                shock_avg = []
                post_shock_avg = []
                
                if sess.session_type == 'TFC_cond':
                    # Iterate over each tone and shock period
                    for tone_onset, tone_offset, shock_onset, shock_offset in zip(sess.tone_onsets, sess.tone_offsets, sess.shock_onsets, sess.shock_offsets):
                        pre_tone_avg.append(S[cell, tone_onset - pre_tone_frames:tone_onset])
                        tone_avg.append(S[cell, tone_onset:tone_onset + tone_frames])
                        post_tone_avg.append(S[cell, tone_onset + tone_frames:tone_onset + tone_frames + post_tone_frames])
                        shock_avg.append(S[cell, shock_onset:shock_onset + shock_frames])
                        post_shock_avg.append(S[cell, shock_onset + shock_frames:shock_onset + shock_frames + post_shock_frames])
                else:
                    for tone_onset, tone_offset in zip(sess.tone_onsets, sess.tone_offsets):
                        pre_tone_avg.append(S[cell, tone_onset - pre_tone_frames:tone_onset])
                        tone_avg.append(S[cell, tone_onset:tone_onset + tone_frames])
                        post_tone_avg.append(S[cell, tone_onset + tone_frames:tone_onset + tone_frames + post_tone_frames])

                # Average the periods
                S_avg[cell, :pre_tone_frames] = np.mean(pre_tone_avg, axis=0)
                S_avg[cell, pre_tone_frames:pre_tone_frames + tone_frames] = np.mean(tone_avg, axis=0)
                S_avg[cell, pre_tone_frames + tone_frames:pre_tone_frames + tone_frames + post_tone_frames] = np.mean(post_tone_avg, axis=0)
                if sess.session_type == 'TFC_cond':
                    S_avg[cell, pre_tone_frames + tone_frames + post_tone_frames:pre_tone_frames + tone_frames + post_tone_frames + shock_frames] = np.mean(shock_avg, axis=0)
                    S_avg[cell, pre_tone_frames + tone_frames + post_tone_frames + shock_frames:] = np.mean(post_shock_avg, axis=0)

        #for n_components in [2, 3]:
        for n_components in [3]:
            for method in [1, 2]:
                if method == 1: # Calculate PCA on Encoding and Apply to Others
                    pca = PCA(n_components=n_components)
                    pca.fit(S_TFC_cond_avg.T)
                    
                    PCs_TFC_cond = pca.transform(S_TFC_cond_avg.T)
                    PCs_Test_B = pca.transform(S_Test_B_avg.T)
                    PCs_Test_B_1wk = pca.transform(S_Test_B_1wk_avg.T)

                elif method == 2: # Concatenate and Calculate PCA
                    S_tot = np.hstack([S_TFC_cond_avg, S_Test_B_avg, S_Test_B_1wk_avg])
                    pca = PCA(n_components=n_components)
                    pca.fit(S_tot.T)

                    PCs_TFC_cond = pca.transform(S_TFC_cond_avg.T)
                    PCs_Test_B = pca.transform(S_Test_B_avg.T)
                    PCs_Test_B_1wk = pca.transform(S_Test_B_1wk_avg.T)
                
                if n_components == 3:
                    best_elev_TFC_cond, best_azim_TFC_cond = find_best_view(PCs_TFC_cond)
                    best_elev_Test_B, best_azim_Test_B = find_best_view(PCs_Test_B)
                    best_elev_Test_B_1wk, best_azim_Test_B_1wk = find_best_view(PCs_Test_B_1wk)

                # Determine the axis limits for PC1, PC2, and PC3
                all_PCs = np.vstack([PCs_TFC_cond, PCs_Test_B, PCs_Test_B_1wk])
                x_limits = (all_PCs[:, 0].min(), all_PCs[:, 0].max())
                y_limits = (all_PCs[:, 1].min(), all_PCs[:, 1].max())
                if n_components == 3:
                    z_limits = (all_PCs[:, 2].min(), all_PCs[:, 2].max())

                x_limits_Test_Bs = min(PCs_Test_B[:, 0].min(), PCs_Test_B_1wk[:, 0].min()), max(PCs_Test_B[:, 0].max(), PCs_Test_B_1wk[:, 0].max())
                y_limits_Test_Bs = min(PCs_Test_B[:, 1].min(), PCs_Test_B_1wk[:, 1].min()), max(PCs_Test_B[:, 1].max(), PCs_Test_B_1wk[:, 1].max())
                if n_components == 3:
                    z_limits_Test_Bs = min(PCs_Test_B[:, 2].min(), PCs_Test_B_1wk[:, 2].min()), max(PCs_Test_B[:, 2].max(), PCs_Test_B_1wk[:, 2].max())

                plot_types = ['full'];
                if method == 1: # 2 is concatenated so don't need to zoom, all will fit.
                    plot_types.append('zoom')

                for plot_type in plot_types:
                    # 3D trajectory (e.g., PC1, PC2, PC3) for TFC_cond, Test_B, and Test_B_1wk in separate subplots
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'} if n_components == 3 else None)
                    axes = axes.flatten()

                    for ax, PCs, title_str in zip(axes, \
                        [PCs_TFC_cond, PCs_Test_B, PCs_Test_B_1wk], \
                        ['TFC_cond', 'Test_B', 'Test_B_1wk']):

                        if n_components == 2:
                            ax.scatter(PCs[pre_tone[0]:pre_tone[1], 0], PCs[pre_tone[0]:pre_tone[1], 1], s=50, color='black', label='Pre-tone Period')
                            ax.scatter(PCs[tone[0]:tone[1], 0], PCs[tone[0]:tone[1], 1], color='blue', label='Tone Period')
                            ax.scatter(PCs[post_tone[0]:post_tone[1], 0], PCs[post_tone[0]:post_tone[1], 1], color='grey', label='Post-tone Period')
                            if title_str == 'TFC_cond':
                                ax.scatter(PCs[shock[0]:shock[1], 0], PCs[shock[0]:shock[1], 1], color='red', s=100, label='Shock Period')
                                ax.scatter(PCs[post_shock[0]:post_shock[1], 0], PCs[post_shock[0]:post_shock[1], 1], color='green', label='Post-shock Period')
                            
                            ax.plot(PCs[:, 0], PCs[:, 1], color='gray', alpha=0.5, label='Trajectory')
                            # Designate start and stop points
                            ax.scatter(PCs[0, 0], PCs[0, 1], color='black', s=200, marker='o', label='Start', zorder=10)
                            ax.scatter(PCs[-1, 0], PCs[-1, 1], color='black', s=200, facecolors='none', edgecolors='black', label='End', zorder=10)
                        
                        elif n_components == 3:
                            ax.scatter(PCs[pre_tone[0]:pre_tone[1], 0], PCs[pre_tone[0]:pre_tone[1], 1], PCs[pre_tone[0]:pre_tone[1], 2], s=50, color='black', label='Pre-tone Period')
                            ax.scatter(PCs[tone[0]:tone[1], 0], PCs[tone[0]:tone[1], 1], PCs[tone[0]:tone[1], 2], color='blue', label='Tone Period')
                            ax.scatter(PCs[post_tone[0]:post_tone[1], 0], PCs[post_tone[0]:post_tone[1], 1], PCs[post_tone[0]:post_tone[1], 2], color='grey', label='Post-tone Period')
                            if title_str == 'TFC_cond':
                                ax.scatter(PCs[shock[0]:shock[1], 0], PCs[shock[0]:shock[1], 1], PCs[shock[0]:shock[1], 2], color='red', s=100, label='Shock Period')
                                ax.scatter(PCs[post_shock[0]:post_shock[1], 0], PCs[post_shock[0]:post_shock[1], 1], PCs[post_shock[0]:post_shock[1], 2], color='green', label='Post-shock Period')
                            
                            ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color='gray', alpha=0.5, label='Trajectory')
                            # Designate start and stop points
                            ax.scatter(PCs[0, 0], PCs[0, 1], PCs[0, 2], color='black', s=200, marker='o', label='Start', zorder=10)
                            ax.scatter(PCs[-1, 0], PCs[-1, 1], PCs[-1, 2], color='black', s=200, facecolors='none', edgecolors='black', label='End', zorder=10)

                        if plot_type == 'full':
                            ax.set_xlim(x_limits)
                            ax.set_ylim(y_limits)
                            if n_components == 3:
                                ax.set_zlim(z_limits)
                        elif plot_type == 'zoom':
                            ax.set_xlim(x_limits_Test_Bs)
                            ax.set_ylim(y_limits_Test_Bs)
                            if n_components == 3:
                                ax.set_zlim(z_limits_Test_Bs)

                        ax.set_xlabel('PC1')
                        ax.set_ylabel('PC2')
                        if n_components == 3:
                            ax.set_zlabel('PC3')
                        ax.set_title(title_str)
                        ax.legend(fontsize='small')

                    if n_components == 3:
                        fig.canvas.mpl_connect('motion_notify_event', on_move)

                    plt.suptitle(f'{m} {mouse_groups[m]} PCA crossreg trial-avg ({plot_type})')
                    plt.show()

                    if want_engram:
                        save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_avg_engram')
                    else:
                        save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_avg')
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(os.path.join(save_path, f'PCA_crossreg_avg-want_engram_{want_engram}-n_components{n_components}-method{method}-{plot_type}-{mouse_groups[m]}-{m}.png'), format='png', dpi=600)
                    if want_svg:
                        svg_save_path = os.path.join(save_path, 'svg')
                        os.makedirs(svg_save_path, exist_ok=True)
                        plt.savefig(os.path.join(svg_save_path, f'PCA_crossreg_avg-want_engram_{want_engram}-n_components{n_components}-method{method}-{plot_type}-{mouse_groups[m]}-{m}.svg'), format='svg')
                    if auto_close:
                        plt.close()
                    print('.', end='')
        print('done.')

#
# Crossreg PCA - FULL time course
# 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

#method = 1 # Method 1: Calculate PCA on Encoding and Apply to Others
           # Method 2: Concatenate and Calculate PCA

marker_size = 2
marker_alpha = 0.6
marker_size_tone = 5
marker_size_shock = 5

auto_angle_adjust=False
preset_angle_adjust=True

want_engram = True
engram_thresh = 0
auto_close = True
want_svg = False

want_gaussian_smoothing = True
smoothing_sigma = 1.5

azim_rotation = 70

mouse_elev_azim = {
    'G05' : {'Time' : (18,4), 'Location': (19,16)},
    'G06' : {'Time' : (17,16), 'Location' : (14,10)},
    'G08' : {'Time' : (36,11), 'Location' : (30,10)},
    'G09' : {'Time' : (27,12), 'Location' : (30,10)},
    'G10' : {'Time' : (27,30), 'Location' : (24,20)},
    'G11' : {'Time' : (24,34), 'Location' : (25,22)},
    'G12' : {'Time' : (30,10), 'Location' : (30,10)},
    'G13' : {'Time' : (30,10), 'Location' : (28,12)},
    'G14' : {'Time' : (25,14), 'Location' : (24,24)},
    'G16' : {'Time' : (30,10), 'Location' : (30,10)},
    'G17' : {'Time' : (24,22), 'Location' : (21,18)},
    'G18' : {'Time' : (25,23), 'Location' : (30,10)},
    'G19' : {'Time' : (13,11), 'Location' : (13,21)},
    'G20' : {'Time' : (24,-33), 'Location' : (26,-23)},
    'G21' : {'Time' : (3,25), 'Location' : (3,25)}
} 

debug_mode = False

def calculate_overlap(ax, PCs):
    """
    Calculate the amount of overlap of points in the 3D plot.
    This is a heuristic function and may need to be adjusted based on specific requirements.
    """
    # Project the 3D points to 2D
    proj_matrix = ax.get_proj()
    PCs_homogeneous = np.hstack([PCs, np.ones((PCs.shape[0], 1))])  # Convert to homogeneous coordinates
    projected_points = PCs_homogeneous @ proj_matrix.T  # Apply the projection matrix
    projected_points /= projected_points[:, 3].reshape(-1, 1)  # Normalize by the fourth coordinate

    # Calculate the density of points in the 2D projection
    hist, xedges, yedges = np.histogram2d(projected_points[:, 0], projected_points[:, 1], bins=50)
    overlap = np.sum(hist > 1)  # Count the number of bins with more than one point
    
    return overlap

def find_best_view(PCs):
    """
    Find the best viewing angles to minimize the amount of overlap of points.
    """
    best_elev = 30
    best_azim = 45
    min_overlap = float('inf')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for elev in range(0, 90, 10):
        for azim in range(0, 360, 10):
            ax.view_init(elev=elev, azim=azim)
            overlap = calculate_overlap(ax, PCs)
            if overlap < min_overlap:
                min_overlap = overlap
                best_elev = elev
                best_azim = azim
    
    plt.close(fig)
    return best_elev, best_azim  # Ensure the function returns the best angles

def on_move(event):
    for ax in axes:
        ax.view_init(elev=axes[0].elev, azim=axes[0].azim)
    fig.canvas.draw_idle()

#### SHOCK & TONE
for m in mouse_groups.keys():
    if m in ['G07', 'G15']:
        continue

    print(f'*** PCA_crossreg: processing {m} {mouse_groups[m]} ', end='')
    S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)

    sess_TFC_cond = TFC_cond[m]
    sess_Test_B = Test_B[m]
    sess_Test_B_1wk = Test_B_1wk[m]

    S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
    S_Test_B = sess_Test_B.S[S_i_Test_B, :]
    S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]

    # Z-scoring (ensuring PCA is centered around origin, and features are standard scaled)
    S_TFC_cond_n = np.nan_to_num((S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True)) / S_TFC_cond.std(axis=1, keepdims=True))
    S_Test_B_n = np.nan_to_num((S_Test_B - S_Test_B.mean(axis=1, keepdims=True)) / S_Test_B.std(axis=1, keepdims=True))
    S_Test_B_1wk_n = np.nan_to_num((S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True)) / S_Test_B_1wk.std(axis=1, keepdims=True))

    # Centering only (not as good)
    #S_TFC_cond_n = np.nan_to_num(S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True))
    #S_Test_B_n = np.nan_to_num(S_Test_B - S_Test_B.mean(axis=1, keepdims=True))
    #S_Test_B_1wk_n = np.nan_to_num(S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True))

    if want_gaussian_smoothing:
        S_TFC_cond_n = gaussian_filter1d(S_TFC_cond_n, sigma=smoothing_sigma, axis=1)
        S_Test_B_n = gaussian_filter1d(S_Test_B_n, sigma=smoothing_sigma, axis=1)
        S_Test_B_1wk_n = gaussian_filter1d(S_Test_B_1wk_n, sigma=smoothing_sigma, axis=1) 
    
    for want_engram in [True]:#[True, False]:
        if want_engram:
            save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_full_engram_toneshock')
        else:
            save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_full_toneshock')
        os.makedirs(save_path, exist_ok=True)
                
        if want_engram:
            score = sp.stats.zscore(np.sum(S_TFC_cond,axis=1))
            engram_cells = np.where(score > engram_thresh)[0]

            S_TFC_cond_n = S_TFC_cond_n[engram_cells, :]
            S_Test_B_n = S_Test_B_n[engram_cells, :]
            S_Test_B_1wk_n = S_Test_B_1wk_n[engram_cells, :]

        for only_tone_shock in [True]:#[True, False]:
            for method in [2]:#[1, 2]:
                if method == 1: # Calculate PCA on Encoding and Apply to Others 
                    # RETIRED since variance likely not the same across the three sessions, which would be a requirement for this method
                    pca = PCA(n_components=3)
                    pca.fit(S_TFC_cond_n.T)
                    
                    PCs_TFC_cond = pca.transform(S_TFC_cond_n.T)
                    PCs_Test_B = pca.transform(S_Test_B_n.T)
                    PCs_Test_B_1wk = pca.transform(S_Test_B_1wk_n.T)

                elif method == 2: # Concatenate and Calculate PCA
                    S_tot = np.hstack([S_TFC_cond_n, S_Test_B_n, S_Test_B_1wk_n])
                    pca = PCA(n_components=3)
                    pca.fit(S_TFC_cond_n.T)

                    PCs_TFC_cond = pca.transform(S_TFC_cond_n.T)
                    PCs_Test_B = pca.transform(S_Test_B_n.T)
                    PCs_Test_B_1wk = pca.transform(S_Test_B_1wk_n.T)
                
                best_elev_TFC_cond, best_azim_TFC_cond = find_best_view(PCs_TFC_cond)
                best_elev_Test_B, best_azim_Test_B = find_best_view(PCs_Test_B)
                best_elev_Test_B_1wk, best_azim_Test_B_1wk = find_best_view(PCs_Test_B_1wk)

                # Determine the axis limits for PC1, PC2, and PC3
                all_PCs = np.vstack([PCs_TFC_cond, PCs_Test_B, PCs_Test_B_1wk])
                x_limits = (all_PCs[:, 0].min(), all_PCs[:, 0].max())
                y_limits = (all_PCs[:, 1].min(), all_PCs[:, 1].max())
                z_limits = (all_PCs[:, 2].min(), all_PCs[:, 2].max())

                # Find the conjunction of minimum and maximum x, y, z limits for all tone and shock sessions across all three PCs
                if only_tone_shock:
                    tone_shock_PCs = np.vstack([
                        np.vstack([PCs_TFC_cond[onset:offset] for onset, offset in zip(sess_TFC_cond.tone_onsets, sess_TFC_cond.tone_offsets)]),
                        np.vstack([PCs_Test_B[onset:offset] for onset, offset in zip(sess_Test_B.tone_onsets, sess_Test_B.tone_offsets)]),
                        np.vstack([PCs_Test_B_1wk[onset:offset] for onset, offset in zip(sess_Test_B_1wk.tone_onsets, sess_Test_B_1wk.tone_offsets)]),
                        np.vstack([PCs_TFC_cond[onset:offset] for onset, offset in zip(sess_TFC_cond.shock_onsets, sess_TFC_cond.shock_offsets)]),
                    ])

                    x_limits_tone_shock = (tone_shock_PCs[:, 0].min(), tone_shock_PCs[:, 0].max())
                    y_limits_tone_shock = (tone_shock_PCs[:, 1].min(), tone_shock_PCs[:, 1].max())
                    z_limits_tone_shock = (tone_shock_PCs[:, 2].min(), tone_shock_PCs[:, 2].max())


                #
                # Plot trajectories in 2D space and PCA space
                #
                # Assuming sess_TFC_cond.loc_X_miniscope_smooth and sess_TFC_cond.loc_Y_miniscope_smooth are numpy arrays
                for trajectory_type in ['Time', 'Location']:
                    for PCs, sess, sess_name in zip([PCs_TFC_cond, PCs_Test_B, PCs_Test_B_1wk], \
                                                    [sess_TFC_cond, sess_Test_B, sess_Test_B_1wk], \
                                                    ['TFC_cond', 'Test_B', 'Test_B_1wk']):
                        fig = plt.figure(figsize=(12, 6))
                        ax1 = fig.add_subplot(121)
                        ax2 = fig.add_subplot(122, projection='3d')
                        #axes = axes.flatten()
                        ax = ax1
                        x = sess.loc_X_miniscope_smooth
                        y = sess.loc_Y_miniscope_smooth

                        if m in ['G09', 'G21']: # Because Miniscope 19.avi was not recorded, see main.py
                            x = x[0:PCs.shape[0]]
                            y = y[0:PCs.shape[0]]
                        '''
                        PCs_time = np.zeros_like(PCs)
                        PCs_location = np.zeros_like(PCs)

                        for i in range(PCs.shape[0]):
                            PCs_time[i] = PCs[i] * values[i]
                            PCs_location[i] = PCs[i] * values[i]
                        '''

                        if trajectory_type == 'Time':
                            # Color by time
                            values = np.linspace(0, 1, len(x))
                            colorbar_label = 'Time (frames)'
                        elif trajectory_type == 'Location':
                            len_x = int(np.ceil(max(x) - min(x)))
                            len_y = int(np.ceil(max(y) - min(y)))
                            # Create a meshgrid of coordinates
                            coords_x = np.linspace(0, 1, len_x)
                            coords_y = np.linspace(0, 1, len_y)
                            mesh_X, mesh_Y = np.meshgrid(x, y)
                            # Define color gradients (e.g., from blue to red horizontally and green to yellow vertically)
                            R = mesh_X  # Red component varies with x
                            G = mesh_Y  # Green component varies with y
                            B = 1 - mesh_X  # Blue component inversely varies with x
                            # Combine the RGB components
                            colors = np.stack((R, G, B), axis=-1)
                            # Plot the gradient
                            plt.figure(figsize=(6, 6))
                            plt.imshow(colors, origin='lower', extent=[0, 1, 0, 1])
                            plt.title('Color-Tiled Square with Unique Colors for Each Position')
                            plt.xlabel('X-axis')
                            plt.ylabel('Y-axis')
                            plt.grid(False)
                            plt.show()

                            #values = np.sqrt(x**2 + y**2)
                            colorbar_label = 'Distance from Origin'
                        sc = ax.scatter(x, y, c=values, cmap='viridis', s=1)
                        fig.colorbar(sc, ax=ax, label=colorbar_label)
                        ax.set_xlabel('X Position')
                        ax.set_ylabel('Y Position')
                        #ax.set_title(f'Mouse trajectory {m} {mouse_groups[m]}')

                        ax = ax2
                        ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color='gray', alpha=0.5, label='Trajectory')
                        sc = ax.scatter(PCs[:, 0], PCs[:, 1], PCs[:, 2], c=values, cmap='viridis', alpha=0.6)
                        #plt.colorbar(sc, ax=ax, label='Distance from Origin')
                        if only_tone_shock:
                            ax.set_xlim(x_limits_tone_shock)
                            ax.set_ylim(y_limits_tone_shock)
                            ax.set_zlim(z_limits_tone_shock)
                        else:
                            ax.set_xlim(x_limits)
                            ax.set_ylim(y_limits)
                            ax.set_zlim(z_limits)
                        
                        ax.set_xlabel('PC1')
                        ax.set_ylabel('PC2')
                        ax.set_zlabel('PC3')
                        def on_move(event):
                            ax.view_init(elev=ax.elev, azim=ax.azim)
                            fig.canvas.draw_idle()
                            # Update the annotation with the current elevation and azimuth
                            annotation.set_text(f'Elev: {ax.elev:.1f}, Azim: {ax.azim:.1f}')
                            annotation.set_position((event.x, event.y))

                        # Create an annotation to display the elevation and azimuth
                        annotation = fig.text(0.02, 0.95, '', transform=fig.transFigure, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

                        fig.canvas.mpl_connect('motion_notify_event', on_move)

                        if debug_mode:
                            elev, azim = ax.elev, ax.azim
                            ax.view_init(elev=elev, azim=azim + azim_rotation)

                        if preset_angle_adjust:
                            ax.view_init(elev=mouse_elev_azim[m][trajectory_type][0], azim=mouse_elev_azim[m][trajectory_type][1])

                        if auto_angle_adjust:
                            ax.view_init(elev=best_elev, azim=best_azim)        
                        #ax.set_title(f'Mouse trajectory in PC space {m} {mouse_groups[m]}')
                        plt.suptitle(f'Mouse trajectory in 2D and PC space {m} {mouse_groups[m]} {sess_name} by {trajectory_type}')

                        if debug_mode and sess_name == 'TFC_cond':
                            input("Press enter to continue")

                        plt.savefig(os.path.join(save_path, f'trajectory-2Dloc_and_PCA_crossreg_full-method{method}-type-{trajectory_type}-{mouse_groups[m]}-{m}-{sess_name}.png'), format='png', dpi=600)
                        if want_svg:
                            svg_save_path = os.path.join(save_path, 'svg')
                            os.makedirs(svg_save_path, exist_ok=True)
                            plt.savefig(os.path.join(svg_save_path, f'trajectory-2Dloc_and_PCA_crossreg_full-method{method}-type-{trajectory_type}-{mouse_groups[m]}-{m}-{sess_name}.svg'), format='svg')
                        print('.', end='')
                        if auto_close:
                            plt.close()

                # 3D trajectory (e.g., PC1, PC2, PC3) for TFC_cond, Test_B, and Test_B_1wk in separate subplots
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
                axes = axes.flatten()
                for ax, PCs, title_str, sess, best_elev, best_azim in zip(axes, \
                        [PCs_TFC_cond, PCs_Test_B, PCs_Test_B_1wk], \
                        ['TFC_cond', 'Test_B', 'Test_B_1wk'], \
                        [sess_TFC_cond, sess_Test_B, sess_Test_B_1wk], \
                        [best_elev_TFC_cond, best_elev_Test_B, best_elev_Test_B_1wk], \
                        [best_azim_TFC_cond, best_azim_Test_B, best_azim_Test_B_1wk]):
                    
                    ax.scatter(PCs[0, 0], PCs[0, 1], PCs[0, 2], color='black', s=200, marker='o', label='Start', zorder=10)
                    ax.scatter(PCs[-1, 0], PCs[-1, 1], PCs[-1, 2], color='black', s=200, facecolors='none', edgecolors='black', label='End', zorder=10)

                    if not only_tone_shock:
                        sc = ax.scatter(PCs[:, 0], PCs[:, 1], PCs[:, 2], c=np.arange(PCs.shape[0]), cmap='viridis', alpha=0.6, label='Trajectory', s=marker_size)
                    ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color='gray', alpha=0.5, label='Trajectory')

                    #sc = ax.scatter(PCs[:, 0], PCs[:, 1], PCs[:, 2], c=np.arange(len(PCs)), cmap='viridis', s=marker_size, alpha=marker_alpha)
                    #tone_scatter = ax.scatter([], [], color='blue', label='Tone Period', s=marker_size_tone, marker='^')
                    #shock_scatter = ax.scatter([], [], color='red', label='Shock Period', s=marker_size_shock, marker='x')
                    for i, (onset, offset) in enumerate(zip(sess.tone_onsets, sess.tone_offsets)):
                        if i == 0:
                            ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='darkblue', s=marker_size_tone, marker='^')
                            ax.plot(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='blue', alpha=0.6, label='Tone Trajectory', linewidth=0.5)
                        elif i == len(sess.tone_onsets) - 1:
                            ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='lightblue', s=marker_size_tone, marker='^')
                            ax.plot(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='lightblue', alpha=0.6, label='Tone Trajectory', linewidth=0.5)                            
                    # Concatenate all onset, offset PCs for sess.tone_onsets/tone_offsets
                    #tone_onset_offset_PCs = np.vstack([PCs[onset:offset] for onset, offset in zip(sess.tone_onsets, sess.tone_offsets)])
                    # Plot the concatenated tone onset/offset PCs as a single curve with a colormap
                    #ax.plot(tone_onset_offset_PCs[:, 0], tone_onset_offset_PCs[:, 1], tone_onset_offset_PCs[:, 2], color='blue', alpha=0.6, label='Tone Trajectory', linewidth=0.5)
                    #ax.scatter(tone_onset_offset_PCs[:, 0], tone_onset_offset_PCs[:, 1], tone_onset_offset_PCs[:, 2], c=np.arange(tone_onset_offset_PCs.shape[0]), cmap='cool', s=marker_size, alpha=marker_alpha)

                    if title_str == 'TFC_cond':

                        if only_tone_shock:
                            shock_PCs = np.vstack([PCs[onset:offset] for onset, offset in zip(sess.shock_onsets, sess.shock_offsets)])
                            x_limits_shock = (shock_PCs[:, 0].min(), shock_PCs[:, 0].max())
                            y_limits_shock = (shock_PCs[:, 1].min(), shock_PCs[:, 1].max())
                            z_limits_shock = (shock_PCs[:, 2].min(), shock_PCs[:, 2].max())

                        for i, (onset, offset) in enumerate(zip(sess.shock_onsets, sess.shock_offsets)):
                            if i == 0:
                                ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='darkred', s=marker_size_shock, marker='x')
                                ax.plot(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='darkred', alpha=0.6, label='Shock Trajectory', linewidth=0.5)
                            elif i == len(sess.shock_onsets) - 1:
                                ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='lightcoral', s=marker_size_shock, marker='x')
                                ax.plot(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='lightcoral', alpha=0.6, label='Shock Trajectory', linewidth=0.5)
                            #else:
                            #    ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], edgecolor='red', facecolor='none', s=marker_size_shock, marker='x')

                    
                    #if title_str == 'TFC_cond':
                    #    for onset, offset in zip(sess.shock_onsets, sess.shock_offsets):
                    #        ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='red', s=marker_size_shock, marker='x')
                    
                    if only_tone_shock:
                        ax.set_xlim(x_limits_tone_shock)
                        ax.set_ylim(y_limits_tone_shock)
                        ax.set_zlim(z_limits_tone_shock)
                    else:
                        ax.set_xlim(x_limits)
                        ax.set_ylim(y_limits)
                        ax.set_zlim(z_limits)
                    
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_zlabel('PC3')
                    elev, azim = ax.elev, ax.azim
                    ax.view_init(elev=elev, azim=azim + azim_rotation)
                    if auto_angle_adjust:
                        ax.view_init(elev=best_elev, azim=best_azim)        
                    ax.set_title(title_str)
                    #ax.legend(handles=[sc.legend_elements()[0][0], tone_scatter, shock_scatter], labels=['Trajectory', 'Tone Period', 'Shock Period'], fontsize='small')

                plt.suptitle(f'{m} {mouse_groups[m]} PCA crossreg full time course (method {method})')
                if not only_tone_shock:
                    plt.colorbar(sc, ax=axes, label='Time (frames)', shrink=0.5)
                plt.show()

                if want_engram:
                    save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_full_engram_toneshock')
                else:
                    save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_full_toneshock')
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'PCA_crossreg_full-want_engram-{want_engram}-only_tone_shock_{only_tone_shock}-method{method}-{mouse_groups[m]}-{m}.png'), format='png', dpi=600)
                if want_svg:
                    svg_save_path = os.path.join(save_path, 'svg')
                    os.makedirs(svg_save_path, exist_ok=True)
                    plt.savefig(os.path.join(svg_save_path, f'PCA_crossreg_full-want_engram-{want_engram}-only_tone_shock_{only_tone_shock}-method{method}-{mouse_groups[m]}-{m}.svg'), format='svg')
                if auto_close:
                    plt.close()
                print('.', end='')

    print('done.')

##
## HERE
##

#### POST-SHOCK & POST-TONE
for m in mouse_groups.keys():
    if m in ['G07', 'G15']:
        continue
    print(f'*** PCA_crossreg: processing {m} {mouse_groups[m]} ', end='')
    S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
    S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)

    sess_TFC_cond = TFC_cond[m]
    sess_Test_B = Test_B[m]
    sess_Test_B_1wk = Test_B_1wk[m]

    S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
    S_Test_B = sess_Test_B.S[S_i_Test_B, :]
    S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]

    S_TFC_cond_n = np.nan_to_num((S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True)) / S_TFC_cond.std(axis=1, keepdims=True))
    S_Test_B_n = np.nan_to_num((S_Test_B - S_Test_B.mean(axis=1, keepdims=True)) / S_Test_B.std(axis=1, keepdims=True))
    S_Test_B_1wk_n = np.nan_to_num((S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True)) / S_Test_B_1wk.std(axis=1, keepdims=True))

    for want_engram in [True]:#[True, False]:
        if want_engram:
            score = sp.stats.zscore(np.sum(S_TFC_cond,axis=1))
            engram_cells = np.where(score > engram_thresh)[0]

            S_TFC_cond_n = S_TFC_cond_n[engram_cells, :]
            S_Test_B_n = S_Test_B_n[engram_cells, :]
            S_Test_B_1wk_n = S_Test_B_1wk_n[engram_cells, :]

        for only_tone_shock in [True, False]:
            for method in [2]:#[1, 2]:
                if method == 1: # Calculate PCA on Encoding and Apply to Others 
                    # RETIRED since variance likely not the same across the three sessions, which would be a requirement for this method
                    # but keeping this here for historical reasons
                    pca = PCA(n_components=3)
                    pca.fit(S_TFC_cond_n.T)
                    
                    PCs_TFC_cond = pca.transform(S_TFC_cond_n.T)
                    PCs_Test_B = pca.transform(S_Test_B_n.T)
                    PCs_Test_B_1wk = pca.transform(S_Test_B_1wk_n.T)

                elif method == 2: # Concatenate and Calculate PCA
                    S_tot = np.hstack([S_TFC_cond_n, S_Test_B_n, S_Test_B_1wk_n])
                    pca = PCA(n_components=3)
                    pca.fit(S_TFC_cond_n.T)

                    PCs_TFC_cond = pca.transform(S_TFC_cond_n.T)
                    PCs_Test_B = pca.transform(S_Test_B_n.T)
                    PCs_Test_B_1wk = pca.transform(S_Test_B_1wk_n.T)

                if auto_angle_adjust:
                    best_elev_TFC_cond, best_azim_TFC_cond = find_best_view(PCs_TFC_cond)
                    best_elev_Test_B, best_azim_Test_B = find_best_view(PCs_Test_B)
                    best_elev_Test_B_1wk, best_azim_Test_B_1wk = find_best_view(PCs_Test_B_1wk)

                # Determine the axis limits for PC1, PC2, and PC3
                all_PCs = np.vstack([PCs_TFC_cond, PCs_Test_B, PCs_Test_B_1wk])
                x_limits = (all_PCs[:, 0].min(), all_PCs[:, 0].max())
                y_limits = (all_PCs[:, 1].min(), all_PCs[:, 1].max())
                z_limits = (all_PCs[:, 2].min(), all_PCs[:, 2].max())

                # Find the conjunction of minimum and maximum x, y, z limits for all tone and shock sessions across all three PCs
                if only_tone_shock:
                    tone_shock_PCs = np.vstack([
                        np.vstack([PCs_TFC_cond[onset:offset] for onset, offset in zip(sess_TFC_cond.tone_offsets, np.array(sess_TFC_cond.tone_offsets)+MINISCOPE_FPS*20)]),
                        np.vstack([PCs_Test_B[onset:offset] for onset, offset in zip(sess_Test_B.tone_offsets, np.array(sess_Test_B.tone_offsets)+MINISCOPE_FPS*20)]),
                        np.vstack([PCs_Test_B_1wk[onset:offset] for onset, offset in zip(sess_Test_B_1wk.tone_offsets, np.array(sess_Test_B_1wk.tone_offsets)+MINISCOPE_FPS*20)]),
                        np.vstack([PCs_TFC_cond[onset:offset] for onset, offset in zip(sess_TFC_cond.shock_offsets, np.array(sess_TFC_cond.shock_offsets)+MINISCOPE_FPS*20)]),
                    ])

                    x_limits_tone_shock = (tone_shock_PCs[:, 0].min(), tone_shock_PCs[:, 0].max())
                    y_limits_tone_shock = (tone_shock_PCs[:, 1].min(), tone_shock_PCs[:, 1].max())
                    z_limits_tone_shock = (tone_shock_PCs[:, 2].min(), tone_shock_PCs[:, 2].max())

                # 3D trajectory (e.g., PC1, PC2, PC3) for TFC_cond, Test_B, and Test_B_1wk in separate subplots
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
                axes = axes.flatten()
                for ax, PCs, title_str, sess, best_elev, best_azim in zip(axes, \
                        [PCs_TFC_cond, PCs_Test_B, PCs_Test_B_1wk], \
                        ['TFC_cond', 'Test_B', 'Test_B_1wk'], \
                        [sess_TFC_cond, sess_Test_B, sess_Test_B_1wk], \
                        [best_elev_TFC_cond, best_elev_Test_B, best_elev_Test_B_1wk], \
                        [best_azim_TFC_cond, best_azim_Test_B, best_azim_Test_B_1wk]):
                    
                    if not only_tone_shock:
                        ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color='gray', alpha=0.6, label='Trajectory')
                    #sc = ax.scatter(PCs[:, 0], PCs[:, 1], PCs[:, 2], c=np.arange(len(PCs)), cmap='viridis', s=marker_size, alpha=marker_alpha)
                    #tone_scatter = ax.scatter([], [], color='blue', label='Tone Period', s=marker_size_tone, marker='^')
                    #shock_scatter = ax.scatter([], [], color='red', label='Shock Period', s=marker_size_shock, marker='x')

                    for i, (onset, offset) in enumerate(zip(sess.tone_offsets, np.array(sess.tone_offsets)+MINISCOPE_FPS*20)):
                        if i == 0:
                            ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='blue', s=marker_size_tone, marker='s')
                        elif i == len(sess.tone_onsets) - 1:
                            ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='lightskyblue', s=marker_size_tone, marker='s')

                    if title_str == 'TFC_cond':

                        if only_tone_shock:
                            shock_PCs = np.vstack([PCs[onset:offset] for onset, offset in zip(sess.shock_offsets, np.array(sess.shock_offsets)+MINISCOPE_FPS*20)])
                            x_limits_shock = (shock_PCs[:, 0].min(), shock_PCs[:, 0].max())
                            y_limits_shock = (shock_PCs[:, 1].min(), shock_PCs[:, 1].max())
                            z_limits_shock = (shock_PCs[:, 2].min(), shock_PCs[:, 2].max())

                        for i, (onset, offset) in enumerate(zip(sess.shock_offsets, np.array(sess.shock_offsets)+MINISCOPE_FPS*20)):
                            if i == 0:
                                ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='red', s=marker_size_shock, marker='o')
                            elif i == len(sess.shock_onsets) - 1:
                                ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='salmon', s=marker_size_shock, marker='o')
                            #else:
                            #    ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], edgecolor='red', facecolor='none', s=marker_size_shock, marker='x')

                    if only_tone_shock:
                        ax.set_xlim(x_limits_tone_shock)
                        ax.set_ylim(y_limits_tone_shock)
                        ax.set_zlim(z_limits_tone_shock)

                    ax.scatter(PCs[0, 0], PCs[0, 1], PCs[0, 2], color='black', s=200, marker='o', label='Start', zorder=10)
                    ax.scatter(PCs[-1, 0], PCs[-1, 1], PCs[-1, 2], color='black', s=200, facecolors='none', edgecolors='black', label='End', zorder=10)
                    
                    #if title_str == 'TFC_cond':
                    #    for onset, offset in zip(sess.shock_onsets, sess.shock_offsets):
                    #        ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='red', s=marker_size_shock, marker='x')
                    
                    if not only_tone_shock:
                        ax.set_xlim(x_limits)
                        ax.set_ylim(y_limits)
                        ax.set_zlim(z_limits)
                    
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_zlabel('PC3')
                    if auto_angle_adjust:
                        ax.view_init(elev=best_elev, azim=best_azim)        
                    ax.set_title(title_str)
                    #ax.legend(handles=[sc.legend_elements()[0][0], tone_scatter, shock_scatter], labels=['Trajectory', 'Tone Period', 'Shock Period'], fontsize='small')

                plt.suptitle(f'{m} {mouse_groups[m]} PCA crossreg full time course (method {method})')
                plt.colorbar(sc, ax=axes, label='Time (frames)', shrink=0.5)
                plt.show()

                if want_engram:
                    save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_full_posttoneshock_engram')
                else:
                    save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_full_posttoneshock')
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'PCA_crossreg_full-want_engram-{want_engram}-only_tone_shock_{only_tone_shock}-method{method}-{mouse_groups[m]}-{m}.png'), format='png', dpi=600)
                if want_svg:
                    svg_save_path = os.path.join(save_path, 'svg')
                    os.makedirs(svg_save_path, exist_ok=True)
                    plt.savefig(os.path.join(svg_save_path, f'PCA_crossreg_full-want_engram-{want_engram}-only_tone_shock_{only_tone_shock}-method{method}-{mouse_groups[m]}-{m}.svg'), format='svg')
                if auto_close:
                    plt.close()
                print('.', end='')
    print('done.')

# UMAP
from umap import UMAP
umap_use_random = True
iterations = 1
save_path = PLOTS_DIR
want_svg = False
marker_size=2
tone_marker_size=10
shock_marker_size=8
marker_alpha = 1
tone_marker_alpha = 1
shock_marker_alpha = 1
plot_S_debug = False

random_seeds = [random.randint(0, 1000000) for _ in range(iterations)]
print(f'Random seeds: {random_seeds}')

for iter, random_seed in zip(range(iterations), random_seeds):
    for n_neighbors in [150]:#enumerate([10, 15, 20, 50, 100, 150, 200]):#enumerate([10, 15, 20, 50]):
        for m in mouse_groups.keys():
            print(f'*** UMAP: processing {m} iter {iter} random_seed {random_seed} n_neighbors {n_neighbors} ...', end='')
            #m = 'G15'

            sess = TFC_cond[m]
            S = sess.S

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            for i, norm_str in enumerate(['Normalized', 'Non-normalized']):
                if norm_str == 'Normalized':
                    #S_normalized = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
                    S_normalized = np.nan_to_num((S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True))
                else:
                    S_normalized = S

                if plot_S_debug:
                    plt.figure(); plt.imshow(S); plt.title('S');
                    plt.figure(); plt.imshow(S_normalized); plt.title('S_normalized');
                    cell_=10; plt.figure(); plt.plot(S[cell_,:],'b'); plt.plot(S_normalized[cell_,:],'r')
                
                if umap_use_random:
                    umap = UMAP(n_components=2, n_neighbors=n_neighbors, n_jobs=-1)
                else:
                    umap = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_seed, n_jobs=1)
                embedding = umap.fit_transform(S_normalized.T)

                first_shock_idx = 0
                last_shock_idx = len(sess.shock_onsets)-1
                first_tone_idx = 0
                last_tone_idx = len(sess.tone_onsets)-1

                for j, (title_str, shock_idx_to_use, tone_idx_to_use) in enumerate(zip(['First', 'Last'], [first_shock_idx, last_shock_idx], [first_tone_idx, last_tone_idx])):
                    ax = axes[i * 2 + j]
                    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=np.arange(S.shape[1]), cmap='viridis', s=marker_size, alpha=marker_alpha)
                    shock_idx = shock_idx_to_use
                    tone_idx = tone_idx_to_use
                    ax.scatter(embedding[sess.tone_onsets[tone_idx]:sess.tone_offsets[tone_idx], 0], embedding[sess.tone_onsets[tone_idx]:sess.tone_offsets[tone_idx], 1], color='blue', label='Tone Period', s=tone_marker_size, alpha=tone_marker_alpha, marker='^')
                    ax.scatter(embedding[sess.shock_onsets[shock_idx]:sess.shock_offsets[shock_idx], 0], embedding[sess.shock_onsets[shock_idx]:sess.shock_offsets[shock_idx], 1], color='red', label='Shock Period', s=shock_marker_size, alpha=shock_marker_alpha, marker='x')
                    ax.set_title(f'UMAP Embedding {title_str} {norm_str}')

            # Create a single legend for all subplots
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), fontsize='small')

            plt.colorbar(sc, ax=axes, label='Frame index', shrink=0.7)
            #plt.tight_layout(rect=[0, 0, 0.85, 0.95])
            #plt.tight_layout(rect=[0, 0, 0.9, 0.9])
            plt.suptitle(f'{m} {mouse_groups[m]} UMAP embedding (n_neighbors={n_neighbors})')
            plt.show()

            save_path = os.path.join(PLOTS_DIR, 'UMAP', f'random_seed-{random_seed}')
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'UMAP-{mouse_groups[m]}-{m}_iter{iter}_seed-{random_seed}-n_neighbors_{n_neighbors}.png'), format='png', dpi=600)
            if want_svg:
                svg_save_path = os.path.join(save_path, 'svg')
                os.makedirs(svg_save_path, exist_ok=True)
                plt.savefig(os.path.join(svg_save_path, f'UMAP-{mouse_groups[m]}-{m}_iter{iter}_seed-{random_seed}-n_neighbors_{n_neighbors}.svg'), format='svg')
            plt.close()
            print('done.')


# UMAP: crossreg
from umap import UMAP
umap_use_random = True
iterations = 1
save_path = PLOTS_DIR
want_svg = False
marker_size=2
tone_marker_size=10
shock_marker_size=8
marker_alpha = 1
tone_marker_alpha = 1
shock_marker_alpha = 1
plot_S_debug = False

random_seeds = [random.randint(0, 1000000) for _ in range(iterations)]
print(f'Random seeds: {random_seeds}')

for iter, random_seed in zip(range(iterations), random_seeds):
    for m in mouse_groups.keys():
        if m in ['G07', 'G15']:
            continue
        print(f'*** UMAP-crossreg: processing {m} iter {iter} random_seed {random_seed} ...', end='')
        #m = 'G15'

        S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)

        sess_TFC_cond = TFC_cond[m]
        sess_Test_B = Test_B[m]
        sess_Test_B_1wk = Test_B_1wk[m]

        S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
        S_Test_B = sess_Test_B.S[S_i_Test_B, :]
        S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]

        shock_idx = [0, len(sess_TFC_cond.shock_onsets)-1]
        tone_idx_TFC_cond = [0 , len(sess_TFC_cond.tone_onsets)-1]
        tone_idx_Test_B = [0 , len(sess_Test_B.tone_onsets)-1]
        tone_idx_Test_B_1wk = [0 , len(sess_Test_B_1wk.tone_onsets)-1]
    
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        #axes = axes.flatten()

        S_TFC_cond_n = np.nan_to_num((S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True)) / S_TFC_cond.std(axis=1, keepdims=True))
        S_Test_B_n = np.nan_to_num((S_Test_B - S_Test_B.mean(axis=1, keepdims=True)) / S_Test_B.std(axis=1, keepdims=True))
        S_Test_B_1wk_n = np.nan_to_num((S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True)) / S_Test_B_1wk.std(axis=1, keepdims=True))

        if umap_use_random:
            umap_TFC_cond = UMAP(n_components=2, n_jobs=-1)
            umap_Test_B = UMAP(n_components=2, n_jobs=-1)
            umap_Test_B_1wk = UMAP(n_components=2, n_jobs=-1)
        else:
            umap_TFC_cond = UMAP(n_components=2, random_state=42, n_jobs=1)
            umap_Test_B = UMAP(n_components=2, random_state=42, n_jobs=1)
            umap_Test_B_1wk = UMAP(n_components=2, random_state=42, n_jobs=1)
        embedding_TFC_cond = umap_TFC_cond.fit_transform(S_TFC_cond_n.T)
        embedding_Test_B = umap_Test_B.fit_transform(S_Test_B_n.T)
        embedding_Test_B_1wk = umap_Test_B_1wk.fit_transform(S_Test_B_1wk_n.T)

        for i, first_last_str in enumerate(['First', 'Last']):
            ax = axes[i, 0]
            sc = ax.scatter(embedding_TFC_cond[:, 0], embedding_TFC_cond[:, 1], c=np.arange(S_TFC_cond.shape[1]), cmap='viridis', s=marker_size, alpha=marker_alpha)
            ax.scatter(embedding_TFC_cond[sess_TFC_cond.tone_onsets[tone_idx_TFC_cond[i]]:sess_TFC_cond.tone_offsets[tone_idx_TFC_cond[i]], 0], \
                       embedding_TFC_cond[sess_TFC_cond.tone_onsets[tone_idx_TFC_cond[i]]:sess_TFC_cond.tone_offsets[tone_idx_TFC_cond[i]], 1], \
                       color='blue', label='Tone Period', s=tone_marker_size, alpha=tone_marker_alpha, marker='^')
            ax.scatter(embedding_TFC_cond[sess_TFC_cond.shock_onsets[shock_idx[i]]:sess_TFC_cond.shock_offsets[shock_idx[i]], 0], \
                       embedding_TFC_cond[sess_TFC_cond.shock_onsets[shock_idx[i]]:sess_TFC_cond.shock_offsets[shock_idx[i]], 1], \
                       color='red', label='Shock Period', s=shock_marker_size, alpha=shock_marker_alpha, marker='x')
            ax.set_title(f'TFC_cond {first_last_str}')

            ax = axes[i, 1]
            sc = ax.scatter(embedding_Test_B[:, 0], embedding_Test_B[:, 1], c=np.arange(S_Test_B.shape[1]), cmap='viridis', s=marker_size, alpha=marker_alpha)
            ax.scatter(embedding_Test_B[sess_Test_B.tone_onsets[tone_idx_Test_B[i]]:sess_Test_B.tone_offsets[tone_idx_Test_B[i]], 0], \
                    embedding_Test_B[sess_Test_B.tone_onsets[tone_idx_Test_B[i]]:sess_Test_B.tone_offsets[tone_idx_Test_B[i]], 1], \
                    color='blue', label='Tone Period', s=tone_marker_size, alpha=tone_marker_alpha, marker='^')
            ax.set_title(f'Test_B {first_last_str}')

            ax = axes[i, 2]
            sc = ax.scatter(embedding_Test_B_1wk[:, 0], embedding_Test_B_1wk[:, 1], c=np.arange(S_Test_B_1wk.shape[1]), cmap='viridis', s=marker_size, alpha=marker_alpha)
            ax.scatter(embedding_Test_B_1wk[sess_Test_B_1wk.tone_onsets[tone_idx_Test_B_1wk[i]]:sess_Test_B_1wk.tone_offsets[tone_idx_Test_B_1wk[i]], 0], \
                    embedding_Test_B_1wk[sess_Test_B_1wk.tone_onsets[tone_idx_Test_B_1wk[i]]:sess_Test_B_1wk.tone_offsets[tone_idx_Test_B_1wk[i]], 1], \
                    color='blue', label='Tone Period', s=tone_marker_size, alpha=tone_marker_alpha, marker='^')
            ax.set_title(f'Test_B_1wk {first_last_str}')

        # Create a single legend for all subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), fontsize='small')

        plt.colorbar(sc, ax=axes, label='Frame index', shrink=0.7)
        #plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        #plt.tight_layout(rect=[0, 0, 0.9, 0.9])
        plt.suptitle(f'{m} {mouse_groups[m]} UMAP embedding')
        plt.show()

        save_path = os.path.join(PLOTS_DIR, 'UMAP', 'crossreg', f'random_seed-{random_seed}')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'UMAP-crossreg-{mouse_groups[m]}-{m}_iter{iter}_seed-{random_seed}.png'), format='png', dpi=600)
        if want_svg:
            svg_save_path = os.path.join(save_path, 'svg')
            os.makedirs(svg_save_path, exist_ok=True)
            plt.savefig(os.path.join(svg_save_path, f'UMAP-crossreg-{mouse_groups[m]}-{m}_iter{iter}_seed-{random_seed}.svg'), format='svg')
        plt.close()
        print('done.')


# UMAP: crossreg fit TFC_cond, transform Test_B and Test_B_1wk
from umap import UMAP
umap_use_random = True
iterations = 1
save_path = PLOTS_DIR
want_svg = False
marker_size=2
tone_marker_size=10
shock_marker_size=8
marker_alpha = 1
tone_marker_alpha = 1
shock_marker_alpha = 1
plot_S_debug = False

random_seeds = [random.randint(0, 1000000) for _ in range(iterations)]
print(f'Random seeds: {random_seeds}')

for iter, random_seed in zip(range(iterations), random_seeds):
    for m in mouse_groups.keys():
        if m in ['G07', 'G15']:
            continue
        print(f'*** UMAP-crossreg: processing {m} iter {iter} random_seed {random_seed} ...', end='')
        #m = 'G15'

        S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
        S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)

        sess_TFC_cond = TFC_cond[m]
        sess_Test_B = Test_B[m]
        sess_Test_B_1wk = Test_B_1wk[m]

        S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
        S_Test_B = sess_Test_B.S[S_i_Test_B, :]
        S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]

        shock_idx = [0, len(sess_TFC_cond.shock_onsets)-1]
        tone_idx_TFC_cond = [0 , len(sess_TFC_cond.tone_onsets)-1]
        tone_idx_Test_B = [0 , len(sess_Test_B.tone_onsets)-1]
        tone_idx_Test_B_1wk = [0 , len(sess_Test_B_1wk.tone_onsets)-1]
    
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        #axes = axes.flatten()

        S_TFC_cond_n = np.nan_to_num((S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True)) / S_TFC_cond.std(axis=1, keepdims=True))
        S_Test_B_n = np.nan_to_num((S_Test_B - S_Test_B.mean(axis=1, keepdims=True)) / S_Test_B.std(axis=1, keepdims=True))
        S_Test_B_1wk_n = np.nan_to_num((S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True)) / S_Test_B_1wk.std(axis=1, keepdims=True))

        if umap_use_random:
            umap_TFC_cond = UMAP(n_components=2, n_jobs=-1)
        else:
            umap_TFC_cond = UMAP(n_components=2, random_state=42, n_jobs=1)
        embedding_TFC_cond = umap_TFC_cond.fit_transform(S_TFC_cond_n.T)
        embedding_Test_B = umap_TFC_cond.transform(S_Test_B_n.T)
        embedding_Test_B_1wk = umap_TFC_cond.transform(S_Test_B_1wk_n.T)

        for i, first_last_str in enumerate(['First', 'Last']):
            ax = axes[i, 0]
            sc = ax.scatter(embedding_TFC_cond[:, 0], embedding_TFC_cond[:, 1], c=np.arange(S_TFC_cond.shape[1]), cmap='viridis', s=marker_size, alpha=marker_alpha)
            ax.scatter(embedding_TFC_cond[sess_TFC_cond.tone_onsets[tone_idx_TFC_cond[i]]:sess_TFC_cond.tone_offsets[tone_idx_TFC_cond[i]], 0], \
                       embedding_TFC_cond[sess_TFC_cond.tone_onsets[tone_idx_TFC_cond[i]]:sess_TFC_cond.tone_offsets[tone_idx_TFC_cond[i]], 1], \
                       color='blue', label='Tone Period', s=tone_marker_size, alpha=tone_marker_alpha, marker='^')
            ax.scatter(embedding_TFC_cond[sess_TFC_cond.shock_onsets[shock_idx[i]]:sess_TFC_cond.shock_offsets[shock_idx[i]], 0], \
                       embedding_TFC_cond[sess_TFC_cond.shock_onsets[shock_idx[i]]:sess_TFC_cond.shock_offsets[shock_idx[i]], 1], \
                       color='red', label='Shock Period', s=shock_marker_size, alpha=shock_marker_alpha, marker='x')
            ax.set_title(f'TFC_cond {first_last_str}')

            ax = axes[i, 1]
            sc = ax.scatter(embedding_Test_B[:, 0], embedding_Test_B[:, 1], c=np.arange(S_Test_B.shape[1]), cmap='viridis', s=marker_size, alpha=marker_alpha)
            ax.scatter(embedding_Test_B[sess_Test_B.tone_onsets[tone_idx_Test_B[i]]:sess_Test_B.tone_offsets[tone_idx_Test_B[i]], 0], \
                    embedding_Test_B[sess_Test_B.tone_onsets[tone_idx_Test_B[i]]:sess_Test_B.tone_offsets[tone_idx_Test_B[i]], 1], \
                    color='blue', label='Tone Period', s=tone_marker_size, alpha=tone_marker_alpha, marker='^')
            ax.set_title(f'Test_B {first_last_str}')

            ax = axes[i, 2]
            sc = ax.scatter(embedding_Test_B_1wk[:, 0], embedding_Test_B_1wk[:, 1], c=np.arange(S_Test_B_1wk.shape[1]), cmap='viridis', s=marker_size, alpha=marker_alpha)
            ax.scatter(embedding_Test_B_1wk[sess_Test_B_1wk.tone_onsets[tone_idx_Test_B_1wk[i]]:sess_Test_B_1wk.tone_offsets[tone_idx_Test_B_1wk[i]], 0], \
                    embedding_Test_B_1wk[sess_Test_B_1wk.tone_onsets[tone_idx_Test_B_1wk[i]]:sess_Test_B_1wk.tone_offsets[tone_idx_Test_B_1wk[i]], 1], \
                    color='blue', label='Tone Period', s=tone_marker_size, alpha=tone_marker_alpha, marker='^')
            ax.set_title(f'Test_B_1wk {first_last_str}')

        # Create a single legend for all subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), fontsize='small')

        plt.colorbar(sc, ax=axes, label='Frame index', shrink=0.7)
        #plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        #plt.tight_layout(rect=[0, 0, 0.9, 0.9])
        plt.suptitle(f'{m} {mouse_groups[m]} UMAP embedding')
        plt.show()

        save_path = os.path.join(PLOTS_DIR, 'UMAP', 'crossreg-TFC', f'random_seed-{random_seed}')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'UMAP-crossreg-{mouse_groups[m]}-{m}_iter{iter}_seed-{random_seed}.png'), format='png', dpi=600)
        if want_svg:
            svg_save_path = os.path.join(save_path, 'svg')
            os.makedirs(svg_save_path, exist_ok=True)
            plt.savefig(os.path.join(svg_save_path, f'UMAP-crossreg-{mouse_groups[m]}-{m}_iter{iter}_seed-{random_seed}.svg'), format='svg')
        plt.close()
        print('done.')


#
# MS Copilot UMAP with concatenated sessions
#
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# UMAP
m = 'G06'
plot_S_debug = False
#sess = TFC_cond[m]
#S = sess.S

S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)

sess_TFC_cond = TFC_cond[m]
sess_Test_B = Test_B[m]
sess_Test_B_1wk = Test_B_1wk[m]

# Assuming S_TFC_cond, S_Test_B, and S_Test_B_1wk are your spike matrices
S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
S_Test_B = sess_Test_B.S[S_i_Test_B, :]
S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]

combined_S = np.hstack((S_TFC_cond, S_Test_B, S_Test_B_1wk))

# Normalize the combined matrix
S_normalized = (combined_S - combined_S.mean(axis=1, keepdims=True)) / combined_S.std(axis=1, keepdims=True)

# UMAP
#umap = UMAP(n_components=2, random_state=42, n_jobs=-1)
umap = UMAP(n_components=2, n_jobs=-1)
embedding = umap.fit_transform(S_normalized.T)

# Define the number of frames for each session
frames_TFC_cond = S_TFC_cond.shape[1]
frames_Test_B = S_Test_B.shape[1]
frames_Test_B_1wk = S_Test_B_1wk.shape[1]

# Concatenate time points for the z-axis
time_points = np.hstack((np.arange(frames_TFC_cond),
                         np.arange(frames_Test_B),
                         np.arange(frames_Test_B_1wk)))

# Plotting the UMAP embedding with different colormaps for each session in 3D
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# TFC_cond session
sc1 = ax.scatter(embedding[:frames_TFC_cond, 0], embedding[:frames_TFC_cond, 1], np.arange(frames_TFC_cond),
                 c=np.arange(frames_TFC_cond), cmap='viridis', label='TFC_cond')

# Test_B session
sc2 = ax.scatter(embedding[frames_TFC_cond:frames_TFC_cond + frames_Test_B, 0],
                 embedding[frames_TFC_cond:frames_TFC_cond + frames_Test_B, 1],
                 np.arange(frames_Test_B) + frames_TFC_cond,  # Adjust time axis
                 c=np.arange(frames_Test_B), cmap='plasma', label='Test_B')

# Test_B_1wk session
sc3 = ax.scatter(embedding[frames_TFC_cond + frames_Test_B:, 0],
                 embedding[frames_TFC_cond + frames_Test_B:, 1],
                 np.arange(frames_Test_B_1wk) + frames_TFC_cond + frames_Test_B,  # Adjust time axis
                 c=np.arange(frames_Test_B_1wk), cmap='inferno', label='Test_B_1wk')

# Create a color bar for each session
cbar1 = plt.colorbar(sc1, ax=ax, label='Frame index (TFC_cond)', shrink=0.5)
cbar2 = plt.colorbar(sc2, ax=ax, label='Frame index (Test_B)', shrink=0.5)
cbar3 = plt.colorbar(sc3, ax=ax, label='Frame index (Test_B_1wk)', shrink=0.5)

# Adjust the layout and add a title
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('Time')
plt.legend()
plt.title('Combined 3D UMAP Embedding with Time Progression')
plt.show()




#
# Odds ratio 
#

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.stats import fisher_exact

marker_size = 2
marker_alpha = 0.6
marker_size_tone = 5
marker_size_shock = 5

want_engram = False
engram_thresh = 0
auto_close = True
want_svg = False

want_gaussian_smoothing = False
smoothing_sigma = 1.5

use_crossreg = False

bin_width = 1 #second
bin_frames = bin_width * MINISCOPE_FPS
spk_cutoff = 0
upper = 1
lower = 0
binarize = True
fisher_alternative = 'greater'

odds_ratio_all = {} # keyed by stim_type, sess, then mouse
S_n_all = {}
S_bin_all = {}
tone_events_all = {}
shock_events_all = {} 
stim_types_all = {'tone': tone_events_all, 'shock': shock_events_all}

#for sess_type, sess_mice in zip(['TFC_cond', 'Test_B', 'Test_B_1wk'], [TFC_cond, Test_B, Test_B_1wk]):
for sess_type, sess_mice in zip(['TFC_cond'], [TFC_cond]):
#for sess_type, sess_mice in zip(['Test_B_1wk'], [Test_B_1wk]):
    for m in mouse_groups.keys():
        if m in ['G07', 'G15'] and (sess_type in ['Test_B', 'Test_B_1wk'] or use_crossreg):
            #print('CONTINUE')
            continue

        print(f'*** Odds ratio: processing {m} {mouse_groups[m]} ', end='')
        sess = sess_mice[m]
        sess_TFC_cond = TFC_cond[m]
        if use_crossreg:
            S_i_sess = get_S_indeces_crossreg(sess, TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
            S_i_TFC_cond = get_S_indeces_crossreg(TFC_cond[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
            #S_i_Test_B = get_S_indeces_crossreg(Test_B[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)
            #S_i_Test_B_1wk = get_S_indeces_crossreg(Test_B_1wk[m], TFC_B_B_1wk_crossreg[m], mapping_TFC_cond_Test_B_Test_B_1wk)

            S = sess.S[S_i_sess, :]
            S_TFC_cond = sess_TFC_cond.S[S_i_TFC_cond, :]
            #S_Test_B = sess_Test_B.S[S_i_Test_B, :]
            #S_Test_B_1wk = sess_Test_B_1wk.S[S_i_Test_B_1wk, :]
        else:
            S = sess.S
            S_TFC_cond = sess_TFC_cond.S
            #S_Test_B = sess_Test_B.S
            #S_Test_B_1wk = sess_Test_B.S

        # Z-scoring (ensuring PCA is centered around origin, and features are standard scaled)
        S_n = np.nan_to_num((S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True))
        S_TFC_cond_n = np.nan_to_num((S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True)) / S_TFC_cond.std(axis=1, keepdims=True))
        #S_Test_B_n = np.nan_to_num((S_Test_B - S_Test_B.mean(axis=1, keepdims=True)) / S_Test_B.std(axis=1, keepdims=True))
        #S_Test_B_1wk_n = np.nan_to_num((S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True)) / S_Test_B_1wk.std(axis=1, keepdims=True))

        # Centering only (not as good)
        #S_TFC_cond_n = np.nan_to_num(S_TFC_cond - S_TFC_cond.mean(axis=1, keepdims=True))
        #S_Test_B_n = np.nan_to_num(S_Test_B - S_Test_B.mean(axis=1, keepdims=True))
        #S_Test_B_1wk_n = np.nan_to_num(S_Test_B_1wk - S_Test_B_1wk.mean(axis=1, keepdims=True))

        if want_gaussian_smoothing:
            S_n = gaussian_filter1d(S_n, sigma=smoothing_sigma, axis=1)
            S_TFC_cond_n = gaussian_filter1d(S_TFC_cond_n, sigma=smoothing_sigma, axis=1)
            #S_Test_B_n = gaussian_filter1d(S_Test_B_n, sigma=smoothing_sigma, axis=1)
            #S_Test_B_1wk_n = gaussian_filter1d(S_Test_B_1wk_n, sigma=smoothing_sigma, axis=1) 
        
        '''
        if want_engram:
            save_path = os.path.join(PLOTS_DIR, 'Odds_ratio_engram')
        else:
            save_path = os.path.join(PLOTS_DIR, 'Odds_ratio')
        os.makedirs(save_path, exist_ok=True)
        '''

        if want_engram: # WARNING this doesn't make sense now for use_crossreg=False
            score = sp.stats.zscore(np.sum(S_TFC_cond_n,axis=1))
            engram_cells = np.where(score > engram_thresh)[0]

            S_n = S_n[engram_cells, :]
            #S_TFC_cond_n = S_TFC_cond_n[engram_cells, :]
            #S_Test_B_n = S_Test_B_n[engram_cells, :]
            #S_Test_B_1wk_n = S_Test_B_1wk_n[engram_cells, :]

        S_bin = np.zeros((S_n.shape[0], math.floor(S_n.shape[1]/bin_frames)))
        #S_TFC_cond_bin = np.zeros((S_TFC_cond_n.shape[0], math.floor(S_TFC_cond_n.shape[1]/bin_frames)))
        #S_Test_B_bin = np.zeros((S_Test_B_n.shape[0], math.floor(S_Test_B_n.shape[1]/bin_frames)))
        #S_Test_B_1wk_bin = np.zeros((S_Test_B_1wk_n.shape[0], math.floor(S_Test_B_1wk_n.shape[1]/bin_frames)))

        '''
        for S_bin, S_n in zip([S_TFC_cond_bin, S_Test_B_bin, S_Test_B_1wk_bin], [S_TFC_cond_n, S_Test_B_n, S_Test_B_1wk_n]):
            curr_frame = 0
            for i in range(S_bin.shape[1]):
                if binarize:
                    S_bin[:,i] = np.where(np.sum(S_n[:,curr_frame:curr_frame+bin_frames],1)/bin_frames >= spk_cutoff, upper, lower)
                else:
                    S_bin[:,i] = np.sum(S_n[:,curr_frame:curr_frame+bin_frames],1) / bin_frames
                curr_frame += bin_frames
        '''
        curr_frame = 0
        for i in range(S_bin.shape[1]):
            if binarize:
                S_bin[:,i] = np.where(np.sum(S_n[:,curr_frame:curr_frame+bin_frames],1)/bin_frames >= spk_cutoff, upper, lower)
            else:
                S_bin[:,i] = np.sum(S_n[:,curr_frame:curr_frame+bin_frames],1) / bin_frames
            curr_frame += bin_frames

        tone_events_sess = np.zeros(S_bin.shape[1])
        #tone_events_TFC_cond = np.zeros(S_TFC_cond_bin.shape[1])
        #tone_events_Test_B = np.zeros(S_Test_B_bin.shape[1])
        #tone_events_Test_B_1wk = np.zeros(S_Test_B_1wk_bin.shape[1])
        if sess_type == 'TFC_cond':
            shock_events_sess = np.zeros(S_bin.shape[1])

        for i, (onset, offset) in enumerate(zip(sess.tone_onsets, sess.tone_offsets)):
            onset = int(np.round(onset/bin_frames))
            offset = int(np.round(offset/bin_frames))
            tone_events_sess[onset:offset] = 1
        if sess_type == 'TFC_cond':
            for i, (onset, offset) in enumerate(zip(sess.shock_onsets, sess.shock_offsets)):
                onset = int(np.round(onset/bin_frames))
                offset = int(np.round(offset/bin_frames))            
                shock_events_sess[onset:offset] = 1

        if sess_type not in S_n_all:
            S_n_all[sess_type] = {}
            S_bin_all[sess_type] = {}
            tone_events_all[sess_type] = {}
            if sess_type == 'TFC_cond':
                shock_events_all[sess_type] = {}

        S_n_all[sess_type][m] = S_n
        S_bin_all[sess_type][m] = S_bin
        if sess_type == 'TFC_cond':
            shock_events_all[sess_type][m] = shock_events_sess
        tone_events_all[sess_type][m] = tone_events_sess

        if sess_type == 'TFC_cond':
            S_bins = [S_bin, S_bin]
            stim_events = [tone_events_sess, shock_events_sess]
            sess_types = ['TFC_cond', 'TFC_cond']
            stim_types = ['tone', 'shock']
        else:
            S_bins = [S_bin]
            stim_events = [tone_events_sess]
            sess_types = [sess_type]
            stim_types = ['tone']
        
        for S_bin, stim_event, sess_type_inner, stim_type in zip(S_bins, stim_events, sess_types, stim_types):
        #for S_bin, stim_events, sess_type, stim_type in \
        #    zip([S_bin, S_TFC_cond_bin], \
        #        [tone_events_TFC_cond, shock_events_TFC_cond], \
        #        ['TFC_cond', 'TFC_cond'], \
        #        ['tone', 'shock']):

            # Calculate odds ratio 
            odds_ratio_fisher_cells = np.zeros((S_bin.shape[0],2))
            odds_ratio_table_cells = np.zeros((S_bin.shape[0],2,2))

            for cell in range(S_bin.shape[0]):

                # Debug - don't uncomment during loop execution! do in debugger one by one
                #plt.figure(); plt.plot(S_TFC_cond_bin[cell,:],'k'); plt.plot(tone_events_TFC_cond,'r') 

                # Calculate the number of events during tone periods and non-tone periods
                stim_spikes = np.sum(S_bin[cell, :] * stim_event)
                non_stim_spikes = np.sum(S_bin[cell, :] * (1 - stim_event))

                # Calculate the number of non-events during tone periods and non-tone periods
                stim_non_spikes = np.sum((1 - S_bin[cell, :]) * stim_event)
                non_stim_non_spikes = np.sum((1 - S_bin[cell, :]) * (1 - stim_event))

                # Calculate the odds ratio
                odds_ratio_table = np.array([[stim_spikes, non_stim_spikes], [stim_non_spikes, non_stim_non_spikes]])
                res = fisher_exact(odds_ratio_table, alternative=fisher_alternative)
                odds_ratio_fisher_cells[cell]= res
                odds_ratio_table_cells[cell] = odds_ratio_table

            if stim_type not in odds_ratio_all:
                odds_ratio_all[stim_type] = {}
            if sess_type_inner not in odds_ratio_all[stim_type]:
                odds_ratio_all[stim_type][sess_type_inner] = {}
            odds_ratio_all[stim_type][sess_type_inner][m] = {'fisher' : odds_ratio_fisher_cells, 'table' : odds_ratio_table_cells}

    #
    # Finish per-mouse odds-ratio calculation loop; now process results.
    #
    if sess_type == 'TFC_cond':
        stim_types = ['tone', 'shock']
    else:
        stim_types = ['tone']
    
    for stim_type in stim_types:
        for group in mice_per_group.keys():
            print(group)
            for m in mice_per_group[group]:
                if m in ['G07', 'G15'] and (sess_type in ['Test_B', 'Test_B_1wk'] or use_crossreg):
                    continue
                print('   '+str(len(np.where(odds_ratio_all[stim_type][sess_type][m]['fisher'][:,1] < 0.05)[0])))
                import matplotlib.pyplot as plt
                import scipy.stats as stats

        # Collect the number of significant cells per group
        sig_cells_per_group = {group: [] for group in mice_per_group.keys()}
        tot_cells_per_group = {group: [] for group in mice_per_group.keys()}
        for group in mice_per_group.keys():
            for m in mice_per_group[group]:
                if m in ['G07', 'G15'] and (sess_type in ['Test_B', 'Test_B_1wk'] or use_crossreg):
                    continue
                sig_cells_per_group[group].append( \
                    len(np.where(odds_ratio_all[stim_type][sess_type][m]['fisher'][:, 1] < 0.05)[0]) / S_bin_all[sess_type][m].shape[0] \
                    )

        # Collect odds ratios for significant cells per group
        odds_ratios_per_group = {group: [] for group in mice_per_group.keys()}
        for group in mice_per_group.keys():
            for m in mice_per_group[group]:
                if m in ['G07', 'G15'] and (sess_type in ['Test_B', 'Test_B_1wk'] or use_crossreg):
                    continue
                significant_indices = np.where(odds_ratio_all[stim_type][sess_type][m]['fisher'][:, 1] < 0.05)[0]
                odds_ratios = odds_ratio_all[stim_type][sess_type][m]['fisher'][significant_indices, 0]
                odds_ratios_per_group[group].extend(odds_ratios)
        # Define colors for each group
        colors = {'mCherry': 'black', 'hM3D': 'red', 'hM4D': 'blue'}
        # Plot histograms of odds ratios for each group
        fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(15,10))#plt.figure(figsize=(15, 10))
        for group, ax in zip(mice_per_group.keys(), axs):
            valid_data = [x for x in odds_ratios_per_group[group] if np.isfinite(x)]
            ax.hist(valid_data, bins=30, alpha=0.7, color=colors[group], label=f'{group} (n={len(valid_data)})')
            ax.set_xlabel('Odds Ratio')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of Odds Ratios for {group} {stim_type} {sess_type}')
            ax.legend()
        plt.tight_layout()
        plt.show()

        # Perform 1-way ANOVA
        anova_data = [sig_cells_per_group[group] for group in mice_per_group.keys()]
        f_val, p_val = stats.f_oneway(*anova_data)
        print(f'ANOVA results {stim_type} {sess_type}: F-value = {f_val}, p-value = {p_val}')

        # Plot the bar plot with scatter points
        plt.figure(figsize=(10, 6))
        groups = list(sig_cells_per_group.keys())
        means = [np.mean(sig_cells_per_group[group]) for group in groups]
        stds = [np.std(sig_cells_per_group[group]) for group in groups]

        # Define colors for each group
        colors = {'mCherry': 'black', 'hM3D': 'red', 'hM4D': 'blue'}

        # Bar plot
        for i, group in enumerate(groups):
            plt.bar(group, means[i], yerr=stds[i], capsize=5, alpha=0.6, color=colors[group], label=f'{group} Mean ± SD')

        # Scatter points
        for i, group in enumerate(groups):
            y = sig_cells_per_group[group]
            x = np.random.normal(loc=i, scale=0.04, size=len(y))  # Add some jitter for better visualization
            plt.scatter(x, y, alpha=0.6, color='black')

        plt.xlabel('Group')
        plt.ylabel('Number of Significant Cells')
        plt.title(f'{sess_type} {stim_type}')
        plt.legend()
        plt.show()

        #
        # Plot random examples for each cell
        #
        for group in mice_per_group.keys():
            for sess_type in ['TFC_cond']:
                for m in mice_per_group[group]:
                    if m in ['G07', 'G15']:
                        continue
                    sig_odds_ratio_cells = np.where(odds_ratio_all[stim_type][sess_type][m]['fisher'][:,1] < 0.05)[0]
                    random_cells = np.random.choice(sig_odds_ratio_cells, min(len(sig_odds_ratio_cells), 9), replace=False)
                    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                    fig.suptitle(f'{m} {group} {sess_type} {stim_type} - Random Cells', fontsize=16)
                    for i, cell in enumerate(random_cells):
                        ax = axes[i // 3, i % 3]
                        ax.plot(S_bin_all[sess_type][m][cell, :], 'k', label='S_bin')
                        ax.plot(stim_types_all[stim_type][sess_type][m], 'r', label=f'{stim_type} events')
                        ax.set_title(f'Cell {cell} {stim_type} {sess_type}')
                        #ax.legend()
                    plt.tight_layout()
                    plt.show()

#######
# FINIS odds-ratio
#######


        # Plot distribution of odds ratio for TFC_cond
        plt.figure(figsize=(10, 6))
        plt.hist(odds_ratio_TFC_cond, bins=30, color='blue', alpha=0.7)
        plt.xlabel('Odds Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Odds Ratio for TFC_cond')
        plt.show()

        # Plot distribution of odds ratio for Test_B
        plt.figure(figsize=(10, 6))
        plt.hist(odds_ratio_Test_B, bins=30, color='green', alpha=0.7)
        plt.xlabel('Odds Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Odds Ratio for Test_B')
        plt.show()

        # Plot distribution of odds ratio for Test_B_1wk
        plt.figure(figsize=(10, 6))
        plt.hist(odds_ratio_Test_B_1wk, bins=30, color='red', alpha=0.7)
        plt.xlabel('Odds Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Odds Ratio for Test_B_1wk')
        plt.show()
        # Plot distribution of log of odds_ratio_TFC_cond
        log_odds_ratio_TFC_cond = np.log(odds_ratio_TFC_cond + 1e-10)  # Add a small constant to avoid log(0)
        plt.figure(figsize=(10, 6))
        plt.hist(log_odds_ratio_TFC_cond, bins=30, color='blue', alpha=0.7)
        plt.xlabel('Log Odds Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Log Odds Ratio for TFC_cond')
        plt.show()
        print("Log odds ratio for S_TFC_cond_bin with tone_events_TFC_cond:", log_odds_ratio_TFC_cond)
        
    # Calculate odds ratio
    odds_ratio_TFC_cond = np.exp(np.nanmean(np.log(S_TFC_cond_n + 1), axis=1))
    odds_ratio_Test_B = np.exp(np.nanmean(np.log(S_Test_B_n + 1), axis=1))
    odds_ratio_Test_B_1wk = np.exp(np.nanmean(np.log(S_Test_B_1wk_n + 1), axis=1))

            for only_tone_shock in [True]:#[True, False]:

                if only_tone_shock:
                    tone_shock_PCs = np.vstack([
                        np.vstack([PCs_TFC_cond[onset:offset] for onset, offset in zip(sess_TFC_cond.tone_onsets, sess_TFC_cond.tone_offsets)]),
                        np.vstack([PCs_Test_B[onset:offset] for onset, offset in zip(sess_Test_B.tone_onsets, sess_Test_B.tone_offsets)]),
                        np.vstack([PCs_Test_B_1wk[onset:offset] for onset, offset in zip(sess_Test_B_1wk.tone_onsets, sess_Test_B_1wk.tone_offsets)]),
                        np.vstack([PCs_TFC_cond[onset:offset] for onset, offset in zip(sess_TFC_cond.shock_onsets, sess_TFC_cond.shock_offsets)]),
                    ])

                    x_limits_tone_shock = (tone_shock_PCs[:, 0].min(), tone_shock_PCs[:, 0].max())
                    y_limits_tone_shock = (tone_shock_PCs[:, 1].min(), tone_shock_PCs[:, 1].max())
                    z_limits_tone_shock = (tone_shock_PCs[:, 2].min(), tone_shock_PCs[:, 2].max())

                #
                # Plot trajectories in 2D space and PCA space
                #
                # Assuming sess_TFC_cond.loc_X_miniscope_smooth and sess_TFC_cond.loc_Y_miniscope_smooth are numpy arrays
                for trajectory_type in ['Time', 'Location']:
                    for PCs, sess, sess_name in zip([PCs_TFC_cond, PCs_Test_B, PCs_Test_B_1wk], \
                                                    [sess_TFC_cond, sess_Test_B, sess_Test_B_1wk], \
                                                    ['TFC_cond', 'Test_B', 'Test_B_1wk']):
                        fig = plt.figure(figsize=(12, 6))
                        ax1 = fig.add_subplot(121)
                        ax2 = fig.add_subplot(122, projection='3d')
                        #axes = axes.flatten()
                        ax = ax1
                        x = sess.loc_X_miniscope_smooth
                        y = sess.loc_Y_miniscope_smooth

                        if m in ['G09', 'G21']: # Because Miniscope 19.avi was not recorded, see main.py
                            x = x[0:PCs.shape[0]]
                            y = y[0:PCs.shape[0]]
                        '''
                        PCs_time = np.zeros_like(PCs)
                        PCs_location = np.zeros_like(PCs)

                        for i in range(PCs.shape[0]):
                            PCs_time[i] = PCs[i] * values[i]
                            PCs_location[i] = PCs[i] * values[i]
                        '''

                        if trajectory_type == 'Time':
                            # Color by time
                            values = np.linspace(0, 1, len(x))
                            colorbar_label = 'Time (frames)'
                        elif trajectory_type == 'Location':
                            len_x = int(np.ceil(max(x) - min(x)))
                            len_y = int(np.ceil(max(y) - min(y)))
                            # Create a meshgrid of coordinates
                            coords_x = np.linspace(0, 1, len_x)
                            coords_y = np.linspace(0, 1, len_y)
                            mesh_X, mesh_Y = np.meshgrid(x, y)
                            # Define color gradients (e.g., from blue to red horizontally and green to yellow vertically)
                            R = mesh_X  # Red component varies with x
                            G = mesh_Y  # Green component varies with y
                            B = 1 - mesh_X  # Blue component inversely varies with x
                            # Combine the RGB components
                            colors = np.stack((R, G, B), axis=-1)
                            # Plot the gradient
                            plt.figure(figsize=(6, 6))
                            plt.imshow(colors, origin='lower', extent=[0, 1, 0, 1])
                            plt.title('Color-Tiled Square with Unique Colors for Each Position')
                            plt.xlabel('X-axis')
                            plt.ylabel('Y-axis')
                            plt.grid(False)
                            plt.show()

                            #values = np.sqrt(x**2 + y**2)
                            colorbar_label = 'Distance from Origin'
                        sc = ax.scatter(x, y, c=values, cmap='viridis', s=1)
                        fig.colorbar(sc, ax=ax, label=colorbar_label)
                        ax.set_xlabel('X Position')
                        ax.set_ylabel('Y Position')
                        #ax.set_title(f'Mouse trajectory {m} {mouse_groups[m]}')

                        ax = ax2
                        ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color='gray', alpha=0.5, label='Trajectory')
                        sc = ax.scatter(PCs[:, 0], PCs[:, 1], PCs[:, 2], c=values, cmap='viridis', alpha=0.6)
                        #plt.colorbar(sc, ax=ax, label='Distance from Origin')
                        if only_tone_shock:
                            ax.set_xlim(x_limits_tone_shock)
                            ax.set_ylim(y_limits_tone_shock)
                            ax.set_zlim(z_limits_tone_shock)
                        else:
                            ax.set_xlim(x_limits)
                            ax.set_ylim(y_limits)
                            ax.set_zlim(z_limits)
                        
                        ax.set_xlabel('PC1')
                        ax.set_ylabel('PC2')
                        ax.set_zlabel('PC3')
                        def on_move(event):
                            ax.view_init(elev=ax.elev, azim=ax.azim)
                            fig.canvas.draw_idle()
                            # Update the annotation with the current elevation and azimuth
                            annotation.set_text(f'Elev: {ax.elev:.1f}, Azim: {ax.azim:.1f}')
                            annotation.set_position((event.x, event.y))

                        # Create an annotation to display the elevation and azimuth
                        annotation = fig.text(0.02, 0.95, '', transform=fig.transFigure, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

                        fig.canvas.mpl_connect('motion_notify_event', on_move)

                        if debug_mode:
                            elev, azim = ax.elev, ax.azim
                            ax.view_init(elev=elev, azim=azim + azim_rotation)

                        if preset_angle_adjust:
                            ax.view_init(elev=mouse_elev_azim[m][trajectory_type][0], azim=mouse_elev_azim[m][trajectory_type][1])

                        if auto_angle_adjust:
                            ax.view_init(elev=best_elev, azim=best_azim)        
                        #ax.set_title(f'Mouse trajectory in PC space {m} {mouse_groups[m]}')
                        plt.suptitle(f'Mouse trajectory in 2D and PC space {m} {mouse_groups[m]} {sess_name} by {trajectory_type}')

                        if debug_mode and sess_name == 'TFC_cond':
                            input("Press enter to continue")

                        plt.savefig(os.path.join(save_path, f'trajectory-2Dloc_and_PCA_crossreg_full-method{method}-type-{trajectory_type}-{mouse_groups[m]}-{m}-{sess_name}.png'), format='png', dpi=600)
                        if want_svg:
                            svg_save_path = os.path.join(save_path, 'svg')
                            os.makedirs(svg_save_path, exist_ok=True)
                            plt.savefig(os.path.join(svg_save_path, f'trajectory-2Dloc_and_PCA_crossreg_full-method{method}-type-{trajectory_type}-{mouse_groups[m]}-{m}-{sess_name}.svg'), format='svg')
                        print('.', end='')
                        if auto_close:
                            plt.close()

                # 3D trajectory (e.g., PC1, PC2, PC3) for TFC_cond, Test_B, and Test_B_1wk in separate subplots
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
                axes = axes.flatten()
                for ax, PCs, title_str, sess, best_elev, best_azim in zip(axes, \
                        [PCs_TFC_cond, PCs_Test_B, PCs_Test_B_1wk], \
                        ['TFC_cond', 'Test_B', 'Test_B_1wk'], \
                        [sess_TFC_cond, sess_Test_B, sess_Test_B_1wk], \
                        [best_elev_TFC_cond, best_elev_Test_B, best_elev_Test_B_1wk], \
                        [best_azim_TFC_cond, best_azim_Test_B, best_azim_Test_B_1wk]):
                    
                    ax.scatter(PCs[0, 0], PCs[0, 1], PCs[0, 2], color='black', s=200, marker='o', label='Start', zorder=10)
                    ax.scatter(PCs[-1, 0], PCs[-1, 1], PCs[-1, 2], color='black', s=200, facecolors='none', edgecolors='black', label='End', zorder=10)

                    if not only_tone_shock:
                        sc = ax.scatter(PCs[:, 0], PCs[:, 1], PCs[:, 2], c=np.arange(PCs.shape[0]), cmap='viridis', alpha=0.6, label='Trajectory', s=marker_size)
                    ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], color='gray', alpha=0.5, label='Trajectory')

                    #sc = ax.scatter(PCs[:, 0], PCs[:, 1], PCs[:, 2], c=np.arange(len(PCs)), cmap='viridis', s=marker_size, alpha=marker_alpha)
                    #tone_scatter = ax.scatter([], [], color='blue', label='Tone Period', s=marker_size_tone, marker='^')
                    #shock_scatter = ax.scatter([], [], color='red', label='Shock Period', s=marker_size_shock, marker='x')
                    for i, (onset, offset) in enumerate(zip(sess.tone_onsets, sess.tone_offsets)):
                        if i == 0:
                            ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='darkblue', s=marker_size_tone, marker='^')
                            ax.plot(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='blue', alpha=0.6, label='Tone Trajectory', linewidth=0.5)
                        elif i == len(sess.tone_onsets) - 1:
                            ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='lightblue', s=marker_size_tone, marker='^')
                            ax.plot(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='lightblue', alpha=0.6, label='Tone Trajectory', linewidth=0.5)                            
                    # Concatenate all onset, offset PCs for sess.tone_onsets/tone_offsets
                    #tone_onset_offset_PCs = np.vstack([PCs[onset:offset] for onset, offset in zip(sess.tone_onsets, sess.tone_offsets)])
                    # Plot the concatenated tone onset/offset PCs as a single curve with a colormap
                    #ax.plot(tone_onset_offset_PCs[:, 0], tone_onset_offset_PCs[:, 1], tone_onset_offset_PCs[:, 2], color='blue', alpha=0.6, label='Tone Trajectory', linewidth=0.5)
                    #ax.scatter(tone_onset_offset_PCs[:, 0], tone_onset_offset_PCs[:, 1], tone_onset_offset_PCs[:, 2], c=np.arange(tone_onset_offset_PCs.shape[0]), cmap='cool', s=marker_size, alpha=marker_alpha)

                    if title_str == 'TFC_cond':

                        if only_tone_shock:
                            shock_PCs = np.vstack([PCs[onset:offset] for onset, offset in zip(sess.shock_onsets, sess.shock_offsets)])
                            x_limits_shock = (shock_PCs[:, 0].min(), shock_PCs[:, 0].max())
                            y_limits_shock = (shock_PCs[:, 1].min(), shock_PCs[:, 1].max())
                            z_limits_shock = (shock_PCs[:, 2].min(), shock_PCs[:, 2].max())

                        for i, (onset, offset) in enumerate(zip(sess.shock_onsets, sess.shock_offsets)):
                            if i == 0:
                                ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='darkred', s=marker_size_shock, marker='x')
                                ax.plot(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='darkred', alpha=0.6, label='Shock Trajectory', linewidth=0.5)
                            elif i == len(sess.shock_onsets) - 1:
                                ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='lightcoral', s=marker_size_shock, marker='x')
                                ax.plot(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='lightcoral', alpha=0.6, label='Shock Trajectory', linewidth=0.5)
                            #else:
                            #    ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], edgecolor='red', facecolor='none', s=marker_size_shock, marker='x')

                    
                    #if title_str == 'TFC_cond':
                    #    for onset, offset in zip(sess.shock_onsets, sess.shock_offsets):
                    #        ax.scatter(PCs[onset:offset, 0], PCs[onset:offset, 1], PCs[onset:offset, 2], color='red', s=marker_size_shock, marker='x')
                    
                    if only_tone_shock:
                        ax.set_xlim(x_limits_tone_shock)
                        ax.set_ylim(y_limits_tone_shock)
                        ax.set_zlim(z_limits_tone_shock)
                    else:
                        ax.set_xlim(x_limits)
                        ax.set_ylim(y_limits)
                        ax.set_zlim(z_limits)
                    
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_zlabel('PC3')
                    elev, azim = ax.elev, ax.azim
                    ax.view_init(elev=elev, azim=azim + azim_rotation)
                    if auto_angle_adjust:
                        ax.view_init(elev=best_elev, azim=best_azim)        
                    ax.set_title(title_str)
                    #ax.legend(handles=[sc.legend_elements()[0][0], tone_scatter, shock_scatter], labels=['Trajectory', 'Tone Period', 'Shock Period'], fontsize='small')

                plt.suptitle(f'{m} {mouse_groups[m]} PCA crossreg full time course (method {method})')
                if not only_tone_shock:
                    plt.colorbar(sc, ax=axes, label='Time (frames)', shrink=0.5)
                plt.show()

                if want_engram:
                    save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_full_engram_toneshock')
                else:
                    save_path = os.path.join(PLOTS_DIR, 'PCA_crossreg_full_toneshock')
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'PCA_crossreg_full-want_engram-{want_engram}-only_tone_shock_{only_tone_shock}-method{method}-{mouse_groups[m]}-{m}.png'), format='png', dpi=600)
                if want_svg:
                    svg_save_path = os.path.join(save_path, 'svg')
                    os.makedirs(svg_save_path, exist_ok=True)
                    plt.savefig(os.path.join(svg_save_path, f'PCA_crossreg_full-want_engram-{want_engram}-only_tone_shock_{only_tone_shock}-method{method}-{mouse_groups[m]}-{m}.svg'), format='svg')
                if auto_close:
                    plt.close()
                print('.', end='')

    print('done.')