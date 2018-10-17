import os
import colorsys
import pickle

import numpy                as np
import matplotlib.pyplot    as plt

from PIL                   import Image
from math                  import sqrt
from scipy.interpolate     import interp1d
from sklearn.cluster       import KMeans

def euclidean(p1, p2):
    """ Euclidean distance between two points (tuples)
    
    Parameters:
    -----------
    p1 / p2: tuple
        Tuple of coordinates (rgb)
        
    Returns:
    --------
    float
        Distance between two points
        
    """
    return sqrt(sum([(p1[i] - p2[i]) ** 2 for i in range(0, 3)]))

def get_points(image_path):
    """ Returns all points in a picture based 
    on their rgb values
    
    Parameters:
    -----------
    image_path: string
        Path to the image
        
    Returns:
    --------
    points : list of strings
        All points (of color; rgb) in the picture
        
    """
    
    # Create Points
    img = Image.open(image_path)
    img.thumbnail((200, 200))
    width, height = img.size

    points = []

    for count, color in img.getcolors(width * height):
        for i in range(count): 
            points.append(list(color))
            
    return points

def get_common_colors(image_path):
    """ Extracts the top 5 most frequent colors 
    that are in an image using the k-Means algorithm
    to cluster the colors. The points are based on RGB
    which allows for a 3D space to be clustered. 
    
    Clusters are formed and the mediod in each cluster is
    chosen as the representative color for that cluster. 
    
    Parameters:
    -----------
    image_path: string
        Path to the image
        
    Returns:
    --------
    colors : list of strings
        list of most common colors in the picture (hex)
    
    """
    # Get points
    points = get_points(image_path)
    
    # Calculate Clusters
    kmeans = KMeans(n_clusters=5, random_state=0).fit(points)
    centers = kmeans.cluster_centers_
    
    # Calculate for each cluster the mediod
    # that point is then representative of the most common color
    # for that cluster
    colors = []
    
    for center in centers:
        smallest_distance = 100000
        closest_point = []
        
        for point in points:
            temp_dist = euclidean(center, point)
            
            if temp_dist < smallest_distance:
                closest_point = point
                smallest_distance = temp_dist

        colors.append('#%02x%02x%02x' % tuple(closest_point))

    return colors

def get_hsv(hexrgb):
    """ convert rgb to hsv for easier plotting
    """
    hexrgb = hexrgb.lstrip("#")  # in case you have Web color specs
    r, g, b = (int(hexrgb[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)

def plot_color(file, savefig=False, dpi=300, file_name='color.png', smoothen=False):
    """ Plot frequent colors on a radial plot
    
    Based on: https://github.com/NicholasARossi/color_me_impressed
    
    Parameters:
    -----------
    file : pkl
        pkl with points (rgb) of (an) image(s)
    savefig : boolean, default False
        Whether or not to save the figure in 
        the current working directory
    dpi : int, default 300
        The dpi of the image to save
    file_name : str, default 'color.png'
        The file name of the image if you want
        to save it
    smoothen : boolean, default False
        Whether or not to smoothen the bars.
        NOTE: If True, then loading will take much 
        longer since smoothing means creating 
        significantly more points.
    """
    
    # Open list of colors
    with open(file, 'rb') as f:
        storage = pickle.load(f)

    # hsv lends itself better than rgb for
    # radial plots and is therefore converted to hsv
    hlist = []
    for group in storage:
        for color in group:
            h,s,v=get_hsv(color)
            if s>0 and v>0:
                hlist.append(h)
    
    if smoothen:
        # Create initial bins
        n=100
        bins = np.arange(-0.01,1.01,0.01)

        probs, bons = np.histogram(hlist, normed=1, bins=bins)
        vect = np.linspace(0.0, 2 * np.pi, n, endpoint=False)

        # Smoothen plot 
        f2 = interp1d(vect, probs[1:], kind='cubic')
        xnew = np.linspace(min(vect), max(vect), 10000, endpoint=False)
        radii = [x if x > 0 else 0 for x in f2(xnew)] # No negatives

        # Get width and x values based on 10000 bins (for smooth plot)
        width = (2 * np.pi) / 10000
        theta = np.linspace(0.0, 2 * np.pi, 10000, endpoint=False)
    else:
        n=100
        radii, bons = np.histogram(hlist, normed=1, bins=n)
        theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        width = (2 * np.pi) / n

    # Create plot
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, radii, width=width, bottom=2,linewidth=0)

    # Use custom colors and opacity
    for r, bar in zip(theta, bars):
        bar.set_facecolor(colorsys.hsv_to_rgb(r/( 2 * np.pi), 1, 1))
        bar.set_alpha(0.75)

    # Small Circle
    bars_circle = ax.bar(theta, [0.1 for _ in theta], width=width, bottom=1.8,linewidth=0)

    # Custom colors small circle
    for r, bar in zip(theta, bars_circle):
        bar.set_facecolor(colorsys.hsv_to_rgb(r/( 2 * np.pi), 1, 1))

    plt.axis('off')

    # Removing as much white space as possible
    plt.tight_layout(pad=0)
               
    if savefig:
        plt.savefig(file_name,dpi=dpi,bbox_inches='tight', pad_inches=0)
    