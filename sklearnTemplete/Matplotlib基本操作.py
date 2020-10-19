#!/usr/bin/env python
# coding: utf-8

# ### 图像设置

# In[ ]:


import matplotli.pyplot as plt

#figure
fig = plt.figure( )
fig2 = plt.figure( figsize = 
plt.figaspect(2,0) )

#axes
fig.add_axes( )
ax1 = fig.add_subplot( 221 ) 
#row-col-num
ax3 = fig.add_subplot( 212 )
fig3, axes = plt.subplots(
nrows=2, ncols=2)
fig4, axes2 = plt.subplots( ncols=3 )


# ### 数据录入

# In[ ]:




import numpy as np
#1D data
x = np.linspace(0, 10, 100)
y = np.cos(x)
z = np.sin(x)

#2D data or images
data = 2* np.random.random((10,10))
data2 = 3 * np.random.random((10,10))
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1-X**2+Y
V = 1+X-Y**2
from Matplotlib.cbook 
import get_sample_data
img = np.load(get_sample_data
('axes_grid/bivariate_normal.npy'))


# ### 图像绘制

# In[ ]:




#1D data
fig, ax = plt.subplots()
lines = ax.plot(x,y) 
#Draw points with lines or makers connecting them
ax.scatter(x,y)
#Draw unconnected points, scaled or colored
axes[0,0].bar([1,2,3],[3,4,5]) 
#Plot vertical rectangles
axes[1,0].barh([0.5,1,2.5], [0,1,2]) 
#Plot horizontal rectangles
axes[1,1].axhline(0.45) 
#Draw a horizontal line across axes
axes[0,1].axvline(0.65) 
#Draw a vertical line across axes
ax.fill(x,y,color='blue') 
#Draw filled polygons
ax.fill_between(x,y,color='yellow') 
#Fill between y-values and o

#2D data or images
fig, ax = plt.subplots( ) 
#Colormapped or RGB arrays
im = ax. imshow(img, cmap='gist_earth', 
interpolation='nearest',vmin=-2,vmax=2)
axes2[0].pcolor(data2) 
#Pseudocolor plot of 2D array
axes2[0].pcolormesh(data) 
#Pseudocolor plot of 2D array
CS = plt.contour(Y,X,U)
#Plot contours
axes2[2].contourf(data1) 
#Plot filled contours
axes2[2] = ax.clabel(CS)
#Label a contour plot

#Vector Field
axes[0,1].arrow(0,0,0.5,0.5) 
#Add an arrow to the axes
axes[1,1].quiver(y,z) 
#Plot a 2D field of arrows
axes[0,1].streamplot(X,Y,U,V) 
#plot a 2D field of arrows

#Data Distributions
ax1.hist(y) #Plot a histogram
ax3.boxplot(y) 
#make a box and whisker plot
ax3.violinplot(z) 
#make a violin plot


# ### 自定义

# In[ ]:




#colors
plt.plot(x, x, x, x**2, x, x**3)
ax.plot(x, y, alpha = 0.4)
ax.plot(x, y, c='k')
fig.colorbar(im, orientation='horizontal')
im = ax.imshow(img, cmap='seismic')

#markers
fig, ax = plt.subplots()
ax.scatter(x,y,marker='.')
ax.plot(x,y,marker='o')

#linestyles
plt.plot(x,y,linewidth=4.0)
plt.plot(x,y,ls='solid')
plt.plot(x,y,ls='--')
plt.plot(x,y,'--',x**2,y**2,'-.')
plt.setp(lines, color='r',linewidth=4.0)

#Text & Annotations
ax.text(1,-2.1, 
'Example Graph', style='italic')
ax.annotate("Sine", xy=(8,0),
xycoords='data',xytext=(10.5,0), 
textcoords='data', arrowprops=
dict(arrowstyle="->",connectionstyle="arc3"),)

#mathtext
plt.title(r'$sigma_i=15$',
 fontsize=20)

#Limits,Legends&Layouts
#Limites&Autoscaling
ax.margins(x=0.0, y=0.1) 
#Add padding to a plot
ax.axis('equal') 
#Set the aspect ratio of the plot to 1
ax.set(xlim=[0,10.5],ylim=[-1.5,1.5]) 
#Set limits for x- and y-axis
ax.set_xlim(0,10.5) 
#Set limits for x-axis

#Legends
ax.set(title='An Example Axes',
ylabel='Y-Axis',xlabel='X-Axis') 
#Set a title and x- and y-axis labels
ax.legend(loc='best') 
#No overlapping plot elements

#Ticks
ax.xaxis.set(ticks=range(1,5),
ticklabels=[3,100,-12,"foo"]) 
#Manually set x-ticks
ax.tick_params(axis='y',
direction ='inout',length=10) 
#Make y-ticks longer and go in and out

#Subplot Spacing
fig3.subplots_adjust(wspace=0.5, 
hspace=0.3,left=0.125,right=0.9,
top=0.9,bottom=0.1) 
#Adjust the spacing between subplots
fig.tight_layout() 
#Fit subplots in to the figure area

#Axis Spines
ax1.spines['top'].set_visible(False) 
#Make the top axis line for a plot invisible
ax1.spines['bottom'].set_position(('outward',10)) 
#Move the bottom axis line outward


# ### 显示与保存

# In[ ]:




#save plot
plt.savefig('foo.png')
plt.savefig('foo.png', transparent=True)

#show plot
plt.show()

#close &clear
plt.cla() #clear an axis
plt.clf() #clear the entire figure
plt.close() #close a window

