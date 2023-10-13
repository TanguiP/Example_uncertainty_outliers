"""
Code written by Tangui PICART to serve as an example for article "Uncertainty and outliers in high-resolution gridded precipitation products over eastern North America"
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import func_uncertainty as func
   
# ---------------------------------------------------------------------------------
# Data collection and setting 
# ---------------------------------------------------------------------------------
print('- Data collection and setting')

# Create an array containning the combinaison of observations name
obs_name = ['CMORPH','IMERG','MSWEP','TMPA','GSMAP','STAGE4','ERA5','PERSIANN']
nb_obs = len(obs_name)
comp_name = np.array([obs_name[i]+"-"+obs_name[j] for i in range(nb_obs-1) for j in range(i+1,nb_obs) ])

# Number of bins computed
bins_I = np.logspace(np.log10(0.125),3,21)

# Thresholds
k_1mmd = 0.125

# ---------------------------------------------------------------------------------
# Import data, apply k and compute climat variables 
# ---------------------------------------------------------------------------------
# Apply the threshold to the dataset
Pi = xr.open_dataset("/pampa/picart/P_201810_git.nc")
Pi = Pi.where(Pi>=k_1mmd).fillna(0).where(Pi)

# Compute climatic data
Pc = [Pi.pr.mean("time").rename("Pm"),(Pi.pr>0).mean("time").rename("F")]
Pc = xr.merge([Pc[0], Pc[1], (Pc[0]/Pc[1]).rename("I")]) 

Pc["Pm"].attrs["units"] = "mm/3h"
Pc["F"].attrs["units"] = "-"
Pc["I"].attrs["units"] = "mm/3h"
Pc["Pm"].attrs["long_name"] = "Mean precipitation"
Pc["F"].attrs["long_name"] = "Frequency"
Pc["I"].attrs["long_name"] = "Intensity"

# Compute intensity distribution
len_time = Pi.count("time").pr
Pb = func.get_Pi_bin(Pi.pr,bins=bins_I).rename("Pb_a")/len_time

Pb['Pb'] = range(0,len(bins_I)-1)
Pb["Pb"].attrs["units"] = "mm"
Pb["Pb"].attrs["long_name"] = "Precipitation accumulation per bin"

# --------------------------------------------------------------------------------------------------------------------------
# Computing the differences
# --------------------------------------------------------------------------------------------------------------------------
print("- Comparison of the products")

# Mean precipitation diff.
dPm = [abs( (Pc.sel(obs=obs_i).Pm-Pc.sel(obs=obs_j).Pm) )
       for i,obs_i in enumerate(obs_name[0:-1]) for j,obs_j in enumerate(obs_name[i+1:]) ]

# IF decomposition diff.   
dI = [Pc.sel(obs=[obs_i,obs_j]).F.min(dim="obs") * abs(Pc.sel(obs=obs_i).I - Pc.sel(obs=obs_j).I)
           for i,obs_i in enumerate(obs_name[0:-1]) for j,obs_j in enumerate(obs_name[i+1:]) ]

dF = [Pc.sel(obs=[obs_i,obs_j]).I.min(dim="obs") * abs(Pc.sel(obs=obs_i).F - Pc.sel(obs=obs_j).F)
           for i,obs_i in enumerate(obs_name[0:-1]) for j,obs_j in enumerate(obs_name[i+1:]) ]

dp = [abs((Pc.sel(obs=obs_i).I - Pc.sel(obs=[obs_i,obs_j]).I.min(dim="obs"))*
          (Pc.sel(obs=obs_i).F - Pc.sel(obs=[obs_i,obs_j]).F.min(dim="obs"))-
          (Pc.sel(obs=obs_j).I - Pc.sel(obs=[obs_i,obs_j]).I.min(dim="obs"))*
          (Pc.sel(obs=obs_j).F - Pc.sel(obs=[obs_i,obs_j]).F.min(dim="obs")))
         for i,obs_i in enumerate(obs_name[0:-1]) for j,obs_j in enumerate(obs_name[i+1:]) ]

# Mean int. distribution  diff.  
dPb = [abs( Pb.sel(obs=obs_i) - Pb.sel(obs=obs_j)).sum("Pb",skipna=True)
       for i,obs_i in enumerate(obs_name[0:-1]) for j,obs_j in enumerate(obs_name[i+1:]) ]

# Instentaneous precipitation diff.  
dPa = [abs(Pi.sel(obs=obs_i).pr-Pi.sel(obs=obs_j).pr).mean(dim="time", skipna=True) 
       for i,obs_i in enumerate(obs_name[0:-1]) for j,obs_j in enumerate(obs_name[i+1:]) ]

# Ratio to get the relative diff.
dR = [(Pc.sel(obs=obs_i).Pm+Pc.sel(obs=obs_j).Pm).where(Pc.sel(obs=obs_i).Pm).where(Pc.sel(obs=obs_j).Pm)
      for i,obs_i in enumerate(obs_name[0:-1]) for j,obs_j in enumerate(obs_name[i+1:]) ]

# Concatenate the previous quantities within a dataset    
dPm = xr.concat(dPm, dim = "comp")
dPb = xr.concat(dPb, dim = "comp")
dPa = xr.concat(dPa, dim = "comp")
dR = xr.concat(dR, dim = "comp")    

dI = xr.concat(dI, dim = "comp")
dF = xr.concat(dF, dim = "comp")
dp = xr.concat(dp, dim = "comp")
dPif = dI+dF+dp

# Set the coordonate
dPm['comp'] = comp_name
dPb['comp'] = comp_name    
dPif['comp'] = comp_name  
dPa['comp'] = comp_name
dR['comp'] = comp_name    

# Create the dataset before saving
dP_abs = xr.merge([dPm.rename("dPm"), dPb.rename("dPb"), dPif.rename("dPif"), dPa.rename("dPa"), dR.rename("dR")])
dP_abs = dP_abs.reindex(lat=list(reversed(dP_abs.lat)))
    
# Compute the relative difference
diff_names = ["dPm","dPb","dPif","dPa"]
dP_rel = dP_abs[diff_names]*100/dP_abs.dR

# --------------------------------------------------------------------------------------------------------------------------
# Parameters for the identification of outliers
# --------------------------------------------------------------------------------------------------------------------------
comp_name_in = dP_abs.comp.values

# Table containing the indices of each comparison per observation
tabl_comp = np.zeros((nb_obs,nb_obs))
for i, obs_i in enumerate(obs_name):
    for j, obs_j in enumerate(obs_name):
        if obs_i!= obs_j: tabl_comp[i,j] = [i for i, s in enumerate(comp_name_in) if ((obs_i in s) and (obs_j in s))][0]

dic_comp = []
for i, obs_i in enumerate(obs_name):
    tab_obs = []
    for j, obs_j in enumerate(obs_name):
        if obs_i!= obs_j: tab_obs.append([int(i) for i, s in enumerate(comp_name_in) if ((obs_i in s) and (obs_j in s))][0])
    dic_comp.append(tab_obs)
dic_comp = dict(zip(obs_name,dic_comp))

# --------------------------------------------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------------------------------------------
# Metric to show
diff_name = diff_names[3]

# Compute the observational uncertainty
print("- Identify outliers and compute the uncertainty")
mean4_dP  = func.mean_std(dP_rel[diff_name],dic_comp) #.sel(k=k_1mmd)
mean4_obs = func.mean_std_obs(dP_rel[diff_name],dic_comp) #.sel(k=k_1mmd)

# Local graphic parameters
cmap = matplotlib.cm.get_cmap('viridis_r') 
bounds = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
ticks =  [10,20,30,40,50,60,70,80,90]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend='both')
datacrs = ccrs.PlateCarree()
extent = [262-360,265-360,32,35]

### Figure: comparison of the regular and new  uncertainty 
fig, axes = plt.subplots(1,2,subplot_kw={'projection': datacrs},figsize=(14,7))
# Regular
axes[0].set_title("Uncertainty (max)")
im = axes[0].imshow(dP_rel[diff_name].max("comp"), transform = datacrs, extent = extent, aspect = 'equal', norm = norm,cmap=cmap)
# New
axes[1].set_title("Uncertainty (withour ourliers)")
im = axes[1].imshow(mean4_dP, transform = datacrs, extent = extent, aspect = 'equal', norm = norm,cmap=cmap)
# Set the colorbar
cbar = fig.colorbar(im, ax=axes[1],
                    orientation='vertical',fraction=0.037, pad=0.025,ticks=ticks)
cbar.set_label(r'obs. uncertainty [%]')
# Graphical setting
for ax in axes:
    ax.set_xticks(dP_rel.lon[::2]-360, crs=datacrs)  
    ax.set_yticks(dP_rel.lat[::2], crs=datacrs)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=1)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none')
    ax.gridlines(color='k', linewidth=.2, crs=datacrs)
fig.tight_layout()
plt.show()

### Figure: Barchat of outliers 
fig, ax = plt.subplots(figsize=(7,6),constrained_layout=True)
# Percentage of grid point on which products are considered
nb_pt = len(dP_abs.lat)*len(dP_abs.lon)
ds_obs_opt = [mean4_obs.where(mean4_obs == obs_i).count()/nb_pt*100 for obs_i in obs_name]
for o_i,obs_i in enumerate(obs_name):
    ax.bar(o_i, ds_obs_opt[o_i])
ax.set_xticks(np.arange(0,len(obs_name)))
ax.set_xticklabels(obs_name, rotation=45, ha="right", position=(0.3,0),rotation_mode='anchor')
ax.set_ylim(0,100) 
ax.set_ylabel(r'Outlier rate [%]') 
plt.show()
