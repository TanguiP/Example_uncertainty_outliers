import numpy as np
import xarray as xr

def get_Pi_bin(obj, bins = [0]):
    """Compute intensity distribution
    - obj (xarray) with a dimension "time"
    - bins (numpy): 1D array with ascending intensities
    """
    def get_pi_bin_a(array_1D_in, bins = [0]):
        return xr.DataArray(np.histogram(array_1D_in, bins, weights=array_1D_in)[0],
                            dims=['Pb'])
    
    # Apply get_pi_bin_a to the xarray dataset
    return xr.apply_ufunc(
        get_pi_bin_a, 
        obj,
        input_core_dims=[["time"]],  # list with one entry per arg
        output_core_dims=[["Pb"]],  # returned data has one dimension
        exclude_dims=set(("time",)),  # dimensions allowed to change size. Must be a set!
        vectorize=True,  # loop over non-core dims
        kwargs={"bins": bins}
    )

def mean_std(obj, dic_comp):
    """ Return the uncertainty without the identified outliers
    - obj (xarray) with a dimension "comp"
    - dic_comp (dict): dictionnary containning the position of the comparisons with each product within the dim "comp" of the xarray
    """
    
    # Function to be applied along the dimension comp
    def get_mean_cond(array_1D_in, dic_comp = None):     
        if np.isnan(array_1D_in).sum() == len(array_1D_in):
            dp_mean4 = np.nan
        else:
            #print(array_1D_in)
            arr_in = np.array(array_1D_in)
            #outlier: a median (not a mean) higher than the 75th percentile of all differences
            obs_med = np.array([np.nanmedian(arr_in[dic_comp[obs_i]]) for obs_i in dic_comp.keys()])
            
            # Nb of obs
            nb_obs = np.count_nonzero(~np.isnan(obs_med))            
            nb_comp = np.count_nonzero(~np.isnan(arr_in))
            bd_std = 1 - (nb_obs-1.5)/nb_comp
            k_std = np.percentile(arr_in[~np.isnan(arr_in)], bd_std*100)
            # Exclude products higher 
            obs_k_in = np.array(list(dic_comp.keys()))[obs_med<k_std]
            # if only comparison between the k bests
            k_in = len(obs_k_in)
            dic_comp_k = np.array([dic_comp[obs_i] for obs_i in obs_k_in])
            comp_k = [np.intersect1d(dic_comp_k[k_obs],dic_comp_k[k_other])[0]
                      for k_obs in range(0,k_in-1) for k_other in range(k_obs+1,k_in)]
            #print(comp_k)
            dp_mean4 = np.array(np.nanmax(arr_in[comp_k]))
        return dp_mean4

    # Apply get_mean_cond to the xarray dataset
    return xr.apply_ufunc(
        get_mean_cond, 
        obj,
        input_core_dims=[["comp"]],  # list with one entry per arg
        #output_core_dims=[["ev_nb_dur"]],  # returned data has one dimension
        exclude_dims=set(("comp",)),  # dimensions allowed to change size. Must be a set!
        vectorize=True,  # loop over non-core dims
        kwargs={"dic_comp":dic_comp},
    )

def mean_std_obs(obj, dic_comp):  
    """ Return the identified outliers
    - obj (xarray) with a dimension "comp"
    - dic_comp (dict): dictionnary containning the position of the comparisons with each product within the dim "comp" of the xarray
    """
    #bd_std = (1-8/29)*100
    def get_mean_cond_inf(array_1D_in, dic_comp = None):
        if np.isnan(array_1D_in).sum() == len(array_1D_in):
            out_obs = [False for i in range(8)]
        else:
            # Compute the mean for each observation
            array_1D_in = np.array(array_1D_in)
            obs_med = np.array([np.nanmedian(array_1D_in[dic_comp[obs_i]]) for obs_i in dic_comp.keys()])
            
            # Nb of obs
            nb_obs = np.count_nonzero(~np.isnan(obs_med))
            nb_comp = np.count_nonzero(~np.isnan(array_1D_in))
            bd_std = 1 - (nb_obs-1.5)/nb_comp
            k_std = np.percentile(array_1D_in[~np.isnan(array_1D_in)], bd_std*100)
            # Return an array with exclude products
            out_obs = [obs_med>=k_std]
        return np.where(out_obs,np.array(list(dic_comp.keys())),np.zeros(8)) 

    # note: apply always moves core dimensions to the end
    xr_out = xr.apply_ufunc(
        get_mean_cond_inf,
        obj,
        input_core_dims=[["comp"]],  # list with one entry per arg
        output_core_dims=[["obs_opt"]],  # returned data has one dimension
        exclude_dims=set(("comp",)),  # dimensions allowed to change size. Must be a set!
        vectorize=True,  # loop over non-core dims
        kwargs={"dic_comp":dic_comp})
    return xr_out 