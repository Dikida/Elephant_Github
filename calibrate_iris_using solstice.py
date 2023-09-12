# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:08:43 2023

@author: sndlovu
"""

import math
import ephem
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import convolve
from scipy.interpolate import interp1d
from solsticemuv import  get_sorce_solstice_muv_psf
from get_solstice_spectrum import  get_solstice_spectrum
import netCDF4 as nc
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")





def is_leap(year):
    """
    Determine if a year is a leap year.
    """
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        return True
    return False

def julian_day(month,day,year,hour):
    """
    Calculate the Julian day for a given date.
    """
    # January and February are counted as months 13 and 14 of the previous year
    if month <= 2:
        year -= 1
        month += 12

    # Calculate the Julian day
    a = year // 100
    b = a // 4
    c = 2 - a + b
    e = 365.25 * (year + 4716)
    f = 30.6001 * (month + 1)
    jd = c + day + e + f - 1524.5

    return jd

def sun_distance(julian_day):
    """
    Calculate the sun distance for a given Julian day.
    """
    # Mean distance from Earth to Sun (in astronomical units, AU)
    mean_distance = 1.496e+8

    # Eccentricity of Earth's orbit
    eccentricity = 0.0167

    # Angle (in radians) between Earth's perihelion and the vernal equinox
    perihelion_angle = math.radians(282.9404)

    # Calculate the sun distance using Kepler's equation
    true_anomaly = math.radians(360 * (julian_day - 1) / 365.2422)  # Mean anomaly
    equation_of_center = (2 * eccentricity * math.sin(true_anomaly + perihelion_angle) +
                         1.25 * eccentricity**2 * math.sin(2 * true_anomaly + 2 * perihelion_angle))
    sun_distance = mean_distance * (1 - eccentricity**2) / (1 + eccentricity * math.cos(true_anomaly + perihelion_angle + equation_of_center))

    return sun_distance
# Read FITS data
file = 'C:/Users/sNdlovu/Desktop/New folder/IRISMosaic_20140317_MgIIh.fits'
hdulist = fits.open(file)
header = hdulist[0].header
header1 = hdulist[1].header
data = hdulist[1].data
hdulist.close()


# Extract wavelength information
w0 = header['CRVAL3']
dw = header['CDELT3']
w = (np.arange(101)-50)*dw + w0 # The units here is Angstroms
w = w/10.0 # Converting Angstroms into nm
w0=w0/10.0 
it = header['CRPIX2']

# Adjust data for detector gain and downsample if necessary
if header['CRPIX2'] == 1503:
    data = data/2
    iris_date = header['CRPIX2']
    year = iris_date//10000
    month =(iris_date//100)%100
    day = iris_date % 100
    jd = julian_day(month, day, year, 12)
    datestr = iris_date[0:10]
    data = data*sun_distance(julian_day)**2




iris_date = header['CRPIX2']
# iris_date = str(iris_date)  # Convert iris_date to a string if it's not already
# year = int(iris_date[0:4])  # Use string slicing to extract the first 4 characters as the year

# #year = int(iris_date[0:4])
# month = int(iris_date[5:7])
# day = int(iris_date[8:10])
ris_date = header['CRPIX2']
year = iris_date//10000
month =(iris_date//100)%100
day = iris_date % 100
jd =  julian_day(month, day, year, 12)   # Julian day 172.0 for summer solstice (e.g., June 21st)

latitude = 30.0  # Latitude of the location
longitude = -90.0  # Longitude of the location

# Extract solstice spectrum and interpolate onto iris wavelength grid
# sol = get_solstice_spectrum(jd/365.25)
# # Filter sol array based on wavelength range
# g = np.where((sol_wave >= np.min(w)) & (sol_wave <= np.max(w)))
# sol_wave = sol['wavelength'][g]
# sol_irr = sol['irradiance'][g]
# f = interp1d(sol_wave, sol_irr, kind='linear', bounds_error=False, fill_value=0.0)
# sol_irr_interp = f(w)
# Extract solstice spectrum and interpolate onto iris wavelength grid
# Extract solstice spectrum and interpolate onto iris wavelength grid
# Extract solstice spectrum and interpolate onto iris wavelength grid
# Extract solstice spectrum and interpolate onto iris wavelength grid
# Extract solstice spectrum and interpolate onto iris wavelength grid

# Extract solstice spectrum and interpolate onto iris wavelength grid
sol_wave, sol_irr = get_solstice_spectrum(jd/365.25)
#sol_wave = sol[0]  # Access the first element (wavelength)
#sol_irr =np.squeeze( sol[1])  # Access the second element (irradiance)

# Print size and content of sol_irr array for debugging
print("sol_irr size:", sol_irr.size)
print("sol_irr content:", sol_irr)

# Check if sol_irr is empty
if sol_irr.size == 0:
    print("No data found in the sol_irr array.")
    # Handle the error or exit gracefully
else:
    # Filter sol arrays based on wavelength range
    g = np.where((sol_wave >= np.min(w)) & (sol_wave <= np.max(w)))
    g=g[0]

    sol_wave_filtered = sol_wave[g]
    sol_irr_filtered = sol_irr[g]

    # Print sizes of filtered arrays for debugging
    print("sol_wave_filtered size:", sol_wave_filtered.size)
    print("sol_irr_filtered size:", sol_irr_filtered.size)

    # Check if filtered arrays are empty
    if sol_wave_filtered.size == 0 or sol_irr_filtered.size == 0:
        print("No data found within the specified wavelength range.")
        # Handle the error or exit gracefully
    else:
        # Interpolate onto iris wavelength grid
        f = interp1d(sol_wave_filtered, sol_irr_filtered, kind='linear', bounds_error=False, fill_value=0.0)
        sol_irr_interp = f(w)


def gaussian(x, mu, sigma):
    # Calculate the exponential term of the Gaussian function
    # print(mu)
    # print(sigma)
    # print(x.size)
  
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    print('size of exponent: ', exponent.size)
    # Calculate the result of the Gaussian function
    #result = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(exponent)
    #result = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)
    result = np.exp(exponent)
   
    return result

# # Plot IRIS AVG Profile
plt.plot(w, data/np.max(data), label='IRIS AVG Profile', color='black')

# # Plot SOLSTICE L3
plt.plot(sol_wave_filtered, sol_irr_filtered/np.max( sol_irr_filtered),'D', label='SOLSTICE L3', color='purple', markersize=1.5)
#plt.show()


# Update SOLSTICE PSF
psf = get_sorce_solstice_muv_psf()
sol_psf = gaussian(w - w0/10.0, psf[1], psf[2]) > 0

# Update SOLSTICE PSF
psf = get_sorce_solstice_muv_psf()
sol_psf = gaussian(w , w0, psf[2]) 
plt.plot(w, sol_psf, '--', label='SOLSTICE PSF', color='dodgerblue', linewidth=3)

# Convolve IRIS AVG Profile with SOLSTICE PSF
c_iris = convolve(data/np.max(data), sol_psf, mode='same', method='direct')

# # Plot convolved IRIS AVG Profile
plt.plot(w, c_iris/np.max(c_iris), label='IRIS AVG Profile Convolved', color='red')



#Set labels and title
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized signal')
datestr = "2023-04-19" # Define the date string here
plt.title('Mg II h ' + datestr)

# Add legend
plt.legend()
plt.show()

# # Define Gaussian kernel
# sigma = 2.0  # Adjust sigma value as needed
# gaussian_kernel = np.exp(-(w - w0)**2 / (2 * sigma**2))

# # # Normalize Gaussian kernel
# gaussian_kernel /= np.max(gaussian_kernel)

# # # Convolve IRIS AVG Profile with Gaussian kernel
# c_iris = convolve(data/np.max(data), gaussian_kernel, mode='same', method='direct')
# #keep = np.where((w > 280.5) and (w < 280.2)))
# keep = np.where(np.logical_and(w > 280.2, w < 280.5))


# keep = keep[0]
# # # Plot convolved IRIS AVG Profile
# plt.plot(w[keep], c_iris[keep], label='IRIS AVG Profile Convolved', color='blue')

# # Plot SOLSTICE PSF
# plt.show()

# Generate a new 'w' array with more points and smaller step size
#w = np.linspace(, 281, 101)

# Generate the Gaussian kernel with sigma = 2.0
# sigma = 2.0
# w0 = np.mean(w)
# gaussian_kernel = np.exp(-(w - w0)**2 / (2 * sigma**2))

# # Normalize the Gaussian kernel
# gaussian_kernel /= np.max(gaussian_kernel)

# # Convolve the data with the Gaussian kernel
# c_iris = convolve(data/np.max(data), gaussian_kernel, mode='same', method='direct')

# # Select the range of 'w' values to plot
# keep = np.where(np.logical_and(w > 280.2, w < 280.5))[0]

# # Plot the convolved IRIS AVG Profile
# plt.plot(w[keep], c_iris[keep], label='IRIS AVG Profile Convolved', color='blue')

# # Show the plot
# plt.show()


# Generate the Gaussian kernel with sigma = 5.0 for a smoother curve
# sigma = 5.0
# w0 = np.mean(w)
# amplitude = 15.0 # Adjust the amplitude of the bell-shaped curve
# gaussian_kernel = amplitude * np.exp(-((w - w0)**2) / (2 * sigma**2))

# # Normalize the Gaussian kernel
# gaussian_kernel /= np.sum(gaussian_kernel)

# # Convolve the data with the Gaussian kernel
# c_iris = convolve(data/np.max(data), gaussian_kernel, mode='same', method='direct')

# # Select the range of 'w' values to plot
# keep = np.where(np.logical_and(w > 280.2, w < 280.5))[0]


# # Plot the convolved IRIS AVG Profile
# plt.plot(w[keep], c_iris[keep], label='IRIS AVG Profile Convolved', color='red')

# # Show the plot
# plt.show()
















# Convolve IRIS AVG Profile with SOLSTICE PSF
# c_iris = convolve(data/np.max(data), sol_psf, mode='same', method='direct')

# # Plot convolved IRIS AVG Profile
# plt.plot(w, c_iris, label='IRIS AVG Profile Convolved', color='blue')




# Convolve SOLSTICE L3 data with SOLSTICE PSF
#c_solstice = convolve(sol_irr/np.max(sol_irr), sol_psf, mode='same', method='direct')


# Plot convolved SOLSTICE L3 data
#plt.plot(sol_wave, c_solstice, label='SOLSTICE L3 Convolved', color='green')

# Set labels and title
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized signal')
# datestr = "2023-04-12" # Define the date string here
# plt.title('Mg II h ' + datestr)

# # Add legend
# plt.legend()

# # Show the plot
# plt.show()






# Plot SOLSTICE L3
#plt.plot(sol_wave, sol_irr/np.max(sol_irr), 'D', label='SOLSTICE L3', color='purple', markersize=5)

# # Plot IRIS AVG Profile
# plt.plot(w, data/np.max(data), label='IRIS AVG Profile', color='black')
# # Plot SOLSTICE L3
# plt.plot(sol_wave, sol_irr/np.max(sol_irr), 'D', label='SOLSTICE L3', color='purple', markersize=5)

# # Update SOLSTICE PSF
# psf = get_sorce_solstice_muv_psf()
# sol_psf = gaussian(w - w0/10.0, psf[1], psf[2]) > 0

# # Plot SOLSTICE PSF
# plt.plot(w, sol_psf/np.max(sol_psf), '--', label='SOLSTICE PSF', color='dodgerblue', linewidth=3)

#  # Convolve iris data with solstice PSF
# c = convolve(data, sol_psf, mode='same', method='direct')

# # # Plot convolved data
# plt.plot(w, c/np.max(c), color='red', label='IRIS CONVOLVED')



# # Set labels and title
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized signal')
# datestr = "2023-04-12" # Define the date string here
# plt.title('Mg II h ' + datestr)

# # Add legend
# plt.legend()

# # Show the plot
# plt.show()


# #Plot data
# plt.plot(w, data/np.max(data), label='IRIS AVG Profile', color='black')
# plt.legend()
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized signal')
# datestr = "2023-04-12" # Define the date string here
# plt.title('Mg II h ' + datestr)
# plt.legend()


# plt.plot(sol_wave, sol_irr/np.max(sol_irr), 'D', label='SOLSTICE L3', color='purple', markersize=1.5)
# plt.legend()
# psf = get_sorce_solstice_muv_psf()
# psf[0] = psf[0]**0.5
# sol_psf = gaussian(w-w0/10.0, psf[1],psf[2]) > 0
# plt.plot(w-w0/10.0, sol_psf/np.max(sol_psf), '--', label='SOLSTICE PSF', color='dodgerblue', linewidth=3)
# plt.legend()
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized signal')
# plt.title('Mg II h'+ 'datestr')
# plt.show()

#   # Plot IRIS AVG Profile
# plt.plot(w, data/np.max(data), label='IRIS AVG Profile', color='black')

# # Plot SOLSTICE L3
# plt.plot(sol_wave, sol_irr/np.max(sol_irr), 'D', label='SOLSTICE L3', color='purple', markersize=1.5)

# # Update SOLSTICE PSF
# psf = get_sorce_solstice_muv_psf()
# #psf[2] = psf[2] * 0.5
# sol_psf = gaussian(w - w0/10.0, psf[1], psf[2]) > 0

# #Plot SOLSTICE PSF
# plt.plot(w, sol_psf/np.max(sol_psf), '--', label='SOLSTICE PSF', color='dodgerblue', linewidth=3)

# # Add legend, xlabel, ylabel, and title
# plt.legend()
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized signal')
# datestr = "2023-04-12" # Define the date string here
# plt.title('Mg II h - ' + 'datestr') # Update the title with the datestr

# # Show the plot
# plt.show()



# # Convolve iris data with solstice PSF
# c = convolve(data, sol_psf, mode='same', method='direct')

# # Plot convolved data
# plt.plot(w, c/np.max(c), color='red', label='IRIS CONVOLVED')
# plt.legend()
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized signal')
# plt.title('Mg II h'+ 'datestr')
# plt.show()


