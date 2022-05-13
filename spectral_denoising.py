""""
This module is used for spectral denoising of time series data.
"""

from statsmodels.tsa.stattools import acf
from scipy.fft import fft, ifft, fftfreq
from pandas.plotting import autocorrelation_plot
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def test_for_white_noise(inverse_transform_noise):
    """
    This function is used to test whether the noise component removed from the signal is
    white noise or not. If it is white noise it returns True, otherwise False.

    The test used to determine white noise or not is from: https://otexts.com/fpp2/wn.html
    """

    # compute autocorrelations
    acf_values = acf(np.real(inverse_transform_noise),nlags=100)

    # compute boundary for acceptable for white noise
    critical_val = 2/np.sqrt(len(inverse_transform_noise))

    # seperate negative and positive values
    postive_values = acf_values[acf_values > 0]
    negative_values = acf_values[acf_values < 0]

    # how many negative and posititve values are contained in the critical region
    num_pos_in_crit = len(postive_values[postive_values < critical_val])
    num_neg_in_crit = len(negative_values[negative_values < critical_val])

    # total percentage of correlation in critical region
    precentage = (num_pos_in_crit+num_neg_in_crit)/len(acf_values) * 100

    complete = None
    if precentage >= 95:
        complete = True
    else:
        complete = False

    return complete, precentage

def find_threshold(signal):
    """
    This function is used to deterine what a good thresholding value is. It works by iterating
    through possible thresholding values until the noise component removed is found to be white
    noise.
    """

    # fft the uninvariete time series signal
    fft_coefficients = fft(signal) # fourier transform returns coefficients

    # plot amplitude vs frequency 
    n = len(signal)

    # get frequencies and psd
    freqs = fftfreq(signal.shape[0]) # x axis of amplitude vs frequency graphs
    psd = np.abs(fft_coefficients)/n # psd is amplitude/N, psd or power spectrum density is the magnitude of the coefficients resulting from fourier transform

    # retrieve highest amplitude coefficient
    threshold = np.max(psd[1:]) / 10 # initial guess for threshold

    # initial guess results
    psd_indices = psd  < threshold # mask the psd signal
    fft_filtered_noise = fft_coefficients*psd_indices
    inverse_transform_noise = ifft(fft_filtered_noise)

    # test if this component is white noise
    white_noise_1_0,percentage = test_for_white_noise(inverse_transform_noise)

    if white_noise_1_0 == True:
            print(f'Threshold found! Threshold: {threshold} Test: {percentage}') 

    # while thresholding has not removed a purely white noise component
    while white_noise_1_0 != True:
        threshold = threshold*.9 
        # noise component we are removing
        psd_indices = psd  < threshold # mask the psd signal
        fft_filtered_noise = fft_coefficients*psd_indices
        inverse_transform_noise = ifft(fft_filtered_noise)

        # is this removed component white noise?
        white_noise_1_0, percentage = test_for_white_noise(inverse_transform_noise)

        if white_noise_1_0 == True:
            print(f'Threshold found! Threshold: {threshold} Test: {percentage}')

    # some white noise evidence
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    sns.histplot(np.real(inverse_transform_noise[10:-10]),kde=True,ax=ax[0])
    autocorrelation_plot(inverse_transform_noise[10:-10],ax=ax[1])
    plt.tight_layout()

    return threshold

def automatic_fourier_denoising(signal, df, split):

    """
    This function implement automatic fourier denoising on a time series signal.

    params: signal: one dimensional time series signal. Preferably a numpy array.
    params: df: OHLC + Date dataframe
    params: split: where the training - testing split happens. From the end. |------split<----|
    """

    # step 1: Apply FFT and find threshold
    white_noise_threshold = find_threshold(signal)

    # step 2: Apply FFT and produce PSD
    fft_coefficients = fft(signal)
    n = len(signal)
    freqs = fftfreq(signal.shape[0]) # x axis of amplitude vs frequency graphs
    psd = np.abs(fft_coefficients)/n # psd is amplitude/N, psd or power spectrum density is the magnitude of the coefficients resulting from fourier transform

    # step 3: Denoise time series signal with this threshold, also retrieve noise component
    psd_indices = psd >  white_noise_threshold # mask the psd signal
    fft_filtered_denoised = fft_coefficients*psd_indices
    inverse_transform_filtered = ifft(fft_filtered_denoised)

    # white noise component
    psd_indices = psd  < white_noise_threshold 
    fft_filtered_noise = fft_coefficients*psd_indices
    inverse_transform_noise = ifft(fft_filtered_noise)

    # step 4: plot psd
    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(freqs[1:int(n/2)],psd[1:int(n/2)])
    ax.set_ylabel('Power spectrum',fontsize=15)
    ax.set_xlabel('Frequencies',fontsize=15)
    ax.set_title('FFT')
    ax.tick_params(labelsize=15)
    plt.tight_layout()

    # step 4: denoising results
    fig,ax = plt.subplots(2,1,figsize=(10,5),sharex=True)
    ax[0].plot(df['Date'][:-split],signal,'-',label='Real data')
    ax[0].plot(df['Date'][:-split],inverse_transform_filtered,'-',label='Inverse fourier filtered')
    ax[1].plot(df['Date'][:-split],inverse_transform_noise,'-',label='Noise component')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title(f'Threshold = {white_noise_threshold}')
    ax[1].set_xlabel('Days',fontsize=15)
    ax[1].set_xticks([df['Date'][:].iloc[x] for x in range(0,len(df['Date'][:-split]),150)])
    plt.tight_layout()

    # return denoised signal
    return np.real(inverse_transform_filtered)