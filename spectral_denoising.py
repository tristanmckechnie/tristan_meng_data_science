""""
This module is used for spectral denoising of time series data.
"""

from statsmodels.tsa.stattools import acf
from scipy.fft import fft, ifft, fftfreq
from pandas.plotting import autocorrelation_plot
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pywt
import copy

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
    if precentage >= 90:
        complete = True
    else:
        complete = False

    return complete, precentage

def find_threshold(signal, verbose=True):
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

    # if white_noise_1_0 == True:
            # print(f'Threshold found! Threshold: {threshold} Test: {percentage}') 

    # while thresholding has not removed a purely white noise component
    while white_noise_1_0 != True:
        threshold = threshold*.9 
        # noise component we are removing
        psd_indices = psd  < threshold # mask the psd signal
        fft_filtered_noise = fft_coefficients*psd_indices
        inverse_transform_noise = ifft(fft_filtered_noise)

        # is this removed component white noise?
        white_noise_1_0, percentage = test_for_white_noise(inverse_transform_noise)

        # if white_noise_1_0 == True:
            # print(f'Threshold found! Threshold: {threshold} Test: {percentage}')

    if verbose != False:
        # some white noise evidence
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        sns.histplot(np.real(inverse_transform_noise[10:-10]),kde=True,ax=ax[0])
        autocorrelation_plot(inverse_transform_noise[10:-10],ax=ax[1])
        plt.tight_layout()

    return threshold

def automatic_fourier_denoising(signal, df, split,verbose=False):

    """
    This function implement automatic fourier denoising on a time series signal.

    params: signal: one dimensional time series signal. Preferably a numpy array.
    params: df: OHLC + Date dataframe
    params: split: where the training - testing split happens. From the end. |------split<----|
    """

    # step 1: Apply FFT and find threshold
    white_noise_threshold = find_threshold(signal,verbose=verbose)

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
    inverse_transform_filtered = np.real(inverse_transform_filtered)

    # remove start and end spikes
    inverse_transform_filtered[0:10] = signal[0:10]
    inverse_transform_filtered[-10:] = signal[-10:]

    return inverse_transform_filtered

def automatic_fourier_denoising_wf(signal,threshold_override=False,threshold=0.5,verbose=False):
    """
    This function is used for fourier denoising an univariate time series signal
    during walk forward validation.

    params: signal: one dimensional time series signal. Preferably a numpy array.
    """

    # step 1: Apply FFT and find threshold
    if threshold_override == False:
        white_noise_threshold = find_threshold(signal,verbose=verbose)
    else:
        white_noise_threshold = threshold

    # step 2: Apply FFT and produce PSD
    fft_coefficients = fft(signal)
    n = len(signal)
    freqs = fftfreq(signal.shape[0]) # x axis of amplitude vs frequency graphs
    psd = np.abs(fft_coefficients)/n # psd is amplitude/N, psd or power spectrum density is the magnitude of the coefficients resulting from fourier transform

    # step 3: Denoise time series signal with this threshold, also retrieve noise component
    psd_indices = psd >  white_noise_threshold # mask the psd signal
    fft_filtered_denoised = fft_coefficients*psd_indices
    inverse_transform_filtered = ifft(fft_filtered_denoised)

    # return denoised signal
    inverse_transform_filtered = np.real(inverse_transform_filtered)

    # remove start and end spikes
    inverse_transform_filtered[0:10] = signal[0:10]
    inverse_transform_filtered[-10:] = signal[-10:]

    return inverse_transform_filtered

def find_wavelet_threshold(signal,wavelet='sym8',verbose=False):
    """
    This function is used to deterine what a good thresholding value is. It works by iterating
    through possible thresholding values until the noise component removed is found to be white
    noise.
    """

   # Create wavelet object and define parameters
    w = pywt.Wavelet(wavelet) 
    maxlev = pywt.dwt_max_level(len(signal), w.dec_len)
    threshold = 0.9 # Threshold for filtering coefficients as part of denoising, the higher this value the more coefficients you set to zero, ie more of the original signal you truncate away / denoise

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(signal, w, level=maxlev) # multi-level decomposition

    # Threshold the wavelet coefficients for each scale / level, thereby removing noise.
    coeffs_thresholded = copy.deepcopy(coeffs)
    for i in range(1, len(coeffs)):
        coeffs_thresholded[i] = pywt.threshold(coeffs[i], threshold*np.max(coeffs[i]),mode='hard')

    # inverse transform coefficient to reconstruct time series signal, minus noise
    datarec = pywt.waverec(coeffs_thresholded, w) # multi-level decomposition reconstruction

    n_datarec = len(datarec)
    n_signal = len(signal)

    if n_datarec > n_signal:
        datarec = datarec[0:n_signal]

    # noise component
    noise = signal - datarec

    # test if this component is white noise
    white_noise_1_0,percentage = test_for_white_noise(noise)

    # if white_noise_1_0 == True:
    print(f'Threshold found! Threshold: {threshold} Test: {percentage}')

    # while thresholding has not removed a purely white noise component
    while white_noise_1_0 != True:
        # iterate on threshold value, this is a geometric iteration
        threshold = threshold*.9 

        # new thresholded signal
        coeffs_thresholded = copy.deepcopy(coeffs)
        for i in range(1, len(coeffs)):
            coeffs_thresholded[i] = pywt.threshold(coeffs[i], threshold*np.max(coeffs[i]),mode='hard')

        # inverse transform coefficient to reconstruct time series signal, minus noise
        datarec = pywt.waverec(coeffs_thresholded, w) # multi-level decomposition reconstruction
        
        n_datarec = len(datarec)
        n_signal = len(signal)

        if n_datarec > n_signal:
            datarec = datarec[0:n_signal]

        noise = signal - datarec
        # is this removed component white noise?
        white_noise_1_0, percentage = test_for_white_noise(noise)

        # if white_noise_1_0 == True:
        print(f'Threshold found! Threshold: {threshold} Test: {percentage}')

    # some white noise evidence
    if verbose != False:
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        sns.histplot(np.real(noise),kde=True,ax=ax[0])
        autocorrelation_plot(noise,ax=ax[1])
        plt.tight_layout()

    return threshold

def automatic_wavelet_denoising(signal,df,split,wavelet='sym8',threshold_override=False,threshold=0.5,verbose=False):
    """
    This function is used for wavelet denoising an univariate time series signal
    during single out of sample.

    params: signal: one dimensional time series signal. Preferably a numpy array.
    params: wavelet: mother wavelet to use for convolation / wavelet decomposition.
    params: df: OHLC + Date dataframe
    params: split: where the training - testing split happens. From the end. |------split<----|
    """

    # step 1: Apply FFT and find threshold
    if threshold_override == False:
        white_noise_threshold = find_wavelet_threshold(signal,wavelet=wavelet,verbose=verbose)
    else:
        white_noise_threshold = threshold

    # step 2: Apply wavelet decomposition
    w = pywt.Wavelet(wavelet) 
    maxlev = pywt.dwt_max_level(len(signal), w.dec_len)

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(signal, w, level=maxlev) # multi-level decomposition

    # Threshold the wavelet coefficients for each scale / level, thereby removing noise.
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], white_noise_threshold*np.max(coeffs[i]),mode='soft')

    # inverse transform coefficient to reconstruct time series signal, minus noise
    datarec = pywt.waverec(coeffs, w) # multi-level decomposition reconstruction

    if threshold_override==True:
        complete,perct = test_for_white_noise(datarec)
        print(f'With your threshold applied the percetange is : {perct}')

    # plot results
    fig, ax = plt.subplots(2,1,figsize=(15,5),sharex=True)
    ax[0].plot(df['Date'][:-split], signal,label='Raw signal')
    ax[0].plot(df['Date'][:-split], datarec,label="De-noised signal using wavelet techniques")

    ax[1].plot(df['Date'][:-split], signal-datarec,label="Noise component")

    max = df.iloc[:-split,:].shape[0]
    relevant_dates = df['Date'][:-split]
    ax[1].set_xticks([relevant_dates.iloc[x] for x in range(0,max,150)])
    ax[0].legend()
    plt.tight_layout()
    plt.show()

    if verbose != False and threshold_override==True:
        noise = signal-datarec
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        sns.histplot(np.real(noise),kde=True,ax=ax[0])
        autocorrelation_plot(noise,ax=ax[1])
        plt.tight_layout()

    return datarec

def automatic_wavelet_denoising_wf(signal,wavelet='sym8',threshold_override=False,threshold=0.5,verbose=False):
    """
    This function is used for wavelet denoising an univariate time series signal
    during walk forward validation.

    params: signal: one dimensional time series signal. Preferably a numpy array.
    params: wavelet: mother wavelet to use for convolation / wavelet decomposition.
    """

    # step 1: Apply FFT and find threshold
    if threshold_override == False:
        white_noise_threshold = find_wavelet_threshold(signal,wavelet=wavelet,verbose=verbose)
    else:
        white_noise_threshold = threshold
    # step 2: Apply wavelet decomposition
    w = pywt.Wavelet(wavelet) 
    maxlev = pywt.dwt_max_level(len(signal), w.dec_len)

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(signal, w, level=maxlev) # multi-level decomposition

    # Threshold the wavelet coefficients for each scale / level, thereby removing noise.
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], white_noise_threshold*np.max(coeffs[i]),mode='hard')

    # inverse transform coefficient to reconstruct time series signal, minus noise
    datarec = pywt.waverec(coeffs, w) # multi-level decomposition reconstruction

    if threshold_override==True:
        complete,perct = test_for_white_noise(datarec)
        print(f'With your threshold applied the percetange is : {perct}')

    return datarec