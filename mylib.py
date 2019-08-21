# -*- coding:utf-8 -*-
"""
Created on 20181010
@author: jiaxp

my python lib for speech signal processing, includes the following functions:

listFile:       get the files with certain appendix in the target path
enframe
overlapAdd
tfRepresent
tfSynthesis
calcSDR
calcSegSNR
calcFwSNR
getIdealMask
addNoise
"""
import os
import numpy as np


def listFile(root_path, appendix='', subfolder=True, update_log=False):
    ''' find all the files in the path with a specified appendix.
        If appendix=='' then return all the files
        inputs:
            root_path:  the target path
            appendix:   str like 'wav' or '.wav'
            subfolder:  if true, search subfolders in root_path as well
            update_log: if true, update log_file. If the list in log_file is
                        wanted, update_log should be False(default)
        outputs:
            file_list:  a string list, elements are abs paths of the wanted
                        files
    '''
    ap_len = len(appendix)
    if ap_len > 0:
        if appendix[0] != '.':
            appendix = '.'+appendix
            ap_len += 1

    log_path = os.path.join(root_path, '_log_{0}.txt'.format(appendix[1:]))
    if os.path.isfile(log_path) and update_log is False:
        # read log_file and return file_list
        file_list = []
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').strip()
                if len(line) > 0:
                    file_list.append(line)
        return file_list

    # get absolute path
    root_path = os.path.abspath(root_path)
    # get the files and directories in current path
    temp_list = os.listdir(root_path)
    # list of wav file
    file_list = []

    for name in temp_list:
        name = os.path.join(root_path, name)
        # if is dir, check the files in it
        if os.path.isdir(name):
            if subfolder:
                file_list.extend(listFile(name, appendix, subfolder=True,
                                          update_log=False))
        # if is file, check whether it has the wanted appendix
        elif os.path.isfile(name):
            if ap_len > 0:
                if name[-1*ap_len:].lower() == appendix.lower():
                    file_list.append(name)
            else:
                file_list.append(name)

    if update_log:
        # update log_file
        with open(log_path, 'w') as f:
            for name in file_list:
                f.write(name+'\n')
    return file_list


def enframe(sig, frame_len, hop_size, window='rect'):
    ''' enframe the target signal (only for one channel)
        inputs:
            sig:        the target signal, should be 1-channel
            frame_len:  the length of the frames
            hop_size:   step_length of the analysis window
            window:     the window type or window-array, default to
                        rect-window (all ones)
        outputs:
            frames:     numpy array, shape=(frame_num, frame_len)
            params:     params dict, keys=['sig_len', 'frame_len', 'hop_size',
                        'scaler', 'window_type', 'window']
    '''
    def checkProperty(w, hop_size):
        ''' check the ola property of the conbination of w and hop_size
                inputs:
                    w:          window, ndarray
                    hop_size:   integer
                outputs:
                    flag:       true or false
                    scaler:      the scaler used for ola
        '''
        w_len = len(w)
        sums = np.zeros(w_len)
        for i in range(w_len):
            for j in range(w_len):
                cur_i = i+j*hop_size
                if cur_i < 0 or cur_i >= w_len:
                    break
                sums[i] = sums[i] + w[cur_i]
            for j in range(1, w_len):
                cur_i = i-j*hop_size
                if cur_i < 0 or cur_i >= w_len:
                    break
                sums[i] = sums[i] + w[cur_i]
        if np.sum(np.abs(sums-sums[0])) < 1e-8:
            flag = True
        else:
            flag = False
        return flag, np.around(sums[0], decimals=5)

    # get the claimed window
    if type(window) == np.ndarray or type(window) == list:
        window_type = 'self_decided'
        window = np.asarray(window)
        if len(window) != frame_len:
            raise Exception("ERR(enframe window err):self_decided window "
                            + "does not match frame_len")
    elif window == 'hanning':
        window_type = 'hanning'
        if frame_len % 2 == 0:
            window = np.hanning(frame_len+1)[1:]
        else:
            window = np.hanning(frame_len)
    elif window == 'hamming':
        window_type = 'hamming'
        if frame_len % 2 == 0:
            window = np.hamming(frame_len+1)[1:]
        else:
            window = np.hamming(frame_len)
    elif window == 'rect':
        window_type = 'rect'
        window = np.ones([frame_len])
    else:
        raise Exception("ERR(enframe window_type err): 'window' should be "
                        + "'hanning','hamming','rect' or a numpy array "
                        + "containing the target window")

    # ola quality check
    [flag, scaler] = checkProperty(window, hop_size)
    if flag is False:
        print("\nWarning(enframe waring): the conbination of window and "
              + "hop_size may not be compatible for overlap-add.\n")

    # allocate the result array
    sig_len = len(sig)
    data_type = type(sig[0])
    frame_num = int(np.ceil((sig_len+frame_len)/hop_size))-1
    frame_matrix = np.zeros([frame_num, frame_len])

    # pad sig with trailing zeros
    sig = np.hstack((np.zeros(frame_len-hop_size), sig, np.zeros(frame_len)))
    for frame_i in range(0, frame_num):
        frame_matrix[frame_i, :] = sig[(frame_i)*hop_size:
                                       (frame_i*hop_size+frame_len)]*window

    params = {}
    params['sig_len'] = sig_len
    params['frame_len'] = frame_len
    params['hop_size'] = hop_size
    params['scaler'] = scaler
    params['window_type'] = window_type
    params['window'] = window
    params['data_type'] = data_type
    return frame_matrix, params


def overlapAdd(frames, params):
    ''' overlap-add, recover the origin 1-channel signal
        inputs:
            frames:     frames in shape (frame_num, frame_len)
            params:     enframe param dict, keys=['sig_len', 'frame_len',
                        'hop_size', 'scaler', 'window_type', 'window',
                        'fft_len', 'data_type']
        outputs:
            sig:        recovered signal, has the same length and type with sig
    '''
    # allocate sig array
    frame_len = params['frame_len']
    if frames.shape[1] != frame_len:
        raise Exception("ERR(ola frame_len err): frame_len in param dict does "
                        + "not match the shape of frames...params['frame_len']"
                        + "={0}, shape of frames is ({1},{2})."
                        .format(frame_len, frames.shape[0], frames.shape[1]))
    frame_num = frames.shape[0]
    hop_size = params['hop_size']
    sig_len_rough = frame_num*hop_size+frame_len
    sig = np.zeros(sig_len_rough)
    # overlap-add
    scaler = params['scaler']
    for i in range(frame_num):
        sig[i*hop_size:i*hop_size+frame_len] += frames[i, :]/scaler
    # remove the leading and trailing zeros added in enframe
    sig_len = params['sig_len']
    data_type = params['data_type']
    sig = np.asarray(sig[frame_len-hop_size:frame_len-hop_size+sig_len],
                     dtype=data_type)

    return sig


def tfRepresent(sig, frame_len=512, hop_size=256, fft_len=None, window='rect'):
    ''' calc the time-freq representation of the input signal
    inputs:
        sig:            input signal, time domain
        frame_len:      frame length in points
        hop_size:       step in points
        fft_len:        fft length, default to frame_len, should be even
        window:         may be 'rect', 'hanning', 'hamming' or ndarray
    outputs:
        tf_matrix:      tf representation of signal(half band),
                        in shape (frame_num, fft_len//2+1)
        params:         param_dict, keys=['sig_len', 'frame_len', 'hop_size',
                        'scaler', 'window_type', 'window', 'fft_len']
    '''
    # check input params
    if fft_len is None:
        fft_len = frame_len
    if fft_len % 2 == 1:
        print("Warning (tfRepresent warning): fft_len is odd while it is "
              + "suggested to be even.")

    # enframe
    [frames, params] = enframe(sig, frame_len, hop_size, window)

    # rfft
    tf_matrix = np.fft.rfft(frames, n=fft_len, axis=1)
    params['fft_len'] = fft_len

    return tf_matrix, params


def tfSynthesis(tf_matrix, params):
    ''' get time domain signal from tf-representation
    inputs:
        tfmatrix:   tf-domain representation in shape (frame_num, freq_bin_num)
        params:     param dict used to recover signal,
                    keys=['sig_len', 'frame_len', 'hop_size', 'scaler',
                          'window_type', 'window', 'fft_len']
    outputs:
        sig:        1-channel time-domain signal
    '''
    # get time-domain frames
    fft_len = params['fft_len']
    frames = np.fft.irfft(tf_matrix, n=fft_len, axis=1)
    frame_len = params['frame_len']
    if fft_len > frame_len:
        frames = frames[:, :frame_len]
    # overlap-add
    sig = overlapAdd(frames, params)
    return sig


def calcSDR(sig_clean, sig_proc):
    '''
    calc SDR
    input:
        sig_clean:      clean signal (sig_len)
        sig_proc:       processed signal
    output:
        sdr:            SDR result
    '''
    if sig_clean.shape != sig_proc.shape:
        raise Exception('ERROR(calc_SDR): inputs shapes do not match.')
    sig_clean = np.asarray(sig_clean, dtype=np.float32)
    sig_proc = np.asarray(sig_proc, dtype=np.float32)
    E_sig = np.sum(sig_clean**2)
    E_distortion = np.sum((sig_clean-sig_proc)**2)
    if E_distortion < 1e-10:
        E_distortion = 1e-10
    sdr = 10*np.log10(E_sig/E_distortion)
    return sdr


def calcSegSNR(sig_clean, sig_proc):
    '''
    calc segment-SNR
    input:
        sig_clean:      clean signal (sig_len)
        sig_proc:       processed signal
    output:
        seg_snr:            segment-SNR result
    '''
    if sig_clean.shape != sig_proc.shape:
        print('ERROR:(calc_segSDR) inputs shapes do not match.')
        return None
    [tf_c, _] = tfRepresent(sig_clean)
    [tf_n, _] = tfRepresent(sig_proc-sig_clean)
    tf_c = np.sum(np.abs(tf_c)**2, axis=1)
    tf_n = np.sum(np.abs(tf_n)**2, axis=1)
    snrs = 10*np.log10(tf_c/tf_n)
    snrs = np.where(snrs > 35, 35, snrs)
    snrs = np.where(snrs < -10, -10, snrs)
    seg_snr = np.mean(snrs)

    return seg_snr


def calcFwSNR(sig_clean, sig_proc, gamma=0.2):
    '''
    calc fwSNR (frequency-weighted snr)
    input:
        sig_clean:      clean signal (sig_len)
        sig_proc:       processed signal
    output:
        fw_snr:         fw_snr result
    '''
    if sig_clean.shape != sig_proc.shape:
        raise Exception('ERROR(calcFwSNR): inputs shapes do not match.')
    [tf_c, _] = tfRepresent(sig_clean)
    [tf_p, _] = tfRepresent(sig_proc)
    snr = 10*np.log10(np.abs(tf_c)**2/np.power(np.abs(tf_p)-np.abs(tf_c), 2))
    W = np.power(np.abs(tf_c), gamma)
    snrs = np.sum(snr*W, axis=1)/np.sum(W, axis=1)
    snrs = np.where(snrs > 35, 35, snrs)
    snrs = np.where(snrs < -10, -10, snrs)
    fw_snr = np.mean(snrs)

    return fw_snr


def getIdealMask(sig_clean, sig_mixed, frame_len, hop_size, window='hanning',
                 fft_len=None, method='ibm', lc=-5, irm_beta=0.5):
    ''' get ideal masks according to 'method'
        inputs:
            sig_clean:      1-channel clean signal
            sig_mixed:      1-channel mixed signal
            method:         mask algorithm, choose among
                            ['ibm', 'irm', 'fft_mask', 'psm', 'cirm']
            lc:             local criteria, only used when method=='ibm'
        outputs:
            mask:           in shape (frame_num, freq_bin_num),
                            when method=='cirm', return [mask_real, mask_imag]
    '''
    # check method
    methods = ['ibm', 'irm', 'fft_mask', 'psm', 'cirm']
    method = method.lower()
    if method not in methods:
        raise Exception("Error(getIdealMask err): method "
                        + "{0} not in given methods: {1}."
                        .format(method, str(methods)))
    # get clean signals' tf_matrix
    [tf_c, _] = tfRepresent(sig_clean, frame_len, hop_size, fft_len, window)
    # generate and return masks according to method
    if method == 'ibm':
        sig_noise = sig_mixed - sig_clean
        [tf_n, _] = tfRepresent(sig_noise, frame_len, hop_size, fft_len,
                                window)
        power_clean = np.abs(tf_c)
        power_noise_lc = np.abs(tf_n)*(10**(lc/20))
        mask = np.where(power_clean > power_noise_lc, 1, 0)
        return mask
    if method == 'irm':
        sig_noise = sig_mixed - sig_clean
        [tf_n, _] = tfRepresent(sig_noise, frame_len, hop_size, fft_len,
                                window)
        mask = np.abs(tf_c)**2/(np.abs(tf_c)**2+np.abs(tf_n)**2)
        mask = mask**irm_beta
        return mask
    if method == 'fft_mask':
        [tf_m, _] = tfRepresent(sig_mixed, frame_len, hop_size, fft_len,
                                window)
        tf_m = np.where(tf_m == 0, 1e-6, tf_m)
        mask = np.abs(tf_c)/np.abs(tf_m)
        return mask
    if method == 'psm':
        [tf_m, _] = tfRepresent(sig_mixed, frame_len, hop_size, fft_len,
                                window)
        tf_m = np.where(tf_m == 0, 1e-6, tf_m)
        phase = tf_n/tf_m
        phase = phase/np.abs(phase)
        cos_theta = phase.real
        mask = np.abs(tf_c)/np.abs(tf_m)*cos_theta
        return mask
    if method == 'cirm':
        [tf_m, _] = tfRepresent(sig_mixed, frame_len, hop_size, fft_len,
                                window)
        tf_m = np.where(tf_m == 0, 1e-6, tf_m)
        cmask = tf_c/tf_m
        return cmask.real, cmask.imag


def addNoise(sig, noise, snr):
    ''' add noise to sig according to snr (1-channel)
        inputs:
            sig
            noise
            snr
        outputs:
            noisy_sig
    '''
    data_type = type(sig[0])
    sig_len = len(sig)
    noise_len = len(noise)
    if noise_len < sig_len:
        raise Exception("noise is shorter than signal...")
    if noise_len > sig_len:
        start = np.random.randint(0, noise_len-sig_len+1)
        noise = noise[start:start+sig_len]
    sig = np.asarray(sig, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    P_sig = np.mean(sig**2)
    P_noise = np.mean(noise**2)
    alpha = np.sqrt(P_sig/P_noise*(10**(-1*snr/10)))
    noisy_sig = np.asarray(sig+alpha*noise, dtype=data_type)
    return noisy_sig


def calcLSDFromSpec(spec_1, spec_2, log_spec=False):
    if not log_spec:
        spec_1 = np.log(np.abs(spec_1))/np.log(10)
        spec_2 = np.log(np.abs(spec_2))/np.log(10)
    dist_array = (spec_1 - spec_2)*20
    dist_array = dist_array**2
    lsd = np.mean(np.sqrt(np.mean(dist_array, axis=1)))
    return lsd


def calcLSDFromWav(sig_1, sig_2, frame_len=512, hop_size=256, fft_len=512,
                   window='hanning'):
    [tf_m1, _] = tfRepresent(sig_1, frame_len, hop_size, fft_len, window)
    [tf_m2, _] = tfRepresent(sig_2, frame_len, hop_size, fft_len, window)
    lsd = calcLSDFromSpec(tf_m1, tf_m2)
    return lsd


if __name__ == '__main__':
    from scipy.io import wavfile
    result_path = './data/results/3000_1noise_3SNR_log_200/babble/6dB'
    clean_path = './data/clean/test'
    mixed_path = './data/mixed/test/babble/6dB'

    file_list = listFile(result_path)
    file_num = len(file_list)
    print('file_num = ', file_num)
    lsds_mixed = np.zeros(file_num)
    lsds_re = np.zeros(file_num)
    frame_len = 1024
    hop_size = 512
    fft_len = 1024
    for i in range(file_num):
        cur_file = file_list[i]
        [fs, sig_re] = wavfile.read(cur_file)
        cur_mixed_file = os.path.join(mixed_path, os.path.basename(cur_file))
        cur_clean_file = os.path.join(clean_path, os.path.basename(cur_file))
        [fs, sig_mixed] = wavfile.read(cur_mixed_file)
        [fs, sig_clean] = wavfile.read(cur_clean_file)
        lsd_mixed = calcLSDFromWav(sig_mixed, sig_clean, frame_len,
                                   hop_size, fft_len)
        lsd_re = calcLSDFromWav(sig_re, sig_clean, frame_len, hop_size,
                                fft_len)
        lsds_mixed[i] = lsd_mixed
        lsds_re[i] = lsd_re
        print("{0}/{1}: lsd {2} --> {3}".format(i, file_num, lsd_mixed,
                                                lsd_re))
    print("summary:\n\tlsd {0} --> {1}".format(np.mean(lsds_mixed),
                                               np.mean(lsds_re)))
