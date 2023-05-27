import sys
import numpy as np


def melodySplit(matrix, WINDOWSIZE=32, HOPSIZE=16, VECTORSIZE=142):
    """Clip a (melody) sequence into short WINDOWSIZE-step snippets under a hop size of HOPSIZE. This function is fitted for 1/4-beat quantized sequence."""
    start_downbeat = 0
    end_downbeat = matrix.shape[0]//16
    assert(end_downbeat - start_downbeat >= 2)
    splittedMatrix = np.empty((0, WINDOWSIZE, VECTORSIZE))
    for idx_T in range(start_downbeat*16, (end_downbeat-(WINDOWSIZE//16 -1))*16, HOPSIZE):
        if idx_T > matrix.shape[0]-32:
            break
        sample = matrix[idx_T:idx_T+WINDOWSIZE, :VECTORSIZE][np.newaxis, :, :]
        #print(sample.shape)
        splittedMatrix = np.concatenate((splittedMatrix, sample), axis=0)
    return splittedMatrix


def chordSplit(chord, WINDOWSIZE=8, HOPSIZE=8):
    """Clip a chord sequence into short WINDOWSIZE-step snippets under a hop size of HOPSIZE. This function is fitted for 1-beat quantized sequence."""
    start_downbeat = 0
    end_downbeat = chord.shape[0]//4
    splittedChord = np.empty((0, WINDOWSIZE, 36))
    for idx_T in range(start_downbeat*4, (end_downbeat-(WINDOWSIZE//4 -1))*4, HOPSIZE):
        if idx_T > chord.shape[0]-8:
            break
        sample = chord[idx_T:idx_T+WINDOWSIZE, :][np.newaxis, :, :]
        splittedChord = np.concatenate((splittedChord, sample), axis=0)
    return splittedChord


def split_phrases(segmentation):
    """Split a phrase label string into individual phrase meta info"""
    if '\n' not in segmentation:
        segmentation += '\n'
    phrases = []
    lengths = []
    current = 0
    while segmentation[current] != '\n':
        if segmentation[current].isalpha():
            j = 1
            while not (segmentation[current + j].isalpha() or segmentation[current + j] == '\n'):
                j += 1
            phrases.append(segmentation[current])
            lengths.append(int(segmentation[current+1: current+j]))
            current += j
    return [(phrases[i], lengths[i], sum(lengths[:i])) for i in range(len(phrases))]  


def chord_shift(prChordSet):
    """Transpose a chord sequence to all 12keys (batch processing)."""
    #prChordSet: (batch, time, feature_dim)
    if prChordSet.shape[-1] == 14:
        prChordSet = prChordSet[:, :, 1: -1]
    elif prChordSet.shape[-1] == 12:
        pass
    else:
        print('Chord Dimention Error')
        sys.exit()
    num_total = prChordSet.shape[0]
    shift_const = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    shifted_ensemble = []
    for i in shift_const:
        shifted_term = np.roll(prChordSet, i, axis=-1)
        shifted_ensemble.append(shifted_term)
    shifted_ensemble = np.array(shifted_ensemble)   #num_pitches * num_pieces * duration * size   #.reshape((-1, prChordSet.shape[1], prChordSet.shape[2]))
    return shifted_ensemble, num_total, shift_const


def computeTIV(chroma):
    """Compute TIV for a sequence of chords. TIV credit to G. Bernardes et al., "A multi-level tonal interval space for modelling pitch relatedness and musical consonance," JNMR, 2016."""
    #inpute size: Time*12
    if (len(chroma.shape)) == 4:
        num_pitch = chroma.shape[0]
        num_pieces = chroma.shape[1]
        chroma = chroma.reshape((-1, 12))
        chroma = chroma / (np.sum(chroma, axis=-1)[:, np.newaxis] + 1e-10) #Time * 12
        TIV = np.fft.fft(chroma, axis=-1)[:, 1: 7] #Time * (6*2)
        TIV = np.concatenate((np.abs(TIV), np.angle(TIV)), axis=-1) #Time * 12
        TIV = TIV.reshape((num_pitch, num_pieces, -1, 12))
    else:
        chroma = chroma / (np.sum(chroma, axis=-1)[:, np.newaxis] + 1e-10) #Time * 12
        TIV = np.fft.fft(chroma, axis=-1)[:, 1: 7] #Time * (6*2)
        TIV = np.concatenate((np.abs(TIV), np.angle(TIV)), axis=-1) #Time * 12
    return TIV #Time * 12


def cosine(query, instance_space):
    """Calculate cosine similarity cos(query, reference). This function is fitted for the chord format."""
    #query: batch_Q * T * 12
    #instance_space: 12 * batch_R * T * 12
    batch_Q, _, _ = query.shape
    shift, batch_R, time, chroma = instance_space.shape
    query = query.reshape((batch_Q, -1))[np.newaxis, :, :]
    instance_space = instance_space.reshape((shift, batch_R, -1))
    #result: 12 * Batch_Q * Batch_R
    result = np.matmul(query, np.transpose(instance_space, (0, 2, 1))) / (np.linalg.norm(query, axis=-1, keepdims=True) * np.transpose(np.linalg.norm(instance_space, axis=-1, keepdims=True), (0, 2, 1)) + 1e-10)
    #result: Batch_Q * Batch_R
    chord_result = np.max(result, axis=0)
    arg_result = np.argmax(result, axis=0)
    return chord_result[0], arg_result[0]


def cosine_rhy(query, instance_space):
    """Calculate cosine similarity cos(query, reference). This function is fitted for the rhythm format."""
    #query: 1 * T * 3
    #instance_space:  batch * T * 3
    batch_Q, _, _ = query.shape
    batch_R, _, _ = instance_space.shape
    query = query.reshape((batch_Q, -1))
    instance_space = instance_space.reshape((batch_R, -1))
    #result: 12 * Batch_Q * Batch_R
    result = np.matmul(query, np.transpose(instance_space, (1, 0))) / (np.linalg.norm(query, axis=-1, keepdims=True) * np.transpose(np.linalg.norm(instance_space, axis=-1, keepdims=True), (1, 0)) + 1e-10)
    return result[0]


def cosine_mel(query, instance_space):
    """Calculate cosine similarity cos(query, reference). This function is fitted for the melody format."""
    #query: 1 * m
    #instance_space:  batch * m
    #result: 12 * Batch_Q * Batch_R
    result = np.matmul(query, instance_space) / (np.linalg.norm(query, axis=-1, keepdims=True) * np.linalg.norm(instance_space, axis=-1, keepdims=True) + 1e-10)
    return result[0]


def cosine_1d(query, instance_space, segmentation):
    """Calculate cosine similarity cos(query, reference). This function for general 1-D sequence in batch processing."""
    #query: T
    #instance space: Batch * T
    #instance_space: batch * vectorLength
    final_result = np.ones((instance_space.shape[0]))
    recorder = []
    start = 0
    for i in segmentation:
        if i.isdigit():
            end = start + int(i) * 16
            result = np.abs(np.dot(instance_space[:, start: end], query[start: end])/(np.linalg.norm(instance_space[:, start: end], axis=-1) * np.linalg.norm(query[start: end]) + 1e-10))
            recorder.append(result)
            final_result = np.multiply(final_result, result)    #element-wise product
            start = end
    candidates = final_result.argsort()[::-1]
    scores = final_result[candidates]
    return candidates, scores, recorder


def cosine_2d(query, instance_space, segmentation, record_chord=None):
    """Calculate cosine similarity cos(query, reference). This function for general 2-D sequence in batch processing."""
    final_result = np.ones((instance_space.shape[0]))
    recorder = []
    start = 0
    for i in segmentation:
        if i.isdigit():
            end = start + int(i) * 4
            result = np.dot(np.transpose(instance_space[:, start: end, :], (0, 2, 1)), query[start: end, :])/(np.linalg.norm(np.transpose(instance_space[:, start: end, :], (0, 2, 1)), axis=-1, keepdims=True) * np.linalg.norm(query[start: end, :], axis=0, keepdims=True) + 1e-10)
            result = np.trace(result, axis1=-2, axis2=-1) /2
            recorder.append(result)

            final_result = np.multiply(final_result, result)
            start = end
    if not record_chord == None:
        record_chord = np.array(record_chord)
        recorder = np.array(recorder)
        assert np.shape(record_chord) == np.shape(recorder)
        final_result = np.array([(np.product(recorder[:, i]) * np.product(record_chord[:, i])) * (2 *recorder.shape[0]) for i in range(recorder.shape[1])])
 
    candidates = final_result.argsort()[::-1]
    scores = final_result[candidates]
    return candidates, scores, recorder


def piano_roll_shift(prpiano_rollSet):
    """Transpose a piano_roll format described in R. Yang et al., "Deep music analogy via latent representation disentanglement," ISMIR 2019."""
    num_total, timeRes, piano_shape = prpiano_rollSet.shape
    shift_const = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    shifted_ensemble = []
    for i in shift_const:
        piano = prpiano_rollSet[:, :, :128]
        rhythm = prpiano_rollSet[:, :, 128: 130]
        chord = prpiano_rollSet[:, :, 130:]
        shifted_piano = np.roll(piano, i, axis=-1)
        shifted_chord = np.roll(chord, i, axis=-1)
        shifted_piano_roll_set = np.concatenate((shifted_piano, rhythm, shifted_chord), axis=-1)
        shifted_ensemble.append(shifted_piano_roll_set)
    shifted_ensemble = np.array(shifted_ensemble).reshape((-1, timeRes, piano_shape))
    return shifted_ensemble, num_total, shift_const
