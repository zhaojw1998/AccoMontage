import numpy as np
import pretty_midi as pm


def bpm_to_rate(bpm):
    return 60 / bpm


def ext_nmat_to_nmat(ext_nmat):
    nmat = np.zeros((ext_nmat.shape[0], 4))
    nmat[:, 0] = ext_nmat[:, 0] + ext_nmat[:, 1] / ext_nmat[:, 2]
    nmat[:, 1] = ext_nmat[:, 3] + ext_nmat[:, 4] / ext_nmat[:, 5]
    nmat[:, 2] = ext_nmat[:, 6]
    nmat[:, 3] = ext_nmat[:, 7]
    return nmat


# def nmat_to_pr(nmat, num_step=32):
#     pr = np.zeros((num_step, 128))
#     for s, e, p, v in pr:
#         pr[s, p]

def nmat_to_notes(nmat, start, bpm):
    notes = []
    for s, e, p, v in nmat:
        assert s < e
        assert 0 <= p < 128
        assert 0 <= v < 128
        s = start + s * bpm_to_rate(bpm)
        e = start + e * bpm_to_rate(bpm)
        notes.append(pm.Note(int(v), int(p), s, e))
    return notes


def ext_nmat_to_pr(ext_nmat, num_step=32):
    # [start measure, no, deno, .., .., .., pitch, vel]
    # This is not RIGHT in general. Only works for 2-bar 4/4 music for now.
    pr = np.zeros((32, 128))
    if ext_nmat is not None:
        for (sb, sq, sde, eb, eq, ede, p, v) in ext_nmat:
            s_ind = int(sb * sde + sq)
            e_ind = int(eb * ede + eq)
            p = int(p)
            pr[s_ind, p] = 2
            pr[s_ind + 1: e_ind, p] = 1  # note not including the last ind
    return pr


def ext_nmat_to_mel_pr(ext_nmat, num_step=32):
    # [start measure, no, deno, .., .., .., pitch, vel]
    # This is not RIGHT in general. Only works for 2-bar 4/4 music for now.
    pr = np.zeros((32, 130))
    pr[:, 129] = 1
    if ext_nmat is not None:
        for (sb, sq, sde, eb, eq, ede, p, v) in ext_nmat:
            s_ind = int(sb * sde + sq)
            e_ind = int(eb * ede + eq)
            p = int(p)
            pr[s_ind, p] = 1
            pr[s_ind: e_ind, 129] = 0
            pr[s_ind + 1: e_ind, 128] = 1  # note not including the last ind
    return pr


def augment_pr(pr, shift=0):
    # it assures to work on single pr
    # for an array of pr, should double-check
    return np.roll(pr, shift, axis=-1)


def augment_mel_pr(pr, shift=0):
    # it only works on single mel_pr. Not on array of it.
    pitch_part = np.roll(pr[:, 0: 128], shift, axis=-1)
    control_part = pr[:, 128:]
    augmented_pr = np.concatenate([pitch_part, control_part], axis=-1)
    return augmented_pr

def pr_to_onehot_pr(pr):
    onset_data = pr[:, :] == 2
    sustain_data = pr[:, :] == 1
    silence_data = pr[:, :] == 0
    pr = np.stack([onset_data, sustain_data, silence_data],
                  axis=-1).astype(np.int64)
    return pr


def piano_roll_to_target(pr):
    #  pr: (32, 128, 3), dtype=bool

    # Assume that "not (first_layer or second layer) = third_layer"
    pr[:, :, 1] = np.logical_not(np.logical_or(pr[:, :, 0], pr[:, :, 2]))
    # To int dtype can make addition work
    pr = pr.astype(int)
    # Initialize a matrix to store the duration of a note on the (32, 128) grid
    pr_matrix = np.zeros((32, 128))

    for i in range(31, -1, -1):
        # At each iteration
        # 1. Assure that the second layer accumulates the note duration
        # 2. collect the onset notes in time step i, and mark it on the matrix.

        # collect
        onset_idx = np.where(pr[i, :, 0] == 1)[0]
        pr_matrix[i, onset_idx] = pr[i, onset_idx, 1] + 1
        if i == 0:
            break
        # Accumulate
        # pr[i - 1, :, 1] += pr[i, :, 1]
        # pr[i - 1, onset_idx, 1] = 0  # the onset note should be set 0.

        pr[i, onset_idx, 1] = 0  # the onset note should be set 0.
        pr[i - 1, :, 1] += pr[i, :, 1]
    return pr_matrix


def target_to_3dtarget(pr_mat, max_note_count=11, max_pitch=107, min_pitch=22,
                       pitch_pad_ind=88, dur_pad_ind=2,
                       pitch_sos_ind=86, pitch_eos_ind=87):
    """
    :param pr_mat: (32, 128) matrix. pr_mat[t, p] indicates a note of pitch p,
    started at time step t, has a duration of pr_mat[t, p] time steps.
    :param max_note_count: the maximum number of notes in a time step,
    including <sos> and <eos> tokens.
    :param max_pitch: the highest pitch in the dataset.
    :param min_pitch: the lowest pitch in the dataset.
    :param pitch_pad_ind: see return value.
    :param dur_pad_ind: see return value.
    :param pitch_sos_ind: sos token.
    :param pitch_eos_ind: eos token.
    :return: pr_mat3d is a (32, max_note_count, 6) matrix. In the last dim,
    the 0th column is for pitch, 1: 6 is for duration in binary repr. Output is
    padded with <sos> and <eos> tokens in the pitch column, but with pad token
    for dur columns.
    """
    pitch_range = max_pitch - min_pitch + 1  # including pad
    pr_mat3d = np.ones((32, max_note_count, 6), dtype=int) * dur_pad_ind
    pr_mat3d[:, :, 0] = pitch_pad_ind
    pr_mat3d[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(32, dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        pr_mat3d[t, cur_idx[t], 0] = p - min_pitch
        binary = np.binary_repr(min(int(pr_mat[t, p]), 32) - 1, width=5)
        pr_mat3d[t, cur_idx[t], 1: 6] = \
            np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        if cur_idx[t] == max_note_count-1:
            continue
        cur_idx[t] += 1
    #print(cur_idx)
    pr_mat3d[np.arange(0, 32), cur_idx, 0] = pitch_eos_ind
    return pr_mat3d


def expand_chord(chord, shift, relative=False):
    # chord = np.copy(chord)
    root = (chord[0] + shift) % 12
    chroma = np.roll(chord[1: 13], shift)
    bass = (chord[13] + shift) % 12
    root_onehot = np.zeros(12)
    root_onehot[int(root)] = 1
    bass_onehot = np.zeros(12)
    bass_onehot[int(bass)] = 1
    if not relative:
        pass
    #     chroma = np.roll(chroma, int(root))
    # print(chroma)
    # print('----------')
    return np.concatenate([root_onehot, chroma, bass_onehot])
