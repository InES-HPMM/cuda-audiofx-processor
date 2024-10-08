

import numpy as np
import wave
from scipy.io import wavfile
import soundfile as sf


def min_rmsd(a, b, max_offset):
    rmsds = []

    for i in range(max_offset+1):
        offset = max_offset - i
        rmsds.append(rmsd(a[offset:], b[:b.size-offset]))

    for i in range(1, max_offset+1):
        rmsds.append(rmsd(a[:-i], b[i:]))

    rmsds, index = np.min(rmsds), np.argmin(rmsds)

    return rmsds, index-max_offset


def rmsd(a, b):
    return np.sqrt(((a-b)**2).sum(-1) / a.size)


if __name__ == "__main__":
    actual, _ = sf.read("/home/nvidia/git/mt/out/actual_output.wav", dtype="float32")
    expected, _ = sf.read("/home/nvidia/git/mt/out/expected_output.wav", dtype="float32")

    print(min_rmsd(expected, actual, 1000))
