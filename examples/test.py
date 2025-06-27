#
#  Copyright © 2025 Agora
#  This file is part of TEN Framework, an open source project.
#  Licensed under the Apache License, Version 2.0, with certain conditions.
#  Refer to the "LICENSE" file in the root directory for more information.
#
import sys, os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../include")))
from ten_vad import TenVad
import librosa
import numpy as np
import scipy.io.wavfile as Wavfile

if __name__ == "__main__":
    input_file, sampling_rate, hop_size = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
    )
    sr, data = Wavfile.read(input_file)

    threshold = 0.5
    ten_vad_instance = TenVad(hop_size, threshold)  # Create a TenVad instance
    num_frames = data.shape[0] // hop_size
    # Streaming inference
    start = time.time()
    for i in range(num_frames):
        audio_data = data[i * hop_size : (i + 1) * hop_size]
        out_probability, out_flag = ten_vad_instance.process(
            audio_data
        )  #  Out_flag is speech indicator (0 for non-speech signal, 1 for speech signal)
        print("[%d] %0.6f, %d" % (i, out_probability, out_flag))

    end = time.time()
    lag = end - start
    print(f"latency:{lag}")
