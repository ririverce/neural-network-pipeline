import numpy as np



def voc_color_decode(segment):
    segment = np.transpose(segment, (1, 2, 0))
    segment = np.argmax(segment, -1)
    decode_segment = np.zeros([segment.shape[0], segment.shape[1], 3])
    segment = np.mod(segment, 32)
    decode_segment[:, :, 1] += np.floor(segment / 16).astype(np.uint8) * 64
    segment = np.mod(segment, 16)
    decode_segment[:, :, 2] += np.floor(segment / 8).astype(np.uint8) * 64
    segment = np.mod(segment, 8)
    decode_segment[:, :, 0] += np.floor(segment / 4).astype(np.uint8) * 128
    segment = np.mod(segment, 4)
    decode_segment[:, :, 1] += np.floor(segment / 2).astype(np.uint8) * 128
    segment = np.mod(segment, 2)
    decode_segment[:, :, 2] += np.floor(segment / 1).astype(np.uint8) * 128
    segment = decode_segment
    return segment