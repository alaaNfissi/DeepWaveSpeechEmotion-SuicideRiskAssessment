#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Voice Activity Detection for Audio Files
Author: Alaa Nfissi
Date: May 19, 2024
Description: This file is used to do voice activity detection for audio files. It uses the webrtcvad library to detect voice activity in audio files.
"""

import collections
import webrtcvad
import contextlib
import wave
import os



def read_wave(path):
    """Reads a .wav file."""
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file."""
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """generate frames
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """collect frames
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b"".join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        pass

    if voiced_frames:
        yield b"".join([f.bytes for f in voiced_frames])


def vad_for_folder(input_path, out_path):

    """do vad for all the audio in the folder"""

    files = os.listdir(input_path)
    for file in files:
        try:
            audio, sample_rate = read_wave(os.path.join(input_path, file))
        except Exception as e:
            print(e)

        vad = webrtcvad.Vad(3)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, 30, vad, frames)

        total_segment = bytes()

        for i, segment in enumerate(segments):
            total_segment += segment

        if len(total_segment) > 6400:  # 0.2 * 16000 * 2
            write_wave(os.path.join(out_path, file), total_segment, sample_rate)


def write_audio(input_path, out_path, vad_paths, vad_labels, vad_sources, label, source):
    """do vad for a single audio file"""
    try:
        audio, sample_rate = read_wave(input_path)
    except Exception as e:
        print(e)
    vad = webrtcvad.Vad(3)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 30, vad, frames)

    total_segment = bytes()

    for i, segment in enumerate(segments):
        total_segment += segment

    max_len = 192000
    min_len = 6400
    flag = False
    for i in range(int(len(total_segment)/max_len)+1):
        if len(total_segment) > min_len:
            if (i+1)*max_len <= len(total_segment):
                write_wave(out_path.split('.w')[0]+f'_{i}.wav', total_segment[i*max_len: (i+1)*max_len], sample_rate)
                vad_paths.append(out_path.split('.w')[0]+f'_{i}.wav')
                vad_labels.append(label)
                vad_sources.append(source)
            else:
                write_wave(out_path.split('.w')[0]+f'_{i}.wav', total_segment[i*max_len:], sample_rate)
                vad_paths.append(out_path.split('.w')[0]+f'_{i}.wav')
                vad_labels.append(label)
                vad_sources.append(source)
            flag = True
    return flag, vad_paths, vad_labels, vad_sources