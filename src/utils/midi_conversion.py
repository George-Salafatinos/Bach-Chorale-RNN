"""
Utilities for converting Bach chorales to and from MIDI format.
"""
import numpy as np
from typing import List, Optional
from mido import Message, MidiFile, MidiTrack, bpm2tempo


def save_chorale_as_midi(chorale: np.ndarray, output_path: str, tempo: int = 120) -> None:
    """
    Save a chorale as a MIDI file.
    
    Args:
        chorale: Numpy array of shape (time_steps, 4) with MIDI note values
        output_path: Path to save the MIDI file
        tempo: Tempo in beats per minute
    """
    midi = MidiFile()
    voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
    
    # Set tempo
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message('program_change', program=0, time=0))  # Piano
    track.append(Message('control_change', control=7, value=100, time=0))  # Volume
    track.append(Message('control_change', control=10, value=64, time=0))  # Pan
    track.append(Message('control_change', control=91, value=30, time=0))  # Reverb
    track.append(Message('control_change', control=93, value=30, time=0))  # Chorus
    track.append(Message('control_change', control=121, value=0, time=0))  # Reset all controllers
    track.append(Message('control_change', control=64, value=0, time=0))  # Damper pedal off
    track.append(Message('control_change', control=123, value=0, time=0))  # All notes off
    track.append(Message('program_change', program=0, time=0))  # Piano
    track.append(Message('set_tempo', tempo=bpm2tempo(tempo), time=0))
    
    # Create a track for each voice
    tracks = []
    for i in range(4):
        track = MidiTrack()
        track.append(Message('program_change', program=0, time=0))  # Piano
        tracks.append(track)
        midi.tracks.append(track)
    
    # Add notes for each voice
    for i, track in enumerate(tracks):
        current_notes = set()  # Keep track of currently playing notes
        
        for t in range(chorale.shape[0]):
            note = chorale[t, i]
            
            # Note off for currently playing notes
            if len(current_notes) > 0:
                for note_to_turn_off in current_notes:
                    track.append(Message('note_off', note=note_to_turn_off, velocity=64, time=0))
                current_notes = set()
            
            # Note on for new note
            if note > 0:
                track.append(Message('note_on', note=int(note), velocity=64, time=240))  # Quarter note
                current_notes.add(int(note))
            else:
                # Add a rest
                track.append(Message('note_off', note=0, velocity=0, time=240))
    
    # Save MIDI file
    midi.save(output_path)


def load_midi_as_chorale(midi_path: str) -> np.ndarray:
    """
    Load a MIDI file as a chorale.
    
    Args:
        midi_path: Path to the MIDI file
        
    Returns:
        Numpy array of shape (time_steps, 4) with MIDI note values
    """
    midi = MidiFile(midi_path)
    
    # Extract tracks for each voice
    tracks = midi.tracks[1:5]  # Skip the first track (tempo track)
    
    # Determine the length of the chorale
    max_time = 0
    for track in tracks:
        time = 0
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' or msg.type == 'note_off':
                max_time = max(max_time, time)
    
    # Convert ticks to time steps (assuming 480 ticks per beat)
    time_steps = max_time // 480 + 1
    
    # Initialize chorale array
    chorale = np.zeros((time_steps, 4))
    
    # Fill in notes
    for voice_idx, track in enumerate(tracks):
        time = 0
        current_step = 0
        
        for msg in track:
            time += msg.time
            current_step = time // 480
            
            if msg.type == 'note_on' and msg.velocity > 0:
                if current_step < time_steps:
                    chorale[current_step, voice_idx] = msg.note
    
    return chorale


def transpose_midi(midi_path: str, output_path: str, semitones: int) -> None:
    """
    Transpose a MIDI file by a given number of semitones.
    
    Args:
        midi_path: Path to the input MIDI file
        output_path: Path to save the transposed MIDI file
        semitones: Number of semitones to transpose by (positive = up, negative = down)
    """
    midi = MidiFile(midi_path)
    
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' or msg.type == 'note_off':
                msg.note = max(0, min(127, msg.note + semitones))
    
    midi.save(output_path)