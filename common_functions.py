import subprocess
from pydub import AudioSegment
import os 

def tostring(lst):
    return ','.join(str(x) for x in lst)

def tolist(str):
    return str.split(',')

def is_hideable(count1,count2):
  framediff = count1 - count2
  if framediff-1 < 0:
    hideable = False 
  else: 
    hideable = True 
  return hideable

def convert_video_to_audio(video_file, output_ext="wav"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    
    
def audio_hideable(nof,sof,len):
  if (nof*sof)<(len*4):
    return False
  else:
    return True

def flatten(lst):
    result = []
    for elem in lst:
        if isinstance(elem, list):
            result.extend(flatten(elem))
        else:
            result.append(elem)
    return result

def to_mono(filename):
  sound = AudioSegment.from_wav(filename)
  sound = sound.set_channels(1)
  sound.export(filename, format="wav")