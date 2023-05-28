import os
import torch
import numpy as np
import torchvision
from PIL import Image
import tempfile
import cv2
import subprocess
import traceback
import soundfile as sf
import math
import joblib
import common_functions as cf
import encrypt_decrypt as ed
from text_in_image import Text_Steganography
import audio_in_video as aiv
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tempfile.tempdir = "/tmp"


def Video_Encode(cover_vid_path, secret_vid_path, password, used_model):

    _model = joblib.load(used_model)
    # calculate fps of both

    cap1 = cv2.VideoCapture(cover_vid_path)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)

    cap2 = cv2.VideoCapture(secret_vid_path)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    # extract audio from video

    cf.convert_video_to_audio(cover_vid_path)
    cover_filename, ext = os.path.splitext(cover_vid_path)
    audio_file = f'{cover_filename}.wav'

    output_vid_path = f'{cover_filename}_(stego){ext}'

    cf.convert_video_to_audio(secret_vid_path)
    secret_filename, ext = os.path.splitext(secret_vid_path)
    secret_audio = f'{secret_filename}.wav'
    cf.to_mono(secret_audio)

    output_filename, ext = os.path.splitext(output_vid_path)
    output_with_audio_path = f'/content/drive/MyDrive/Colab Notebooks/DEMO/Dependencies/output{ext}'

    # create temporary files for frames of both videos

    temp_cover = tempfile.TemporaryFile()
    temp_secret = tempfile.TemporaryFile()

    try:
        count1 = 0
        count2 = 0
        print('Extracting Cover Video Frames...')
        # cover video to frames in a temporary file:
        while True:

            is_read, frame = cap1.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            if count1 == 0:

                first_cover_frame = frame
                count1 += 1
                continue

            temp_cover.write(frame.tobytes())
            count1 += 1

        print('total number of frames in cover =', count1)

        # secret video to frames in a temporary file:

        print('Extracting Secret Video Frames...')
        while True:
            is_read, frame = cap2.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            if count2 == 0:
                first_secret_frame = frame

            temp_secret.write(frame.tobytes())
            count2 += 1

        print('total number of frames in secret =', count2)

        # read secret audio
        signal, sr = sf.read(secret_audio, dtype='int16')
        length = len(signal)
        nof = count1-count2-1
        sof = first_cover_frame.size
        aud_meta = [length, sr]

        channel1 = np.array(signal)
        channel1 = channel1 + 32768
        channel1 = np.array(signal)
        channel1 = channel1 + 32768
        front = channel1/256
        end = front - front.astype(int)
        front = front.astype(int)

        end = end * 256
        end = end.astype(int)

        front = list(front)
        end = list(end)

        if not (cf.is_hideable(count1, count2)):
            print("The secret video can't be hidden in the cover video")

        else:

            if cf.audio_hideable(nof, sof, length):
                hide_audio = True

            else:
                hide_audio = False
                print("Audio in the Secret Video can't be hidden. To hide the audio as well, please select a larger cover video.\n1.Proceed Without Audio\n2.Cancel")
                choice = input('select any one: ')
                if choice != '1':
                    return None

            # Read one by one and hide:

            meta = [count2, fps2, list(reversed(first_secret_frame.shape[:2]))]
            meta = cf.flatten(meta)

            if hide_audio == True:
                for a in aud_meta:
                    meta.append(a)

            cover_frame_bytes = first_cover_frame.tobytes()
            secret_frame_bytes = first_secret_frame.tobytes()

            dimension = tuple(reversed(first_cover_frame.shape[:2]))
            stego_vid = cv2.VideoWriter(
                output_vid_path, cv2.VideoWriter_fourcc(*'PNG '), fps1, dimension)

            # hide metadata in the first frame:

            im = Image.fromarray(first_cover_frame)
            im.save('firstframe.png')

            metastr = cf.tostring(meta)

            print('Hiding Metadata...')
            encrypted = ed.encrypt(metastr, password)
            text_to_hide = cf.tostring(encrypted)
            Text_Steganography.encode(
                'firstframe.png', 'Text_hidden_inside_firstframe.png', text_to_hide)
            firstframe = cv2.imread('Text_hidden_inside_firstframe.png')

            stego_vid.write(firstframe)

            temp_cover.seek(0)
            temp_secret.seek(0)

            count1 = 0
            count2 = 0
            print('hiding secret video...')
            while True:

                # read cover frame
                read_cover_bytes = temp_cover.read(len(cover_frame_bytes))
                if read_cover_bytes == b'':
                    break
                cover_frame_obt = np.frombuffer(
                    read_cover_bytes, dtype=first_cover_frame.dtype)
                cover_frame_obt = cover_frame_obt.reshape(
                    first_cover_frame.shape)

                # hide

                # read secret frame

                read_secret_bytes = temp_secret.read(len(secret_frame_bytes))

                if read_secret_bytes == b'':
                    if hide_audio:
                        cover_image = Image.fromarray(
                            cover_frame_obt).convert('YCbCr')
                        Y, Cb, Cr = cover_image.split()
                        Cb = np.array(Cb)
                        Cr = np.array(Cr)

                        info_front = aiv.Audio_Encode(Cb, front, count1)
                        info_end = aiv.Audio_Encode(Cr, end, count2)

                        front = info_front['rem']
                        count1 = info_front['count']
                        Cb_obt = info_front['frame']

                        end = info_end['rem']
                        count2 = info_end['count']
                        Cr_obt = info_end['frame']

                        Cb_obt_img = Image.fromarray(Cb_obt)
                        Cr_obt_img = Image.fromarray(Cr_obt)

                        stego_img = Image.merge(
                            "YCbCr", (Y, Cb_obt_img, Cr_obt_img))

                        stego_img = np.array(stego_img)

                        stego_vid.write(stego_img)

                        if front == []:
                            hide_audio = False

                    else:
                        stego_vid.write(cover_frame_obt)

                else:

                    secret_frame_obt = np.frombuffer(
                        read_secret_bytes, dtype=first_secret_frame.dtype)
                    secret_frame_obt = secret_frame_obt.reshape(
                        first_secret_frame.shape)

                    cover = Image.fromarray(cover_frame_obt)
                    secret = Image.fromarray(secret_frame_obt)

                    transform = torchvision.transforms.Compose(
                        [torchvision.transforms.Resize(cover.size), torchvision.transforms.ToTensor()])
                    cover = transform(cover)
                    secret = transform(secret)

                    cover = cover.unsqueeze(0)
                    secret = secret.unsqueeze(0)
                    stego_frame = predict(_model, cover, secret, 'encoder')

                    # write stego_frame

                    stego_vid.write(stego_frame)

            temp_cover.close()
            temp_secret.close()

            # release the stego video

            stego_vid.release()

            print('Merging Audio...')
            subprocess.call(['ffmpeg', '-i', output_vid_path, '-i', audio_file,
                            '-c:v', 'png', '-c:a', 'aac', output_with_audio_path])

            os.remove('firstframe.png')
            os.remove('Text_hidden_inside_firstframe.png')
            os.remove(secret_audio)
            os.remove(audio_file)
            # os.remove(output_vid_path)
            print('The Stego Video is generated in the provided path')

            return output_with_audio_path

    except Exception as e:
        print(traceback.format_exc())
        temp_cover.close()
        temp_secret.close()


def Video_Decode(stego_vid_path, password, used_model):

    _model = joblib.load(used_model)

    cap = cv2.VideoCapture(stego_vid_path)

    is_read, frame = cap.read()

    filename, ext = os.path.splitext(stego_vid_path)

    reveal_vid_path = f'{filename}_(revealed){ext}'

    first_stego_frame = frame
    im = Image.fromarray(first_stego_frame)
    im.save('toreveal.png')

    extracted = Text_Steganography.decode('toreveal.png')

    encrypted = cf.tolist(extracted)
    decrypt_password = password
    try:
        decrypted = ed.decrypt(encrypted, decrypt_password)
        print('Extracting Meta data...')
        metastr_obt = bytes.decode(decrypted)
        meta_obt = cf.tolist(metastr_obt)
        meta_obt = [float(x) for x in meta_obt]
    except:

        print("Incorrect Password!!")
        return None

    count = int(meta_obt[0])
    fps = meta_obt[1]
    dimension = (int(meta_obt[2]), int(meta_obt[3]))

    reveal_vid = cv2.VideoWriter(
        reveal_vid_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, dimension)

    secret_size = (int(meta_obt[3]), int(meta_obt[2]))
    print('extracting secret frames...')
    for i in range(count):
        is_read, frame = cap.read()
        steg = Image.fromarray(frame)

        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(steg.size), torchvision.transforms.ToTensor()])
        steg = transform(steg)

        steg = steg.unsqueeze(0)

        reveal_frame = predict(_model, steg, steg, 'decoder', secret_size)

        reveal_vid.write(reveal_frame)

    if len(meta_obt) > 4:
        reveal_audio = True
        lenaudio = int(meta_obt[4])

        sr = int(meta_obt[5])

        print('extracting audio...')
        nop = lenaudio*4

        loop = math.ceil(nop/first_stego_frame.size)
        nop1 = nop
        nop2 = nop

        count1 = 0
        data1 = 0
        count2 = 0
        data2 = 0
        front_obt = []
        end_obt = []
        for i in range(loop):
            is_read, frame = cap.read()

            stego_image = Image.fromarray(frame)

            Y, Cb, Cr = stego_image.split()

            Cb = np.array(Cb)
            Cr = np.array(Cr)

            info1 = aiv.Audio_Decode(Cb, nop1, count1, data1)
            nop1 = nop1 - first_stego_frame.size
            count = info1['count']
            data = info1['lastdata']
            front_obt.append(info1['audio'])

            info2 = aiv.Audio_Decode(Cr, nop2, count2, data2)
            nop2 = nop2 - first_stego_frame.size
            count = info1['count']
            data = info1['lastdata']
            end_obt.append(info1['audio'])

        front_obt = cf.flatten(front_obt)
        end_obt = cf.flatten(end_obt)
        front_obt = np.array(front_obt)
        end_obt = np.array(end_obt)
        audio_obt = front_obt + (end_obt/256)
        audio_obt = (audio_obt*256).astype(int)
        audio_obt = (audio_obt-32768).astype('int16')

        sf.write('revealed_audio.wav', audio_obt, sr)

    else:
        reveal_audio = False

    print('merging frames...')
    reveal_vid.release()
    os.remove('toreveal.png')

    if reveal_audio:
        print('merging audio...')
        filename, ext = os.path.splitext(reveal_vid_path)
        output_path = f'/content/drive/MyDrive/Colab Notebooks/DEMO/Dependencies/outputVid{ext}'
        subprocess.call(['ffmpeg', '-i', reveal_vid_path, '-i',
                        'revealed_audio.wav', '-c:v', 'libx264', '-c:a', 'aac', output_path])
        os.remove('revealed_audio.wav')
        os.remove(reveal_vid_path)
        return output_path

    else:
        return reveal_vid_path
