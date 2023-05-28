from PIL import Image
import numpy as np
from pydub import AudioSegment
import pywt
import soundfile as sf
import common_functions as cf
import encrypt_decrypt as ed
from text_in_image import Text_Steganography
from PIL import Image
import numpy as np
from pydub import AudioSegment
import pywt
import soundfile as sf
import cv2
import os
import traceback
import math
import tempfile
import subprocess


def Audio_Encode(img, audio_8b, count=0):

    shape = img.shape
    aud = audio_8b.copy()
    img = img.ravel()
    for i in range(len(img)):
        try:
            data = aud[0]
        except:
            break
        bits = data & 192
        bits = bits >> 6
        img[i] = ((img[i] & 0b11111100) | bits)

        count += 1
        data = data << 2

        aud[0] = data
        if count == 4:
            del aud[0]
            count = 0
    info = {
        'frame': img.reshape(shape),
        'rem': aud,
        'count': count,
    }
    return info


def Audio_Decode(encoded_img, nop, count=0, data=0):

    img = encoded_img.ravel()

    if nop < img.size:
        loop = nop
    else:
        loop = img.size

    audio_obt = []

    for i in range(loop):
        pix = img[i]
        bits = pix & 3
        data = data << 2 | bits
        count += 1
        if count == 4:
            audio_obt.append(data)
            data = 0
            count = 0

        info = {
            'audio': audio_obt,
            'count': count,
            'lastdata': data
        }

    return info


def AIV_Encode(cover_vid_path, secret_audio_path, password):

    # calculate fps of both

    cap1 = cv2.VideoCapture(cover_vid_path)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)

    cf.convert_video_to_audio(cover_vid_path)
    cover_filename, ext = os.path.splitext(cover_vid_path)
    audio_file = f'{cover_filename}.wav'
    output_vid_path = f'{cover_filename}_encoded{ext}'
    cf.to_mono(secret_audio_path)

    output_filename, ext = os.path.splitext(output_vid_path)
    output_with_audio_path = f'{output_filename}_with_audio{ext}'

    temp_cover = tempfile.TemporaryFile()

    try:
        count1 = 0

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

        # read secret audio
        signal, sr = sf.read(secret_audio_path, dtype='int16')
        length = len(signal)
        nof = count1-1
        sof = first_cover_frame.size

        channel1 = np.array(signal)
        channel1 = channel1 + 32768
        front = channel1/256
        end = front - front.astype(int)
        front = front.astype(int)

        end = end * 256
        end = end.astype(int)

        # Read one by one and hide:

        meta = [length, sr]

        cover_frame_bytes = first_cover_frame.tobytes()

        dimension = tuple(reversed(first_cover_frame.shape[:2]))

        stego_vid = cv2.VideoWriter(
            output_vid_path, cv2.VideoWriter_fourcc(*'PNG '), fps1, dimension)

        # hide metadata in the first frame:

        im = Image.fromarray(first_cover_frame)
        im.save('firstframe.png')

        metastr = cf.tostring(meta)
        encrypted = ed.encrypt(metastr, password)
        text_to_hide = cf.tostring(encrypted)
        Text_Steganography.encode(
            'firstframe.png', 'Text_hidden_inside_firstframe.png', text_to_hide)
        firstframe = cv2.imread('Text_hidden_inside_firstframe.png')

        stego_vid.write(firstframe)
        print("Text Steganography")
        temp_cover.seek(0)

        count1 = 0
        count2 = 0
        hide = True
        loopcount = 0
        while True:

            # read cover frame
            read_cover_bytes = temp_cover.read(len(cover_frame_bytes))
            if read_cover_bytes == b'':
                break
            cover_frame_obt = np.frombuffer(
                read_cover_bytes, dtype=first_cover_frame.dtype)
            cover_frame_obt = cover_frame_obt.reshape(first_cover_frame.shape)

            # hide

            if hide:
                cover_image = Image.fromarray(cover_frame_obt).convert('YCbCr')
                Y, Cb, Cr = cover_image.split()
                Cb = np.array(Cb)
                Cr = np.array(Cr)

                info_front = Audio_Encode(Cb, list(front), count1)
                info_end = Audio_Encode(Cr, list(end), count2)

                front = info_front['rem']
                count1 = info_front['count']
                Cb_obt = info_front['frame']

                end = info_end['rem']
                count2 = info_end['count']
                Cr_obt = info_end['frame']

                Cb_obt_img = Image.fromarray(Cb_obt)
                Cr_obt_img = Image.fromarray(Cr_obt)

                stego_img = Image.merge("YCbCr", (Y, Cb_obt_img, Cr_obt_img))

                stego_img = np.array(stego_img)

                stego_vid.write(stego_img)

                if front == []:
                    hide = False

            else:
                stego_vid.write(cover_frame_obt)

            loopcount += 1

            if loopcount % 1000 == 0:
                print('loop: ', loopcount)

        temp_cover.close()
        print("Releasing video...")
        # release the stego video

        stego_vid.release()

        os.remove('firstframe.png')
        os.remove('Text_hidden_inside_firstframe.png')

        subprocess.call(['ffmpeg', '-i', output_vid_path, '-i', audio_file,
                        '-c:v', 'png', '-c:a', 'aac', output_with_audio_path])

        os.remove(audio_file)
        os.remove(output_vid_path)
        print("video released")

        return output_with_audio_path

    except Exception as e:
        print(traceback.format_exc())
        temp_cover.close()


def AIV_Decode(stego_vid_path, password):

    cap = cv2.VideoCapture(stego_vid_path)

    is_read, frame = cap.read()

    filename, ext = os.path.splitext(stego_vid_path)
    reveal_aud_path = f'{filename}_revealed.wav'

    first_stego_frame = frame
    im = Image.fromarray(first_stego_frame)
    im.save('toreveal.png')

    extracted = Text_Steganography.decode('toreveal.png')

    encrypted = cf.tolist(extracted)
    decrypt_password = password
    try:
        decrypted = ed.decrypt(encrypted, decrypt_password)
        metastr_obt = bytes.decode(decrypted)
        meta_obt = cf.tolist(metastr_obt)
        meta_obt = [float(x) for x in meta_obt]
    except Exception as e:
        print(traceback.format_exc())
        print("Incorrect Password!!")
        return None

    lenaudio = int(meta_obt[0])
    sr = int(meta_obt[1])

    print('lenaudio:', lenaudio)
    print('sr: ', sr)
    print('size of frame: ', first_stego_frame.size)
    nop = lenaudio*4
    loop = math.ceil(nop/first_stego_frame.size)
    print('loop: ', loop)
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

        info1 = Audio_Decode(Cb, nop1, count1, data1)
        nop1 = nop1 - first_stego_frame.size
        count = info1['count']
        data = info1['lastdata']
        front_obt.append(info1['audio'])

        info2 = Audio_Decode(Cr, nop2, count2, data2)
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

    sf.write(reveal_aud_path, audio_obt, sr)

    os.remove('toreveal.png')

    return reveal_aud_path
