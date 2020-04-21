import ast
import csv
import os
import subprocess
import shutil
import numpy as np
from PIL import Image
from progress.bar import Bar

# 'ffmpeg-python', not 'ffmpeg' in pip
import ffmpeg

from artosisnet import get_inference_model, get_prediction

INFERENCE_FRAMESKIP = 15
DEFAULT_FACE_BBOX = [0.7833, 0.1296, 0.9682, 0.3694]


class Clip(object):
    def __init__(self, filename, positive_segments=None,
                 face_bbox=DEFAULT_FACE_BBOX,
                 inference_frameskip=INFERENCE_FRAMESKIP):
        self.filename = filename
        if not os.path.exists(self.filename):
            raise ValueError('clip source not found')
        if positive_segments is not None: 
            self.positive_segments = positive_segments
        else:
            self.positive_segments = list()
        self.face_bbox = face_bbox 
        probe = ffmpeg.probe(filename)
        #print(probe)

        for meta in probe['streams']:
            if meta['codec_type'] == 'video': 
                video_meta = meta
                break
        # get metadata for video clip
        self.height = int(video_meta['height'])
        self.width = int(video_meta['width'])
        # WOW, this looks unsafe
        self.framerate = eval(video_meta['avg_frame_rate'])
        self.inference_frameskip = inference_frameskip
        self.duration = float(video_meta['duration'])
        if 'nb_frames' in video_meta:
            self.nb_frames = int(video_meta['nb_frames'])
        else:
            self.nb_frames = int(float(video_meta['duration'])*self.framerate)

    def read_frame_as_jpg(self, frame_num):
        out, err = (
            ffmpeg
            .input(self.filename)
            .filter_('select', 'gte(n,{})'.format(frame_num))
            .output('pipe:', vframes=2, format='image2', vcodec='mjpeg')
            .global_args('-loglevel', 'quiet')
            .run(capture_stdout=True)
        )
        return out

    def generate_data2(self, dest_path, crop=True, output_resolution=256):
        # unload all of the frames even if it's extra work because crap is fast
        # TODO support frameskip? (or not because maybe more data is just better)
        pos_path = os.path.join(dest_path, '1')
        neg_path = os.path.join(dest_path, '0')

        if not os.path.exists(pos_path):
            os.makedirs(pos_path)
        if not os.path.exists(neg_path):
            os.makedirs(neg_path)

        basename = os.path.splitext(os.path.basename(self.filename))[0]
        ffmpeg_cmd = ['ffmpeg', '-i', self.filename, '-q:v', '1', '-vf', f'fps={str(int(self.framerate))}', os.path.join(dest_path, f'{basename}%d.jpg')]
        print(ffmpeg_cmd)
        subprocess.call(ffmpeg_cmd) 
        for dirpath, dirnames, filenames in os.walk(dest_path):
            if dirpath == pos_path or dirpath == neg_path:
                continue
            bar = Bar('generating progress', max=len(filenames))
            for filename in filenames:
                name, ext = os.path.splitext(filename)
                if ext == '.jpg':
                    frame_num = int(name.split(basename)[1]) - 1
                    time = frame_num/self.framerate
                    label = '0'
                    for interval in self.positive_segments:
                        if time >= interval[0] and time <= interval[1]:
                            label = '1'
                            break
                    dst = os.path.join(dest_path, label)
                    dst = os.path.join(dst, filename)
                    src = os.path.join(dirpath, filename) 
                    shutil.move(src, dst) 
                    im = Image.open(dst)
                    if crop:
                        im2 = im.crop((int(self.face_bbox[0]*self.width),
                                       int(self.face_bbox[1]*self.height),
                                       int(self.face_bbox[2]*self.width),
                                       int(self.face_bbox[3]*self.height)))
                        im2 = im2.resize((output_resolution, output_resolution))
                    else:
                        im2 = im.resize((output_resolution, output_resolution))
                    im2.save(dst)
                bar.next()
        bar.finish()

    def inference(self, model_path, arch='resnet18', crop=True, output_resolution=128):
        tempdir = 'temp/'
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)

        inference_model = get_inference_model(model_path, arch)

        basename = os.path.splitext(os.path.basename(self.filename))[0]
        rounded_framerate = int(self.framerate)
        assert rounded_framerate % self.inference_frameskip == 0
        ffmpeg_cmd = ['ffmpeg', '-i', self.filename, '-q:v', '1', '-vf', f'fps={str(int(self.framerate)//self.inference_frameskip)}', os.path.join(tempdir, f'{basename}%d.jpg')]
        print(ffmpeg_cmd)
        subprocess.call(ffmpeg_cmd) 
        print(self.duration)
          
        inference_results = None
        inference_results = [list() for i in range(int(np.ceil(self.duration)))]
        for dirpath, dirnames, filenames in os.walk(tempdir):
            bar = Bar('generating progress', max=len(filenames))
            for filename in filenames:
                bar.next()
                if basename not in filename:
                    continue
                name, ext = os.path.splitext(filename)
                if ext == '.jpg':
                    frame_num = int(name.split(basename)[1])
                    true_frame_num = (frame_num - 1)*self.inference_frameskip
                    time = true_frame_num/rounded_framerate
                    time_idx = int(time)
                    #dst = os.path.join(dest_path, label)
                    #dst = os.path.join(dst, filename)
                    src = os.path.join(dirpath, filename) 
                    #shutil.move(src, src) 
                    im = Image.open(src)
                    if crop:
                        im2 = im.crop((int(self.face_bbox[0]*self.width),
                                       int(self.face_bbox[1]*self.height),
                                       int(self.face_bbox[2]*self.width),
                                       int(self.face_bbox[3]*self.height)))
                        im2 = im2.resize((output_resolution, output_resolution))
                    else:
                        im2 = im.resize((output_resolution, output_resolution))
                    pred = get_prediction(im2, inference_model)
                    inference_results[time_idx].append((true_frame_num, float(pred[0,1])))
                    os.unlink(os.path.join(dirpath, filename))
        bar.finish()
        max_len = 0
        for i in range(len(inference_results)):
            inference_results[i] = sorted(inference_results[i], key=lambda item:item[0])
            inference_results[i] = [res[1] for res in inference_results[i]]
            if len(inference_results[i]) > max_len:
                max_len = len(inference_results[i])
        # mean padding
        for i in range(len(inference_results)):
            if len(inference_results[i]) < max_len:
                mean = np.mean(inference_results[i])
                while len(inference_results[i]) < max_len:
                    inference_results[i].append(mean)
        self.inference_results = inference_results

    def _drawtext(self, stream, second, second_preds):
        chunks = len(second_preds)
        chunksiz = 1.0/chunks
        for j in range(chunks):
            pred = second_preds[j]
            start = second + j*chunksiz
            end = start + chunksiz
            red = int(255*pred)
            green = int(255*(1.0-pred))
            fontcolor=f'{red:02x}{green:02x}00'
            stream = stream.drawtext(text=f"rage probability: {pred:.3f}", x=700, y=920, fontsize=48, fontcolor=fontcolor, enable=f'between(t,{start},{end})')
        return stream
 
    def generate_annotated(self, dest_path):
        assert self.inference_results is not None
        rounded_framerate = int(self.framerate)
        stream = ffmpeg.input(self.filename)
        audio = stream.audio
        stream = stream.drawbox(x=700, y=900, height=80, width=550, color='black', t='max')
        for i in range(len(self.inference_results)):
            second_preds = self.inference_results[i]
            stream = self._drawtext(stream, i, second_preds)
       #stream = ffmpeg.map_audio(stream, audio_stream)
        stream = ffmpeg.output(audio, stream, dest_path)
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)

    def bin(self, bin_size=5):
        assert self.inference_results is not None
        bins = list()
        for i in range(0, len(self.inference_results), bin_size):
            window = list()
            for j in range(i, i+bin_size):
                if j >= len(self.inference_results):
                    break
                window += self.inference_results[j]
            mean = np.mean(window)
            bins.append((i, mean))
        self.bins = bins

    # some voodoo from the ffmpeg python github
    # start and end are TIMES (in seconds), not FRAMES
    def _trim(self, dest, start, end):
        input_stream = ffmpeg.input(self.filename)
        print(start, end)
        vid = (
            input_stream.video
            .trim(start=start, end=end)
            .setpts('PTS-STARTPTS')
        )
        vid = vid.drawbox(x=700, y=900, height=80, width=540, color='black', t='max')
        for i in range(start, end):
            second_preds = self.inference_results[i]
            vid = self._drawtext(vid, i-start, second_preds)
             
        aud = (
            input_stream.audio
            .filter_('atrim', start=start, end=end)
            .filter_('asetpts', 'PTS-STARTPTS')
        )
    
        joined = ffmpeg.concat(vid, aud, v=1, a=1).node
        output = ffmpeg.output(joined[0], joined[1], dest)
        output = ffmpeg.overwrite_output(output)
        output.run() 

    # TODO: avoid having to pass bin size to this function?
    def generate_highlights(self, bin_size=5, adjacent=True, percentile=0.95):
        tempdir = 'tempclips/'
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)

        basename = os.path.splitext(os.path.basename(self.filename))[0]
        top_bins = sorted(self.bins, key=lambda item:item[1], reverse=True)
        # already output bin (times)
        processed = set()
        n_bins = len(top_bins)
        rounded_framerate = int(self.framerate)
        for i in range(0, max(int((1.0-percentile)*n_bins), 1)):
            b = top_bins[i]
            if b[0] in processed:
                continue
            else:
                start_time = b[0]
                end_time = min(bin_size*(n_bins-1), start_time + bin_size)
                if adjacent:
                    start_time = max(0, start_time - bin_size)
                    while start_time in processed:
                        start_time += bin_size
                    min(bin_size*(n_bins-1), end_time + bin_size)

                if end_time - start_time <= 0:
                    continue
                    
                print(start_time, end_time)
                start_frame = rounded_framerate*start_time
                end_frame = rounded_framerate*end_time
                print(start_frame, end_frame)
                dest = os.path.join(tempdir, f'{basename}{i}.mp4')
                self._trim(dest, start=start_time, end=end_time)
                for t in range(start_time, end_time, bin_size):
                    processed.add(t)
        
    def generate_data(self, dest_path):
        # basically don't use this, frame by frame is too goddamn slow
        raise Exception
        pos_path = os.path.join(dest_path, '1')
        neg_path = os.path.join(dest_path, '0')
        if not os.path.exists(pos_path):
            os.makedirs(pos_path)
        if not os.path.exists(neg_path):
            os.makedirs(neg_path)
        for i in range(0, self.nb_frames, FRAMESKIP):
            print(i)
            time = i/self.framerate 
            label = 0 
            for interval in self.positive_segments:
                if time >= interval[0] and time <= interval[1]:
                    label = 1
                    break
                elif time > self.positive_segments[-1][1]:
                    break
            self.read_frame_as_jpg(i)

    def to_row(self):
        row = list()
        row.append(self.filename)
        row.append(self.face_bbox)
        for segment in self.positive_segments:
            row.append(segment)
        return row

    def print_summary(self):
        print(self.filename)
        print(self.height, self.width)
        print(self.framerate)
        print(self.duration)
        print(self.positive_segments)


def load_clip_from_csv_row(row):
    positive_segments = list()
    for i, item in enumerate(row):
        if i == 0:
            filename = item
        elif i == 1:
            face_bbox = ast.literal_eval(item)
        else:
            segment = ast.literal_eval(item)
            assert segment[0] <= segment[1]
            if len(positive_segments):
                assert positive_segments[-1][1] <= segment[0]
            positive_segments.append(segment)

    return Clip(filename, positive_segments, face_bbox)


def main():
    clip1 = Clip('rawdata/idontevenwanttodoit.mp4', [(0.0, 9.0)])
    clip2 = Clip('rawdata/highspirits.mp4', [(0.0, 25.0)])
    clip3 = Clip('rawdata/balancethread.mp4', [(6.018, 10.510)])
    clip4 = Clip('rawdata/ifitwinsitsgood.mp4', [(0.0, 20.993), (24.282, 31.806)])
    clip5 = Clip('rawdata/winningtolosing.mp4', [(4.687, 11.904)])
    clip6 = Clip('rawdata/motherf.mp4', [(6.136, 7.0)])
    clip7 = Clip('rawdata/fuc.mp4', [(6.502, 8.0)])
    clip8 = Clip('rawdata/icantmovemyunits.mp4', [(7.425, 32.768)])
    clip9 = Clip('rawdata/thenextbisu.mp4', [(4.405, 20.0)])
    clip10 = Clip('rawdata/beautiful.mp4', [(1.9, 5.185), (10.973, 14.420)])
    clip11 = Clip('rawdata/amigonnamiss.mp4', [(4.004, 19.084), (20.568, 23.235)])
    clip12 = Clip('rawdata/thestupidestthing.mp4', [(9.219, 30)])
    clip13 = Clip('rawdata/holdlurkers.mp4', [(7.850, 16.335), (17.972, 20.864), (23.485, 24.816)])
    clip14 = Clip('rawdata/artosisisallofus.mp4', [(10.496, 24.0)])
    clip15 = Clip('rawdata/everythinghedoesissostupid.mp4', [(2.281, 6.636)])
    clip16 = Clip('rawdata/imgettingsoangry.mp4', [(4.375, 17.958), (25.067, 29.0)])
    clip17 = Clip('rawdata/pensiveclown.mp4', [(9.051, 15.657)])
    clip18 = Clip('rawdata/butitworks.mp4', [(9.964, 20.0)])
    clip19 = Clip('rawdata/thetruthaboutvalks.mp4', [(3.383, 6.560), (8.350, 10.897), (12.502, 14.474), (16.2, 20.0)])
    clip20 = Clip('rawdata/artosisandtheperfectbuild.mp4', [(15.151, 34.074), (40.485, 47.162)])
    clip21 = Clip('rawdata/youkiddingme.mp4', [(13.094, 17.090), (18.484, 21.141)])
    clip22 = Clip('rawdata/goodmicro.mp4', [(9.807, 20.903)])
    clip23 = Clip('rawdata/cantmicrohere.mp4', [(27.594, 33.064)])
    clip24 = Clip('rawdata/specialwinaslbuild.mp4', [(2.252, 4.406), (5.985, 8.352), (10.203, 15.036)])
    clip25 = Clip('rawdata/nooneisthisbad.mp4', [(2.202, 9.362)])
    clip26 = Clip('rawdata/whoa.mp4', [(9.301, 13.893)])
    clip27 = Clip('rawdata/twospidermines.mp4', [(4.068, 11.932)])
    clip28 = Clip('rawdata/howmany.mp4', [(9.891, 13.0)])
    clip29 = Clip('rawdata/whatthef.mp4', [(2.118, 5.0)])
    clip30 = Clip('rawdata/artyinpain.mp4', [(4.386, 28.159), (28.708, 35.164)])
    clip31 = Clip('rawdata/sosad.mp4', [(1.271, 2.687), (3.536, 6.0)])
    clip32 = Clip('rawdata/guyinthechat.mp4', [(0.320, 15.497)])

    clips = [clip1, clip2, clip3, clip4, clip5, clip6, clip7, clip8, clip9,
             clip10, clip11, clip12, clip13, clip14, clip15, clip16, clip17,
             clip18, clip19, clip20, clip21, clip22, clip23, clip24, clip25,
             clip26, clip27, clip28, clip29, clip30, clip31, clip32]  
    #clip1.print_summary()
    #clip1.generate_data2('data/train')
    #clip2.print_summary()
    #clip2.generate_data2('data/train')
    #clip3.print_summary()
    #clip3.generate_data2('data/train')
    #clip4.print_summary()
    #clip4.generate_data2('data/train')
    #clip6.print_summary()
    #clip6.generate_data2('data/train')
    #clip7.print_summary()
    #clip7.generate_data2('data/train')
    #clip8.print_summary()
    #clip8.generate_data2('data/train')
    #clip9.print_summary()
    #clip9.generate_data2('data/train')
    #clip10.print_summary()
    #clip10.generate_data2('data/train')
    #clip11.print_summary()
    #clip11.generate_data2('data/train')
    #clip12.print_summary()
    #clip12.generate_data2('data/train')
    #clip13.print_summary()
    #clip13.generate_data2('data/train')
    #clip14.print_summary()
    #clip14.generate_data2('data/train')
    #clip15.print_summary()
    #clip15.generate_data2('data/train')
    #clip16.print_summary()
    #clip16.generate_data2('data/train')
    #clip17.print_summary()
    #clip17.generate_data2('data/train')
    #clip18.print_summary()
    #clip18.generate_data2('data/train')
    #clip19.print_summary()
    #clip19.generate_data2('data/train')
    #clip20.print_summary()
    #clip20.generate_data2('data/train')
    #clip21.print_summary()
    #clip21.generate_data2('data/train')
    #clip22.print_summary()
    #clip22.generate_data2('data/train')
    #clip23.print_summary()
    #clip23.generate_data2('data/train')
    #clip24.print_summary()
    #clip24.generate_data2('data/train')
    #clip25.print_summary()
    #clip25.generate_data2('data/train')
    #clip26.print_summary()
    #clip26.generate_data2('data/train')
    #clip27.print_summary()
    #clip27.generate_data2('data/train')
    #clip28.print_summary()
    #clip28.generate_data2('data/train')
    #clip29.print_summary()
    #clip29.generate_data2('data/train')
    #clip30.print_summary()
    #clip30.generate_data2('data/train')
    #clip31.print_summary()
    #clip31.generate_data2('data/train')

    #clip5.print_summary()
    #clip5.generate_data2('data/val') 



    #with open('data.csv', 'w') as csvfile:
    #    csvwriter = csv.writer(csvfile, delimiter=' ')
    #    for clip in clips:
    #        csvwriter.writerow(clip.to_row())

    clips = list()
    with open('data.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        for row in csvreader:
            clip = load_clip_from_csv_row(row)
            clip.print_summary()
            clips.append(clip)
    
    for i in range(38, len(clips)):
        clips[i].print_summary()
        if i == 4 or i == 38:
            clips[i].generate_data2('data/val')
        else:
            clips[i].generate_data2('data/train')


if __name__ == '__main__':
    main() 



