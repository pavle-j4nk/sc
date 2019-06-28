from filter.graystyle_filter import GraystyleFilter
from filter.number_areas_filter import NumberAreaFilter
from frame_reader import VideoReader
from line import Line
from number_recognizer import NumberRecognizer
from simple_frame_display import SimpleFrameDisplay

out = open("./data/out.txt", "w")
out.write("RA 124/2015 Pavle Jankovic\n")
out.write("file\tsum\n")

for i in range(0, 10):
    file = "video-" + str(i) + ".avi"
    file_location = "./data/" + file

    vr = VideoReader(file_location, SimpleFrameDisplay())

    nr = NumberRecognizer()

    naf = NumberAreaFilter(nr)

    blue_line = Line.blue(naf)
    green_line = Line.green(naf)

    # vr.add_filter(NoiseReductionFilter())
    vr.add_filter(GraystyleFilter())
    vr.add_filter(naf)
    vr.add_filter(blue_line)
    vr.add_filter(green_line)
    vr.read_all_frames()

    sum = blue_line.get_sum() - green_line.get_sum()
    print(file + "\t" + str(sum))
    out.write(file + "\t" + str(sum) + "\n")

out.close()