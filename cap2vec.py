import skipthoughts
import pandas as pd
import numpy as np


input_filename = "input_file.csv"
output_folder  = "./textVectors/"


# Load captions and file names.
def gtFileLoad(gt_filename):
    ground_truth_file = pd.read_csv(gt_filename, names \
    =['file_name', 'captions'], skiprows=1)
    ground_truth_file = ground_truth_file.convert_objects(convert_numeric=True)
    print 'Ground truth file  loaded'
    return ground_truth_file

# convert captions to skip-thought vectors.
def caption2Vectors(video_filename, video_captions):
    
    captions  = []
    video_ctr = 0

    #Prepare processed lsit of captions.
    for caption in video_captions:
        caption = caption.decode('utf-8').strip()
        caption = caption.strip()
        caption = caption.rstrip('\n')
        captions.append(caption)

    print "Total number of captions are: "
    print len(captions)
    print "Initializing Skipthoughts vectorizer."

    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)

    #first 2400 -> uni-skip model, last 2400 ->bi-skip model
    print "Starting Vectorizer conversion."
    vectors = []

    print "Skip-thought vector size is :" 
    vectors = encoder.encode(captions)
    print vectors.shape

    #save captions as numpy files using corresponding filenames.
    for caption in vectors:
        file_name_video = video_filename[video_ctr]
        output_file = output_folder+str(file_name_video)+".npy"
         
        with open(output_file, 'w') as f2:
            np.save(f2, caption)
            print str(file_name_video) + "    extracted"
        video_ctr+=1    


ground_truth_file = gtFileLoad(input_filename)

print " Hello, Im a small script to covnert sentences to vectors using skip-thought vectors."


caption2Vectors(ground_truth_file['file_name'], ground_truth_file['captions'])
print "vectorization complete!"