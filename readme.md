# Caption to Skipthought vectors + classifer

The following scripts are used to convert text video captions to skipthought vectors and classify them using support vector machines with 5 fold CV. 
The video captions were manually collected inorder to augment the LIRIS ACCEDE Dataset [http://liris-accede.ec-lyon.fr/].
For access to  email proof of approval of LIRIS ACCEDE dataset to tpt7797 at rit dot edu.

## Requirements

1. skipthoughts.py [https://github.com/ryankiros/skip-thoughts]
2. The following python libraries:    
    1. Numpy
    2. Pandas
    3. ScikitLearn
3. CSV file in the following format. filename, captions
4. (optional) for calssifier script filename, label 

## Getting Started

### Captions -> Vectors
1. Change line #6 to match your input file.
2. Optionally, change output folder in line #7
3. Grab the uni and bi models from the gettign started section in [https://github.com/ryankiros/skip-thoughts].

### Vectors-> Classifier
Change lines 9-16 and line 23 as needed.

## Running 
### Captions -> Vectors
```
python cap2skip.py
```
### Vectors-> Classifier
```
python Text_svm.py
```
## To Do
1. Flesh out documentation.
2. link back to main.

## Citation
If this helps, please consider citing :)

```
@article{thomas2017emotional,
  title={The Emotional Impact of Audio-Visual Stimuli},
  author={Thomas, Titus Pallithottathu},
  year={2017}
}
```
