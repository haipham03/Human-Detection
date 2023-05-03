# XLA-Human-Detection

## Team members

* Nguyễn Tiến Hùng
* Phạm Hoàng Hải
* Mai Thanh Hải
* Dương Hoàng Hải
* Nguyễn Hữu Đạt

## Dependencies

* OpenCV
* scikit-image ```pip install scikit-image==0.19.3```
* scikkit-learn ```pip install scikit-learn==0.20.2```

## Running Detection

To test on images run, `python detect.py -i <path to image>`

For example, `python detect.py -i sample_images/crop001671.png`

## Example input and expected output

These are some examples outputs of the code

![Pedestrian](sample_images/crop001671.png?raw=true "Sample Results")
![Pedestrian](sample_images/output_001671.png?raw=true "Sample Results")

![Pedestrian](sample_images/person_and_bike_208.png?raw=true "Sample Results")
![Pedestrian](sample_images/output_pb208.png?raw=true "Sample Results")
