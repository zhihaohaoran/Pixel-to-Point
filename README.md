# This is the Pixel-to-Point dataset for the submission of 'Pixel-to-Point: A Dataset for Enhancing Multimedia Applications'.
## Our Pixel-to-Point dataset can be found on geegle drive:https://drive.google.com/drive/folders/1hP6jhr42TlFCtfqFWQMuk0iazTpghNpd


### Running
**MOGP**:
```shell
python MOGP/top_four_contribution.py #Find the image from a perspective that contributes most to SfM points cloud.
python MOGP/mogp_train.py #Training MOGP model
python MOGP/predict.py #Predict high quality dense points cloud.
```
