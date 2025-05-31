# This is the Pixel-to-Point dataset for the submission of 'Pixel-to-Point: A Dataset for Enhancing Multimedia Applications'.
## Our Pixel-to-Point dataset can be found on google drive:https://drive.google.com/drive/folders/1hP6jhr42TlFCtfqFWQMuk0iazTpghNpd?usp=sharing

### Running
**MOGP**:
```shell
python top_four_contribution.py #Find the image from a perspective that contributes most to SfM points cloud.
python mogp_train.py #Training MOGP model
python predict.py #Predict high quality dense points cloud.
```
