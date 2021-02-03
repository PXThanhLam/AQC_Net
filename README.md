# AQC_Net
Implement of the paper: A deep learning and image-based model for air quality estimation. <br/>

Run python train.py --train_folder PATH_TO_YOUR_FOLDER --val_folder --PATH_TO_YOUR_FOLDER --num_label NUBER_OF_DESIRED_LABEL --batch_size YOUR_DESIRED_BATCH_SIZE
to train. <br/>

Default of num_label is 6 and batch_size is 32. <br/>

Your folder should organized like this: <br/>

ImgRoot/ <br/>
&nbsp; label1/ <br/>
&nbsp; &nbsp; img1.png <br/>
&nbsp; &nbsp; img2.png <br/>
&nbsp; &nbsp; .... <br/>
&nbsp; label2/ <br/>
&nbsp; &nbsp; img1.png <br/>
&nbsp; &nbsp; img2.png <br/>
&nbsp; &nbsp; .... <br/>
&nbsp; ... <br/>
