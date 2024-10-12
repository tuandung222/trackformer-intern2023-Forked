- Please don't give me any questions because I'm not able to answer them. I'm sorry for any inconvenience.
- I'm sorry that I don't release the data at my last intern in 2023 due to the company's policy. But I will guide you through the process of training and tracking with TrackFormer. I will also provide some of my code for you to understand the process. I hope this will help you to understand the process of training and tracking with TrackFormer. If you have any question, feel free to ask me. I will try to help you as much as I can.
- I have put some qualitative results in gif format that is lacated in `output_videos` and `output_gifs` folder. You can see the results of my tracking with TrackFormer.
- This is my old guild for training and tracking with TrackFormer. I have modify the code to work with multiple-class tracking. The code is not clean, but it works. I will update the code later.
    1. Raw data for training, before splitting is located in "bktris_training_data" folder.
    2. Label for training data is located in "aws_annotations".
    3. Trained model for vehicle tracking (4 class) is located in "training_results" folder. Choose the best MOTA, or best IDF1 model for tracking.
    4. Config for training and tracking are located in cfgs, open "my_train.yaml" and "my_track.yaml" for detailed.
    5. If you want to run some of my experiment notebooks, you should put the files in the right folder. Some of my notebooks help orgranizing data into MOT17 style, and train/validate splitting.
    6. I have modify "src/generate_coco_from_mot.py" to generate MOT17 's style data format to Coco format. The data folder for training and validation always put in "data" folder.
    7. I have implement BKTRISSequence to put change MOT17 's style data format to right tracking format. Read the source code for detail.
    8. I have modify some file for visuliazation with wandb, the config setting is in in train.py. If you choose to visualize with visdom, start the server before training.
    9. I have modify track_utils and tracker to work with multiple-class tracking. I also have modify classifier, loss function. For tunning model, read full source code in src folder!
    10. Evaluate video data and results are located in "bktris_evaluate_data"
