- Please refrain from asking me questions as I may not be able to answer them. I apologize for any inconvenience. Also, please don't inquire about the code's appearance; I only started researching AI three months before my internship.

- Unfortunately, I cannot release the data from my last internship in 2023 due to company policy. However, I will guide you through the training and tracking process with TrackFormer and provide some of my code to help you understand the process. Feel free to ask any questions, and I will do my best to assist you.
- I have included some qualitative results in GIF format in the `output_videos` and `output_gifs` folders. You can view the tracking results with TrackFormer there.
- This is my previous guide for training and tracking with TrackFormer. I have modified the code to support multi-class tracking. The code is not clean but functional, and I will update it later.
    1. Raw training data, before splitting, is located in the "bktris_training_data" folder.
    2. Training data labels are in the "aws_annotations" folder.
    3. The trained model for vehicle tracking (4 classes) is in the "training_results" folder. Choose the best MOTA or best IDF1 model for tracking.
    4. Training and tracking configurations are in the cfgs folder. Open "my_train.yaml" and "my_track.yaml" for details.
    5. To run some of my experiment notebooks, place the files in the correct folders. Some notebooks help organize data into MOT17 style and handle train/validate splitting.
    6. I have modified "src/generate_coco_from_mot.py" to convert MOT17 style data format to COCO format. Training and validation data should be placed in the "data" folder.
    7. I have implemented BKTRISSequence to convert MOT17 style data format to the correct tracking format. Refer to the source code for details.
    8. I have modified some files for visualization with wandb. The configuration settings are in train.py. If you choose to visualize with visdom, start the server before training.
    9. I have modified track_utils and tracker for multi-class tracking. I also modified the classifier and loss function. For model tuning, refer to the full source code in the src folder.
    10. Evaluation video data and results are in the "bktris_evaluate_data" folder.