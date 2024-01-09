import kfp
import kfp.dsl as dsl

from typing import Dict, List

from kfp import compiler
from kfp.dsl import Artifact, Input, Output

videos = [
    "gs://YOUR_BUCKET_NAME/Noam_Chomsky_2011_interview_part_4.ogv",
]

@dsl.component(
    base_image='python:3.11',
    packages_to_install=[
        'google-cloud-aiplatform',
        'opencv-python-headless'
    ]
)
def analyze_images(video_filepath: str) -> List:
    import cv2
    import datetime
    import json
    import os

    from google.cloud import aiplatform

    aiplatform.init()

    DIRECTORY = video_filepath.replace("gs://YOUR_BUCKET_NAME/", "").replace(".ogv", "")

    # Open video file and get the frame rate (FPS)
    cap = cv2.VideoCapture(video_filepath.replace("gs://", "/gcs/"))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise Exception("Could not open the video file")

    frames = []
    # Get one frame per second (~1/60 sampling)
    for i in range(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps)):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read frame
        ret, frame = cap.read()

        if ret:
            # Write frame to GCS
            if not os.path.exists(f"/gcs/YOUR_BUCKET_NAME/{DIRECTORY}"):
                os.makedirs(f"/gcs/YOUR_BUCKET_NAME/{DIRECTORY}")
            cv2.imwrite(
                f"/gcs/YOUR_BUCKET_NAME/{DIRECTORY}/{i}.png",
                frame
            )
            frames.append(
                f"gs://YOUR_BUCKET_NAME/{DIRECTORY}/{i}.png"
            )
        else:
            print("Could not read frame:", i)

    # Release video
    cap.release()

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create JSON file with frame paths
    with open(f"/gcs/YOUR_BUCKET_NAME/{DIRECTORY}/frames-{now}.jsonl", "a") as f:
        for frame in frames:
            # Dump JSON with a newline
            f.write(json.dumps({
                "content": frame,
                "mimeType": "image/png"
            }) + "\n")

    # Run Vertex AI batch prediction
    classification_model = aiplatform.Model(
        "projects/712655513370/locations/us-central1/models/3225120491941396480"
    )

    batch_prediction_job = classification_model.batch_predict(
        job_display_name=f"YOUR_BUCKET_NAME-{now}",
        gcs_source=f"gs://YOUR_BUCKET_NAME/{DIRECTORY}/frames-{now}.jsonl",
        gcs_destination_prefix=f"gs://YOUR_BUCKET_NAME/{DIRECTORY}/predictions",
        sync=True,
    )

    batch_prediction_job.wait()

    # Get prediction results by walking the gcs_output_directory
    output_directory = batch_prediction_job.output_info.gcs_output_directory
    predictions = []
    for root, dirs, files in os.walk(output_directory.replace("gs://", "/gcs/")):
        for file in files:
            if file.endswith(".jsonl"):
                with open(os.path.join(root, file), "r") as f:
                    for line in f.readlines():
                        predictions.append(json.loads(line))

    # Sort predictions by frame_number (derived from instance.content)
    predictions.sort(key=lambda x: int(x["instance"]["content"].split("/")[-1].replace(".png", "")))

    # Get chapter blocks
    trailing_display_name = None
    blocks = []
    max_frame = 0
    for prediction in predictions:
        # Get the associated displayName for the prediction.confidences
        confidences = prediction["prediction"]["confidences"]
        display_names = prediction["prediction"]["displayNames"]
        for i, confidence in enumerate(confidences):
            if confidence == max(confidences):
                # Get the display name and frame number from the prediction
                display_name = display_names[i]
        frame_number = int(prediction["instance"]["content"].split("/")[-1].replace(".png", ""))
        max_frame = max(max_frame, frame_number)
        # If the display name is different from the previous one, we have a new categorization block
        if display_name != trailing_display_name:
            blocks.append({
                "start": frame_number,
                "end": frame_number,
                "display_name": display_name
            })
        else:
            # Update the end frame number
            blocks[-1]["end"] = frame_number
        # Update the trailing display name
        trailing_display_name = display_name

    # Print chapter blocks
    for block in blocks:
        display_name = block["display_name"]
        start = datetime.timedelta(seconds=block["start"] / fps)
        end = datetime.timedelta(seconds=block["end"] / fps)
        print(f"{display_name}: {start} - {end}")

    return blocks

@dsl.pipeline(
    name="video-chapters-inference"
)
def chapter_identification():
    with dsl.ParallelFor(
        name="videos",
        items=videos,
        # Parallelism driven by max concurrency quota
        parallelism=5
    ) as video:
        analyze_images_task = analyze_images(
            video_filepath=video,
        )

compiler.Compiler().compile(chapter_identification, 'pipeline.json')