import os
import sys
from pathlib import Path
import shutil
from typing import Optional, Union
from zipfile import ZipFile
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io
import requests
import kagglehub
from tqdm import tqdm

from ..configs.paths import DATA_RAW_PATH, ucf101_annotations_parent_folder, hmdb51_root


def download_drive_folder(folder_id, dest_path, max_videos=None, creds_file=None):
    if creds_file is None:
        from ..configs.paths import google_credential_file_path
        creds_file = google_credential_file_path

    creds = service_account.Credentials.from_service_account_file(
        creds_file, scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build('drive', 'v3', credentials=creds)
    os.makedirs(dest_path, exist_ok=True)

    # List all files in the folder
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])

    # Separate videos folder
    videos = [f for f in files if f['name'] == 'videos' and f['mimeType'] == 'application/vnd.google-apps.folder']
    others = [f for f in files if f not in videos]
    # Download non-video files
    for file in tqdm(others, desc="Non video files"):
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            subfolder_path = os.path.join(dest_path, file['name'])
            os.makedirs(subfolder_path, exist_ok=True)
            download_drive_folder(file['id'], subfolder_path, creds_file=creds_file)
            continue
        file_path = os.path.join(dest_path, file['name'])
        if not os.path.exists(file_path):
            request = service.files().get_media(fileId=file['id'])
            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    # Download videos (with limit)
    if videos:
        vid_folder_id = videos[0]['id']
        vquery = f"'{vid_folder_id}' in parents and trashed=false"
        vresults = service.files().list(q=vquery, fields="files(id, name)", pageSize=max_videos).execute()
        vfiles = vresults.get('files', [])[:max_videos] if max_videos else vresults.get('files', [])

        vpath = os.path.join(dest_path, 'videos')
        os.makedirs(vpath, exist_ok=True)

        for file in tqdm(vfiles, desc="Video files", total=max_videos):
            vfile_path = os.path.join(vpath, file['name'])
            if not os.path.exists(vfile_path):
                request = service.files().get_media(fileId=file['id'])
                fh = io.FileIO(vfile_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

    return f"Downloaded to {dest_path}"

def download_and_extract(
    url: str,
    dest_dir: str,
    zip_name: str
) -> str:
    """Download and unzip UCF101 train/test split files."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"{zip_name}")

    # Download
    if not os.path.exists(zip_path):
        with requests.get(url, stream=True, verify=False) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    # Extract
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    return dest_dir

def download_kaggle_dataset(dataset_name: str, dest_path: Optional[str] = None,) -> Union[str, Path]:
    """Download a kaggle dataset to a specified parent folder, defaults to the hpc scratch path under data/raw/"""
    dest_path= DATA_RAW_PATH if dest_path is None else dest_path
    os.environ["KAGGLEHUB_CACHE"] = dest_path
    dataset_path = kagglehub.dataset_download(dataset_name, )
    if dataset_name == "pypiahmad/realistic-action-recognition-ucf50-dataset":
        dataset_path = Path(dataset_path) / "UCF11_updated_mpg"
        return_path = Path(dest_path) / dataset_name / "pytorch_format"
        return_path.mkdir(parents=True, exist_ok=True)

        for class_folder in Path(dataset_path).iterdir():
            
            if class_folder.is_dir():
                class_name = class_folder.name
                class_target = return_path / class_name
                class_target.mkdir(parents=True, exist_ok=True)
                print(class_target)
                # Move all videos from group subfolders into one folder per class
                for group_folder in class_folder.iterdir():
                    if group_folder.is_dir():
                        for video_file in group_folder.glob("*.mpg"):
                            shutil.copy(video_file, class_target / video_file.name)
        print(f"PyTorch format dataset created at: {return_path}")
    elif dataset_name == "pevogam/ucf101":
        download_and_extract(
            url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip",
            dest_dir = ucf101_annotations_parent_folder,
            zip_name="UCF101-Annotations.zip"
        )
        return_path = dataset_path
    elif dataset_name == "easonlll/hmdb51":
        # Will download 
        path = kagglehub.dataset_download("easonlll/hmdb51")
        shutil.move(path, dest_path)
        print(f"Dataset downloaded to {dest_path}")
    else:
        print(f"No processing implemented for {dataset_name}, saved to {dataset_path}")
        return_path = dataset_path
    
    return return_path

