#!/bin/bash
# Adapted from https://github.com/cvdfoundation/kinetics-dataset
# Usage:
#   ./src/data/kinetics-scripts/k400-download.sh [train|val|test|all] [root_dl] [root_dl_targz]
# Example:
#   ./src/data/kinetics-scripts/k400-download.sh train /data/k400 /data/k400_targz

split=$1
root_dl=${2:-"/work3/fasco/data/raw/kinetics400"} # "/work3/fasco/data/raw/kinetics400"
root_dl_targz=${3:-"/localnvme/fasco/data/kinetics400_targz"} # "/work3/fasco/data/raw/kinetics400_targz"

mkdir -p "$root_dl" "$root_dl_targz"

# --- Start by downloading extras ---
curr_dl="${root_dl}/annotations"
mkdir -p "$curr_dl"
wget -nc https://s3.amazonaws.com/kinetics/400/annotations/train.csv -P "$curr_dl"
wget -nc https://s3.amazonaws.com/kinetics/400/annotations/val.csv -P "$curr_dl"
wget -nc https://s3.amazonaws.com/kinetics/400/annotations/test.csv -P "$curr_dl"

wget -nc http://s3.amazonaws.com/kinetics/400/readme.md -P "$root_dl"


download_split () {
  curr_dl=$1
  url=$2
  mkdir -p "$curr_dl"
  wget -nc -i "$url" -P "$curr_dl"

  # verify tarballs
  for f in "$curr_dl"/*.tar; do
    [ -f "$f" ] || continue
    if ! tar -tf "$f" >/dev/null 2>&1; then
      echo "Corrupted: $f → re-downloading..."
      rm -f "$f"
      wget -c -i "$url" -P "$curr_dl"
    fi
  done
}

case "$split" in
  train)
    download_split "${root_dl_targz}/train" "https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt"
    ;;
  val)
    download_split "${root_dl_targz}/val" "https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt"
    ;;
  test)
    download_split "${root_dl_targz}/test" "https://s3.amazonaws.com/kinetics/400/test/k400_test_path.txt"
    ;;
  all)
    download_split "${root_dl_targz}/train" "https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt"
    download_split "${root_dl_targz}/val" "https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt"
    download_split "${root_dl_targz}/test" "https://s3.amazonaws.com/kinetics/400/test/k400_test_path.txt"
    ;;
  *)
    echo "Usage: $0 [train|val|test|all] [root_dl] [root_dl_targz]"
    exit 1
    ;;
esac

curr_dl="${root_dl_targz}/replacement"
mkdir -p "$curr_dl"
wget -nc "https://s3.amazonaws.com/kinetics/400/replacement_for_corrupted_k400.tgz" -P "$curr_dl"
echo -e "\nDone! Data in: $root_dl, Tars in: $root_dl_targz"
