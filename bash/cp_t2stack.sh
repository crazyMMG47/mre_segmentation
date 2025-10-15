SRC="/home/smooi/Desktop/toast/data/toast_pipe_data"
DEST="/home/smooi/Desktop/mre_noise_unprocessed"

shopt -s nullglob
for d in "$DEST"/*/; do
  sid="$(basename "$d")"

  cand1="$SRC/$sid/${sid}_t2stack.nii"
  cand2="$SRC/$sid/t2stack.nii"
  cand3="$SRC/$sid/${sid}_t2stack.nii.gz"
  cand4="$SRC/$sid/t2stack.nii.gz"

  src=""
  for c in "$cand1" "$cand2" "$cand3" "$cand4"; do
    [[ -f "$c" ]] && { src="$c"; break; }
  done

  if [[ -n "$src" ]]; then
    cp -v "$src" "$d"
  else
    echo "Missing t2stack for $sid" >&2
  fi
done
