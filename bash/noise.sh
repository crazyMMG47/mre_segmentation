#!/usr/bin/env bash
set -euo pipefail

CSV="$HOME/Desktop/toast/subject_lookup.csv"           
SRC="$HOME/Desktop/imgraw_helen"             # your imgraw root
DEST="$HOME/Desktop/mre_noise_unprocessed"       # where to copy

mkdir -p "$DEST"

# Skip header, read SubjectID,OrigCode
tail -n +2 "$CSV" | while IFS=, read -r SID OC; do
  # trim quotes/spaces
  SID="${SID//\"/}"; SID="${SID//[$'\r\t ']/}"
  OC="${OC//\"/}";   OC="${OC//[$'\r\t ']/}"

  folder=""

  # --- Mapping rules from OrigCode -> folder name in imgraw ---
  if [[ "$OC" =~ ^[GS]_Nemours_([0-9]{4})$ ]]; then
    folder="Nemours_MRE_${BASH_REMATCH[1]}"
  elif [[ "$OC" =~ ^[GS]_MreVol_([0-9]+)$ ]]; then
    folder="Mre_Vol_${BASH_REMATCH[1]}"
  elif [[ "$OC" =~ ^[GS]_Cc_([0-9]+)$ ]]; then
    folder="Cc${BASH_REMATCH[1]}"
  elif [[ "$OC" =~ ^[GS]_Cc_([0-9]+)(Pre|Post)$ ]]; then
    folder="Cc${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
  elif [[ "$OC" =~ ^[GS]_a?EA_([0-9]+[A-Z]?)$ ]]; then
    folder="EA${BASH_REMATCH[1]}"
  elif [[ "$OC" =~ ^[GS]_aSTE_([0-9]{3})$ ]]; then
    folder="STE_${BASH_REMATCH[1]}_scan1"
  # (Optional) U01 if you ever need it:
  elif [[ "$OC" =~ ^[GS]_a?U01_([0-9]+)$ ]]; then
    folder="U01_${BASH_REMATCH[1]}"
  else
    continue  # unhandled code → skip
  fi

  if [[ -d "$SRC/$folder" ]]; then
    echo "cp -r \"$SRC/$folder\" \"$DEST/$SID\""
    cp -r "$SRC/$folder" "$DEST/$SID"
  fi
done
