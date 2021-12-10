#!/bin/bash
for ((i = 1 ; i <= $1; i++)); do
    python generate_users.py -o $2$i.json -p "video_qual,auto_task,playbooks,rank_alert,colab,ticketing,ingest" -r "{\"video_qual\": [1, 3]}" -i $3 -c "auto_task,playbooks,rank_alert,colab,ticketing,ingest" --clear-winners
done
